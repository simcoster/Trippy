"""Stdlib HTTP server for the price sandbox container."""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .ast_check import PriceFunctionError, compile_quote
from .execute import run_quote
from .params import QuoteParams

MAX_FUNCTIONS = 30
MAX_BATCH = 30
MAX_SOURCE_BYTES = 64_000
DEFAULT_PORT = 8503

_functions: dict[int, str] = {}
_sha256: dict[int, str] = {}


def load_functions(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Replace the in-memory set. Rejects anything the AST allowlist forbids."""
    global _functions, _sha256
    if len(entries) > MAX_FUNCTIONS:
        return {"ok": False, "error": f"at most {MAX_FUNCTIONS} functions"}
    approved: dict[int, str] = {}
    hashes: dict[int, str] = {}
    rejected: list[dict[str, Any]] = []
    for item in entries:
        try:
            site_id = int(item["site_id"])
            source = str(item.get("source") or "")
            digest = str(item.get("sha256") or "")
        except (KeyError, TypeError, ValueError):
            rejected.append({"error": "invalid entry"})
            continue
        if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
            rejected.append({"site_id": site_id, "error": "source too large"})
            continue
        try:
            compile_quote(source)
        except PriceFunctionError as exc:
            rejected.append({"site_id": site_id, "error": str(exc)})
            continue
        approved[site_id] = source
        hashes[site_id] = digest
    _functions = approved
    _sha256 = hashes
    return {
        "ok": True,
        "loaded": sorted(approved),
        "rejected": rejected,
    }


def quote_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    if len(items) > MAX_BATCH:
        return {"ok": False, "error": f"at most {MAX_BATCH} quotes per request"}
    results: list[dict[str, Any]] = []
    for item in items:
        request_id = item.get("id")
        try:
            site_id = int(item["site_id"])
        except (KeyError, TypeError, ValueError):
            results.append({"id": request_id, "ok": False, "error": "site_id required"})
            continue
        source = _functions.get(site_id)
        if source is None:
            results.append({"id": request_id, "ok": False, "error": "unknown site"})
            continue
        try:
            params = QuoteParams.from_mapping(item.get("params") or {})
            quoted = run_quote(source, params)
        except Exception as exc:
            results.append({"id": request_id, "ok": False, "error": str(exc)})
            continue
        results.append(
            {
                "id": request_id,
                "ok": True,
                "site_id": site_id,
                "price": quoted.price,
                "explanation": quoted.explanation,
            }
        )
    return {"ok": True, "results": results}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("[price-sandbox] " + (format % args) + "\n")

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0 or length > 2_000_000:
            return None
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            self._write_json(
                200,
                {"ok": True, "loaded": len(_functions)},
            )
            return
        self._write_json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
        except json.JSONDecodeError:
            self._write_json(400, {"ok": False, "error": "invalid json"})
            return
        if path == "/load":
            entries = []
            if isinstance(payload, dict):
                entries = list(payload.get("functions") or [])
            self._write_json(200, load_functions(entries))
            return
        if path == "/quote":
            items = []
            if isinstance(payload, dict):
                items = list(payload.get("quotes") or [])
            self._write_json(200, quote_batch(items))
            return
        self._write_json(404, {"ok": False, "error": "not found"})


def serve(host: str = "0.0.0.0", port: int = DEFAULT_PORT) -> None:
    httpd = ThreadingHTTPServer((host, port), _Handler)
    httpd.serve_forever()


def main() -> None:
    port = int(os.environ.get("PRICE_SANDBOX_PORT") or DEFAULT_PORT)
    serve(port=port)


if __name__ == "__main__":
    main()
