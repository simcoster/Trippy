"""HTTP client the loader and quote callers use to talk to the price sandbox."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

from .params import QuoteParams, QuoteResult

DEFAULT_TIMEOUT_S = 5.0


def sandbox_url() -> str | None:
    raw = (os.environ.get("PRICE_SANDBOX_URL") or "").strip()
    return raw or None


@dataclass(frozen=True)
class LoadedFunction:
    site_id: int
    source: str
    sha256: str


@dataclass(frozen=True)
class QuoteRequest:
    request_id: str
    site_id: int
    params: QuoteParams


def _post(url: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = response.read()
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError("sandbox returned a non-object")
    return parsed


def load_into_sandbox(
    functions: list[LoadedFunction],
    *,
    base_url: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    url = (base_url or sandbox_url() or "").rstrip("/")
    if not url:
        raise RuntimeError("PRICE_SANDBOX_URL is not set")
    return _post(
        urljoin(url + "/", "load"),
        {
            "functions": [
                {
                    "site_id": item.site_id,
                    "source": item.source,
                    "sha256": item.sha256,
                }
                for item in functions
            ]
        },
        timeout_s=timeout_s,
    )


def quote_via_sandbox(
    requests: list[QuoteRequest],
    *,
    base_url: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, QuoteResult]:
    """Return successful quotes keyed by request_id. Connection errors raise."""
    url = (base_url or sandbox_url() or "").rstrip("/")
    if not url:
        raise RuntimeError("PRICE_SANDBOX_URL is not set")
    payload = _post(
        urljoin(url + "/", "quote"),
        {
            "quotes": [
                {
                    "id": item.request_id,
                    "site_id": item.site_id,
                    "params": item.params.to_json(),
                }
                for item in requests
            ]
        },
        timeout_s=timeout_s,
    )
    out: dict[str, QuoteResult] = {}
    for row in payload.get("results") or []:
        if not isinstance(row, dict) or not row.get("ok"):
            continue
        request_id = str(row.get("id") or "")
        if not request_id:
            continue
        out[request_id] = QuoteResult(
            price=float(row["price"]),
            explanation=str(row.get("explanation") or ""),
        )
    return out


def sandbox_reachable(*, base_url: str | None = None) -> bool:
    url = (base_url or sandbox_url() or "").rstrip("/")
    if not url:
        return False
    try:
        with urllib.request.urlopen(urljoin(url + "/", "health"), timeout=1.5) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
