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
from .server import MAX_BATCH

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
    parent_site_id: int | None = None


@dataclass(frozen=True)
class QuoteReply:
    request_id: str
    ok: bool
    price: float | None = None
    explanation: str = ""
    error: str | None = None


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


def _replies_from_payload(
    requests: list[QuoteRequest], payload: dict[str, Any]
) -> list[QuoteReply]:
    if not payload.get("ok") and payload.get("error"):
        err = str(payload["error"])
        return [
            QuoteReply(request_id=item.request_id, ok=False, error=err)
            for item in requests
        ]
    by_id: dict[str, dict[str, Any]] = {}
    for row in payload.get("results") or []:
        if isinstance(row, dict) and row.get("id") is not None:
            by_id[str(row["id"])] = row
    out: list[QuoteReply] = []
    for item in requests:
        row = by_id.get(item.request_id)
        if row is None:
            out.append(
                QuoteReply(request_id=item.request_id, ok=False, error="no_result")
            )
        elif row.get("ok"):
            out.append(
                QuoteReply(
                    request_id=item.request_id,
                    ok=True,
                    price=float(row["price"]),
                    explanation=str(row.get("explanation") or ""),
                )
            )
        else:
            out.append(
                QuoteReply(
                    request_id=item.request_id,
                    ok=False,
                    error=str(row.get("error") or "quote_failed"),
                )
            )
    return out


def quote_replies(
    requests: list[QuoteRequest],
    *,
    base_url: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> list[QuoteReply]:
    """Every request in order, including jail errors. Chunks at MAX_BATCH."""
    url = (base_url or sandbox_url() or "").rstrip("/")
    if not url:
        raise RuntimeError("PRICE_SANDBOX_URL is not set")
    endpoint = urljoin(url + "/", "quote")
    out: list[QuoteReply] = []
    for start in range(0, len(requests), MAX_BATCH):
        chunk = requests[start : start + MAX_BATCH]
        payload = _post(
            endpoint,
            {
                "quotes": [
                    {
                        "id": item.request_id,
                        "site_id": item.site_id,
                        "parent_site_id": item.parent_site_id,
                        "params": item.params.to_json(),
                    }
                    for item in chunk
                ]
            },
            timeout_s=timeout_s,
        )
        out.extend(_replies_from_payload(chunk, payload))
    return out


def quote_via_sandbox(
    requests: list[QuoteRequest],
    *,
    base_url: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, QuoteResult]:
    """Return successful quotes keyed by request_id. Connection errors raise."""
    out: dict[str, QuoteResult] = {}
    for reply in quote_replies(
        requests, base_url=base_url, timeout_s=timeout_s
    ):
        if reply.ok and reply.price is not None:
            out[reply.request_id] = QuoteResult(
                price=reply.price, explanation=reply.explanation
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
