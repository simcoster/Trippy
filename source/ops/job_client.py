"""Streamlit talks to jobs through this module: HTTP if JOBS_URL, else local."""

from __future__ import annotations

import os

import httpx

from source.ops.jobs import JOBS, JobSpec, spec

__all__ = [
    "JOBS",
    "JobSpec",
    "cancel",
    "log_tail",
    "spec",
    "start",
    "status",
]


def _jobs_url() -> str | None:
    raw = os.environ.get("JOBS_URL", "").strip()
    return raw.rstrip("/") or None


def _raise_http(exc: httpx.HTTPStatusError) -> None:
    detail = exc.response.text
    try:
        payload = exc.response.json()
        detail = str(payload.get("detail") or detail)
    except ValueError:
        pass
    raise RuntimeError(detail) from exc


def start(job_id: str, *, site_id: int | None = None) -> dict:
    base = _jobs_url()
    if not base:
        from source.ops import jobs as local

        return local.start(job_id, site_id=site_id)
    with httpx.Client(timeout=15.0) as client:
        try:
            response = client.post(
                f"{base}/jobs",
                json={"job_id": job_id, "site_id": site_id},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            _raise_http(exc)
        except httpx.RequestError as exc:
            raise RuntimeError(f"jobs service unreachable: {exc}") from exc
        data = response.json()
        return data if isinstance(data, dict) else {}


def cancel() -> None:
    base = _jobs_url()
    if not base:
        from source.ops import jobs as local

        local.cancel()
        return
    with httpx.Client(timeout=15.0) as client:
        try:
            response = client.post(f"{base}/cancel")
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            _raise_http(exc)
        except httpx.RequestError as exc:
            raise RuntimeError(f"jobs service unreachable: {exc}") from exc


def status() -> dict | None:
    base = _jobs_url()
    if not base:
        from source.ops import jobs as local

        snap = local.status()
        if not snap:
            return None
        out = dict(snap)
        out["log_tail"] = local.log_tail(snap.get("log_path"))
        return out
    with httpx.Client(timeout=5.0) as client:
        try:
            response = client.get(f"{base}/status")
            response.raise_for_status()
        except httpx.RequestError:
            return {"running": False, "label": "jobs service unreachable", "log_tail": ""}
        except httpx.HTTPStatusError as exc:
            _raise_http(exc)
        data = response.json()
        if not data:
            return None
        return data if isinstance(data, dict) else None


def log_tail(path: str | None, *, n: int = 80) -> str:
    base = _jobs_url()
    if not base:
        from source.ops import jobs as local

        return local.log_tail(path, n=n)
    snap = status()
    if not snap:
        return ""
    return str(snap.get("log_tail") or "")
