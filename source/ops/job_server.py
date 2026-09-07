"""HTTP wrapper around the job runner. Runs in the jobs Compose service."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from source.ops import jobs

app = FastAPI(title="Trippy jobs")


class StartBody(BaseModel):
    job_id: str
    site_id: int | None = Field(default=None)


def _with_tail(snap: dict | None) -> dict | None:
    if not snap:
        return None
    out = dict(snap)
    out["log_tail"] = jobs.log_tail(snap.get("log_path"))
    return out


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.get("/status")
def get_status() -> dict:
    snap = _with_tail(jobs.status())
    if snap is None:
        return {}
    return snap


@app.post("/jobs")
def post_job(body: StartBody) -> dict:
    try:
        return _with_tail(jobs.start(body.job_id, site_id=body.site_id)) or {}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/cancel")
def post_cancel() -> dict[str, bool]:
    try:
        jobs.cancel()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True}


def main() -> None:
    import uvicorn

    port = int(os.environ.get("JOBS_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
