"""Start, poll, and cancel scrape/clear CLIs.

Does not import scraper modules. The jobs Compose service runs these as
subprocesses; Streamlit talks to that service over HTTP when JOBS_URL is set.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs" / "jobs"
LOCK_PATH = LOG_DIR / "current.json"

Extra = Literal["site", "campsite"] | None


@dataclass(frozen=True)
class JobSpec:
    id: str
    label: str
    group: Literal["scrape", "clear"]
    steps: tuple[tuple[str, ...], ...]
    extra: Extra = None
    destructive: bool = False
    confirm: str = ""


def _py(*args: str) -> tuple[str, ...]:
    return ("python", *args)


JOBS: tuple[JobSpec, ...] = (
    JobSpec(
        id="scrape-sites",
        label="Discover sites",
        group="scrape",
        steps=(_py("source/scraper/discover_sites.py"),),
    ),
    JobSpec(
        id="scrape-rooms",
        label="Rooms",
        group="scrape",
        extra="site",
        steps=(_py("-m", "source.scraper.rules_ingest.rooms"),),
    ),
    JobSpec(
        id="scrape-prices",
        label="Prices",
        group="scrape",
        extra="site",
        steps=(_py("-m", "source.scraper.info_site.scrape", "--prices"),),
    ),
    JobSpec(
        id="scrape-rules",
        label="Rules",
        group="scrape",
        extra="site",
        steps=(_py("-m", "source.scraper.rules_ingest.ingest"),),
    ),
    JobSpec(
        id="scrape-info",
        label="Info (rooms → prices → rules)",
        group="scrape",
        extra="site",
        steps=(
            _py("-m", "source.scraper.rules_ingest.rooms"),
            _py("-m", "source.scraper.info_site.scrape", "--prices"),
            _py("-m", "source.scraper.rules_ingest.ingest"),
        ),
    ),
    JobSpec(
        id="scrape-availability",
        label="Availability",
        group="scrape",
        extra="site",
        steps=(_py("source/scraper/populate_availability.py"),),
    ),
    JobSpec(
        id="scrape-reviews",
        label="Reviews",
        group="scrape",
        extra="campsite",
        steps=(_py("-m", "source.scraper.populate_reviews_and_claims"),),
    ),
    JobSpec(
        id="populate-claims",
        label="Populate claims",
        group="scrape",
        extra="campsite",
        steps=(_py("-m", "source.scraper.populate_claims"),),
    ),
    JobSpec(
        id="scrape-all",
        label="Sites + info + availability",
        group="scrape",
        steps=(
            _py("source/scraper/discover_sites.py"),
            _py("-m", "source.scraper.rules_ingest.rooms"),
            _py("-m", "source.scraper.info_site.scrape", "--prices"),
            _py("-m", "source.scraper.rules_ingest.ingest"),
            _py("source/scraper/populate_availability.py"),
        ),
    ),
    JobSpec(
        id="update-tables",
        label="Alembic upgrade",
        group="scrape",
        steps=(("alembic", "upgrade", "head"),),
    ),
    JobSpec(
        id="clear-availability",
        label="Clear availability",
        group="clear",
        extra="site",
        destructive=True,
        confirm="Deletes availability rows. Types, rules, and prices stay.",
        steps=(_py("scripts/clear_availability.py"),),
    ),
    JobSpec(
        id="clear-reviews",
        label="Clear reviews",
        group="clear",
        destructive=True,
        confirm="Truncates reviews and claims. Campsites stay.",
        steps=(_py("scripts/clear_reviews_and_claims.py"),),
    ),
    JobSpec(
        id="clear-claims",
        label="Clear claims",
        group="clear",
        destructive=True,
        confirm="Deletes claims and nulls reviews.is_relevant. Review text stays.",
        steps=(_py("scripts/clear_claims.py"),),
    ),
    JobSpec(
        id="clear-info",
        label="Clear info",
        group="clear",
        destructive=True,
        confirm="Clears rules, prices, types, names, vocabulary, and availability.",
        steps=(_py("scripts/clear_info.py", "--yes"),),
    ),
    JobSpec(
        id="clear-rules",
        label="Clear rules",
        group="clear",
        extra="site",
        destructive=True,
        confirm="Deletes site-level campsite_rules. Per-unit rows and vocabulary stay.",
        steps=(_py("scripts/clear_rules.py"),),
    ),
)

JOBS_BY_ID = {job.id: job for job in JOBS}


def spec(job_id: str) -> JobSpec:
    try:
        return JOBS_BY_ID[job_id]
    except KeyError:
        raise KeyError(f"unknown job {job_id!r}") from None


def _pythonpath() -> str:
    scraper = str(ROOT / "source" / "scraper")
    return os.pathsep.join((str(ROOT), scraper))


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x00100000, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_lock() -> dict | None:
    try:
        data = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _write_lock(payload: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = LOCK_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(LOCK_PATH)


def status() -> dict | None:
    data = _read_lock()
    if not data:
        return None
    pid = int(data.get("pid") or 0)
    running = _pid_running(pid)
    data["running"] = running
    if not running and data.get("exit_code") is None:
        data["exit_code"] = -1
        data["finished_at"] = data.get("finished_at") or datetime.now(UTC).isoformat()
    return data


def log_tail(path: str | None, *, n: int = 80) -> str:
    if not path:
        return ""
    log = Path(path)
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def _extra_argv(job: JobSpec, site_id: int | None) -> tuple[str, ...]:
    if site_id is None or job.extra is None:
        return ()
    if job.extra == "campsite":
        return ("--campsite-id", str(site_id))
    return ("--site", str(site_id))


def start(job_id: str, *, site_id: int | None = None) -> dict:
    job = spec(job_id)
    current = status()
    if current and current.get("running"):
        raise RuntimeError(f"already running: {current.get('job_id')}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_DIR / f"{job.id}-{stamp}.log"
    extra = _extra_argv(job, site_id)
    argv = [
        sys.executable,
        "-m",
        "source.ops.jobs",
        "exec",
        job.id,
        *extra,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath()
    env["TRIPPY_JOB_LOG"] = str(log_path)
    log_path.touch()
    with log_path.open("ab") as log:
        proc = subprocess.Popen(
            argv,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    for _ in range(100):
        data = _read_lock()
        if data and int(data.get("pid") or 0) == proc.pid:
            return data
        if proc.poll() is not None:
            raise RuntimeError(
                log_tail(str(log_path)) or f"job {job.id} exited before it recorded a lock"
            )
        time.sleep(0.05)
    data = _read_lock()
    if data:
        return data
    raise RuntimeError(f"job {job.id} started (pid {proc.pid}) but lock was not written")


def cancel() -> None:
    current = status()
    if not current or not current.get("running"):
        raise RuntimeError("no running job")
    pid = int(current["pid"])
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            capture_output=True,
        )
    else:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                return
    deadline = time.time() + 8
    while time.time() < deadline and _pid_running(pid):
        time.sleep(0.2)
    if _pid_running(pid) and os.name != "nt":
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    data = _read_lock() or current
    data["running"] = False
    data["exit_code"] = data.get("exit_code") if data.get("exit_code") is not None else -15
    data["finished_at"] = datetime.now(UTC).isoformat()
    _write_lock(data)


def _step_argv(step: tuple[str, ...], extra: list[str]) -> list[str]:
    if step and step[0] == "alembic":
        return [sys.executable, "-m", "alembic", *step[1:]]
    if step and step[0] == "python":
        return [sys.executable, *step[1:], *extra]
    return [sys.executable, *step, *extra]


def _exec(job_id: str, extra: list[str]) -> int:
    job = spec(job_id)
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath()
    payload = {
        "job_id": job.id,
        "label": job.label,
        "pid": os.getpid(),
        "log_path": env.get("TRIPPY_JOB_LOG"),
        "started_at": datetime.now(UTC).isoformat(),
        "exit_code": None,
        "finished_at": None,
    }
    _write_lock(payload)
    code = 0
    try:
        for step in job.steps:
            argv = _step_argv(step, extra)
            print("+", " ".join(argv), flush=True)
            result = subprocess.run(argv, cwd=ROOT, env=env, check=False)
            if result.returncode != 0:
                code = result.returncode
                return code
        return 0
    finally:
        data = _read_lock() or payload
        data["exit_code"] = code
        data["finished_at"] = datetime.now(UTC).isoformat()
        _write_lock(data)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 2 or args[0] != "exec":
        print(
            "usage: python -m source.ops.jobs exec <job-id> [--site N|--campsite-id N]",
            file=sys.stderr,
        )
        return 2
    return _exec(args[1], args[2:])


if __name__ == "__main__":
    raise SystemExit(main())
