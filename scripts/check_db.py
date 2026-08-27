"""Verify Docker Desktop and the Trippy Postgres container are running."""

from __future__ import annotations

import subprocess
import sys

POSTGRES_CONTAINER = "trippy-postgres"
DOCKER_TIMEOUT_S = 15


def _log(msg: str) -> None:
    print(msg, flush=True)


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=DOCKER_TIMEOUT_S,
    )


def check_docker_desktop() -> None:
    _log(f"Checking Docker Desktop (docker info, timeout {DOCKER_TIMEOUT_S}s)...")
    try:
        result = _run(["docker", "info"])
    except FileNotFoundError as exc:
        raise RuntimeError(
            "docker CLI not found on PATH. Is Docker Desktop installed?"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"docker info timed out after {DOCKER_TIMEOUT_S}s. "
            "Docker Desktop may be starting or stuck — open it and retry."
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        msg = "Docker Desktop is not running (or docker CLI is unavailable)."
        if detail:
            msg = f"{msg}\n{detail}"
        raise RuntimeError(msg)
    _log("  Docker Desktop OK.")


def check_postgres_container() -> None:
    _log(f"Checking container {POSTGRES_CONTAINER!r}...")
    try:
        result = _run(
            [
                "docker",
                "inspect",
                "-f",
                "{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}",
                POSTGRES_CONTAINER,
            ]
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"docker inspect timed out after {DOCKER_TIMEOUT_S}s."
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"Postgres container {POSTGRES_CONTAINER!r} not found. "
            f"Start it with: docker compose up -d db"
            + (f"\n{detail}" if detail else "")
        )

    parts = (result.stdout or "").strip().split()
    status = parts[0] if parts else ""
    health = parts[1] if len(parts) > 1 else ""
    _log(f"  status={status or 'unknown'} health={health or 'n/a'}")

    if status != "running":
        raise RuntimeError(
            f"Postgres container {POSTGRES_CONTAINER!r} is {status or 'unknown'}, "
            f"expected running. Start it with: docker compose up -d db"
        )
    if health and health not in ("healthy", "starting"):
        raise RuntimeError(
            f"Postgres container {POSTGRES_CONTAINER!r} health is {health!r}."
        )
    if health == "starting":
        raise RuntimeError(
            f"Postgres container {POSTGRES_CONTAINER!r} is still starting; "
            f"wait until healthy then retry."
        )


def check_db() -> None:
    check_docker_desktop()
    check_postgres_container()
    _log(f"OK: Docker Desktop up, {POSTGRES_CONTAINER} running.")


def main() -> None:
    try:
        check_db()
    except RuntimeError as exc:
        print(exc, file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
