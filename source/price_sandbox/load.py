"""Push site_price_functions into the price sandbox, then exit.

The sandbox has no Postgres. This process is the trusted loader: it reads
approved sources from the DB, POST /load, and dies. Streamlit / FastAPI
only quote. Re-run after scrape-prices, a sandbox restart, or compose up.
The jail server must not import this module.
"""

from __future__ import annotations

import argparse
import sys
import time

from dotenv import load_dotenv

from db.connect import connect, database_url
from source.price_sandbox.client import (
    LoadedFunction,
    load_into_sandbox,
    sandbox_reachable,
    sandbox_url,
)
from source.scraper import tls as _tls  # noqa: F401
from source.scraper.info_site.db import load_price_functions

DEFAULT_WAIT_S = 20.0


def wait_for_sandbox(
    *, wait_s: float = DEFAULT_WAIT_S, base_url: str | None = None
) -> bool:
    deadline = time.monotonic() + max(0.0, wait_s)
    while True:
        if sandbox_reachable(base_url=base_url):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.4)


def load_sandbox(*, base_url: str | None = None) -> dict:
    with connect(database_url()) as conn:
        rows = load_price_functions(conn)
    result = load_into_sandbox(
        [
            LoadedFunction(
                site_id=row.site_id, source=row.source, sha256=row.sha256
            )
            for row in rows
        ],
        base_url=base_url,
    )
    result["n_db"] = len(rows)
    return result


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--if-up",
        action="store_true",
        help="exit 0 if the sandbox is unreachable",
    )
    parser.add_argument(
        "--wait-s",
        type=float,
        default=DEFAULT_WAIT_S,
        help="seconds to wait for /health before giving up",
    )
    args = parser.parse_args(argv)
    url = sandbox_url()
    if not url:
        msg = "PRICE_SANDBOX_URL is not set"
        if args.if_up:
            print(f"price-sandbox loader: skip ({msg})", flush=True)
            return 0
        print(msg, file=sys.stderr)
        return 1
    if not wait_for_sandbox(wait_s=args.wait_s):
        msg = f"sandbox not reachable at {url}"
        if args.if_up:
            print(f"price-sandbox loader: skip ({msg})", flush=True)
            return 0
        print(msg, file=sys.stderr)
        return 1
    try:
        result = load_sandbox()
    except Exception as exc:
        print(f"price-sandbox loader failed: {exc}", file=sys.stderr)
        return 1
    if not result.get("ok"):
        print(
            f"price-sandbox loader: {result.get('error') or result}",
            file=sys.stderr,
        )
        return 1
    loaded = result.get("loaded") or []
    rejected = result.get("rejected") or []
    print(
        f"price-sandbox loader: loaded {len(loaded)} function(s) "
        f"from {result.get('n_db', 0)} row(s)",
        flush=True,
    )
    if rejected:
        print(f"    rejected: {rejected}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
