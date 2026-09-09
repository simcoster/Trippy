"""Run evals/planner_v1.json through extractor + planner and score against gold.

    just run-eval
    just run-eval -- --ids E01,H02
    just run-eval -- --no-copy
    uv run python -m source.eval.run --ids E01,H02
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from db.connect import connect, database_url
from db.experiments import SEARCH_PATH, copy_public, table_name
from source.agent.graph import extractor_node, planner_node
from source.eval.score import score_case

load_dotenv()

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL = _ROOT / "evals" / "planner_v1.json"
# Occupancy for the benchmark is availability_frozen, not this table.
EVAL_COPY_SKIP = ("availability",)


def _content(msg) -> str:
    raw = getattr(msg, "content", "")
    return raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)


def _as_dict(msg) -> dict | None:
    try:
        data = json.loads(_content(msg))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def apply_run_env(spec: dict) -> None:
    env = (spec.get("run") or {}).get("env") or {}
    for key, value in env.items():
        os.environ[str(key)] = str(value)


def refresh_experiments_from_public(
    *, skip: Sequence[str] = EVAL_COPY_SKIP
) -> None:
    """Overwrite experiments from public. Skipped tables stay empty clones."""
    skipped = ", ".join(skip) or "none"
    print(f"copying public → experiments (skip {skipped})", flush=True)
    with connect(database_url(), options=SEARCH_PATH) as conn:
        with conn.cursor() as cur:
            copy_public(cur, skip=skip)
        conn.commit()
    print("copy done.", flush=True)


def _require_frozen(table: str) -> None:
    rel = table_name(table)
    with connect(database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS ("
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = current_schema() "
                "AND table_name = %s"
                ")",
                (rel,),
            )
            if not cur.fetchone()[0]:
                raise SystemExit(
                    f"{rel} is missing. Run: just setup-experiments freeze-availability"
                )
            cur.execute(f"SELECT count(*) FROM {rel}")
            n = cur.fetchone()[0]
    print(f"{rel}: {n} row(s)", flush=True)


def run_one(query: str) -> tuple[dict | None, dict | None]:
    """Extractor then planner. Skips the light/cleaner node."""
    messages = [HumanMessage(content=query)]
    extracted = extractor_node({"messages": messages})
    extract_msgs = extracted.get("messages") or []
    extract = None
    for msg in extract_msgs:
        data = _as_dict(msg)
        if data is not None:
            extract = data
            break
    messages = messages + list(extract_msgs)
    planned = planner_node({"messages": messages})
    planner = None
    for msg in planned.get("messages") or []:
        data = _as_dict(msg)
        if data is not None and "fits" in data:
            planner = data
            break
    return extract, planner


def _summarize_planner(planner: dict | None) -> dict:
    if not planner:
        return {}
    fits = []
    for row in planner.get("fits") or []:
        why = []
        for entry in row.get("why") or []:
            if not isinstance(entry, dict):
                continue
            why.append(
                {
                    "query": entry.get("query"),
                    "site_amenity": entry.get("site_amenity"),
                    "stated_amenity": entry.get("stated_amenity"),
                    "claim": entry.get("claim"),
                    "reason": entry.get("reason"),
                }
            )
        fits.append(
            {
                "campsite_id": row.get("campsite_id"),
                "campsite": row.get("campsite"),
                "accommodation_type": row.get("accommodation_type"),
                "price_per_night": row.get("price_per_night"),
                "why": why,
            }
        )
    return {
        "fits_count": len(planner.get("fits") or []),
        "rejected_count": planner.get("rejected_count"),
        "skipped": planner.get("skipped"),
        "fits": fits,
    }


def write_report(path: Path, spec: dict, rows: list[dict], wall: float) -> None:
    passed = sum(1 for r in rows if r["score"]["ok"])
    easy_p = sum(
        1 for r in rows if r["difficulty"] == "easy" and r["score"]["ok"]
    )
    easy_n = sum(1 for r in rows if r["difficulty"] == "easy")
    hard_p = sum(
        1 for r in rows if r["difficulty"] == "hard" and r["score"]["ok"]
    )
    hard_n = sum(1 for r in rows if r["difficulty"] == "hard")
    lines = [
        f"# Planner eval `{spec.get('id')}`",
        "",
        f"- as_of: {spec.get('as_of')}",
        f"- wall: {wall:.1f}s",
        f"- pass: **{passed}/{len(rows)}** (easy {easy_p}/{easy_n}, hard {hard_p}/{hard_n})",
        f"- env: `TRIPPY_SCHEMA={os.environ.get('TRIPPY_SCHEMA')}` "
        f"`TRIPPY_AVAILABILITY_TABLE={os.environ.get('TRIPPY_AVAILABILITY_TABLE')}` "
        f"`TRIPPY_TODAY={os.environ.get('TRIPPY_TODAY')}`",
        "",
        "| id | diff | result | date | fit sites | failures |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        score = row["score"]
        mark = "PASS" if score["ok"] else "FAIL"
        date = score.get("extract_date") or {}
        date_s = f"{date.get('start', '')}→{date.get('end', '')}" if date else ""
        fails = "; ".join(score.get("failures") or []) or ""
        lines.append(
            f"| {row['id']} | {row['difficulty']} | {mark} | {date_s} | "
            f"{score.get('fit_sites')} | {fails} |"
        )
    fails = [r for r in rows if not r["score"]["ok"]]
    if fails:
        lines.extend(["", "## Failures", ""])
        for row in fails:
            lines.append(f"### {row['id']} — {row['query']}")
            lines.append("")
            for item in row["score"]["failures"]:
                lines.append(f"- {item}")
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", default=str(DEFAULT_EVAL), help="eval JSON")
    parser.add_argument("--ids", default="", help="comma-separated case ids")
    parser.add_argument("--out-dir", default="", help="report directory")
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Do not refresh experiments from public first",
    )
    args = parser.parse_args(argv)
    spec_path = Path(args.eval)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not args.no_copy:
        refresh_experiments_from_public()
    apply_run_env(spec)
    table = os.environ.get("TRIPPY_AVAILABILITY_TABLE") or "availability"
    print(f"TRIPPY_SCHEMA={os.environ.get('TRIPPY_SCHEMA')}", flush=True)
    print(f"TRIPPY_TODAY={os.environ.get('TRIPPY_TODAY')}", flush=True)
    _require_frozen(table)

    cases = list(spec.get("queries") or [])
    if args.ids:
        want = {part.strip() for part in args.ids.split(",") if part.strip()}
        cases = [c for c in cases if c.get("id") in want]
        missing = want - {c.get("id") for c in cases}
        if missing:
            raise SystemExit(f"unknown ids: {sorted(missing)}")
    if not cases:
        raise SystemExit("no cases")

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "reports" / "evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_md = out_dir / f"{stamp}.md"
    report_json = out_dir / f"{stamp}.json"

    rows: list[dict] = []
    started = time.monotonic()
    for case in cases:
        cid = str(case.get("id"))
        query = str(case.get("query") or "")
        print(f"\n=== {cid} {case.get('difficulty')} ===", flush=True)
        print(query, flush=True)
        t0 = time.monotonic()
        extract, planner = run_one(query)
        elapsed = time.monotonic() - t0
        score = score_case(case.get("expect") or {}, extract, planner)
        mark = "PASS" if score["ok"] else "FAIL"
        print(
            f"{mark} {elapsed:.1f}s date={score.get('extract_date')} "
            f"fits={score.get('fit_sites')} {score.get('failures')}",
            flush=True,
        )
        rows.append(
            {
                "id": cid,
                "difficulty": case.get("difficulty"),
                "query": query,
                "seconds": round(elapsed, 1),
                "score": score,
                "extract": extract,
                "planner": _summarize_planner(planner),
            }
        )
    wall = time.monotonic() - started
    dump = {
        "eval": spec.get("id"),
        "at": datetime.now().isoformat(timespec="seconds"),
        "seconds": round(wall, 1),
        "pass": sum(1 for r in rows if r["score"]["ok"]),
        "n": len(rows),
        "cases": rows,
    }
    report_json.write_text(
        json.dumps(dump, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    write_report(report_md, spec, rows, wall)
    passed = dump["pass"]
    print(f"\n{passed}/{len(rows)} pass in {wall:.1f}s", flush=True)
    print(f"report {report_md}", flush=True)
    print(f"dump   {report_json}", flush=True)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
