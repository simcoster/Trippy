"""Run evals/planner_v1.json through extractor + planner and score against gold.

    just run-eval
    just run-eval -- --ids E01,H02
    just run-eval -- --no-copy
    just run-eval -- --model 30B
    just run-eval -- --judge-concurrency 1
    just run-eval -- --no-judge-compact
    just run-eval -- --recommender
    just run-eval -- --judge-batch --judge-model glm
    uv run python -m source.eval.run --from-json reports/evals/2026-09-10_185136.json
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
from source.agent.claim_judge import (
    judge_batch,
    judge_compact,
    judge_concurrency,
    judge_model,
)
from source.agent.graph import extractor_node, planner_node
from source.agent.recommender import recommend_from_payload, recommendation_row
from source.agent.timing import (
    STAGE_ORDER,
    collect_stages,
    format_stages,
    merge_snapshots,
)
from source.eval.score import score_case
from source.scraper.amenity_enrichment.llm import (
    chat_usd_per_mtok,
    collect_llm_usage,
    embed_usd_per_mtok,
)

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


def format_usage_line(usage: dict | None) -> str:
    if not usage:
        return ""
    inn = int(usage.get("input_tokens") or 0)
    out = int(usage.get("output_tokens") or 0)
    if inn <= 0 and out <= 0:
        return ""
    bits = [f"tokens in={inn} out={out}"]
    for bucket in usage.get("by_role") or []:
        role = str(bucket.get("role") or "")
        bi = int(bucket.get("input_tokens") or 0)
        bo = int(bucket.get("output_tokens") or 0)
        n = int(bucket.get("calls") or 0)
        if n > 1:
            bits.append(f"{role} in={bi} out={bo}×{n}")
        elif n > 0 or bi or bo:
            bits.append(f"{role} in={bi} out={bo}")
    return " ".join(bits)


def _usage_totals(rows: list[dict]) -> dict:
    inn = 0
    out = 0
    roles: dict[str, dict[str, int]] = {}
    for row in rows:
        usage = row.get("usage") or {}
        inn += int(usage.get("input_tokens") or 0)
        out += int(usage.get("output_tokens") or 0)
        for bucket in usage.get("by_role") or []:
            role = str(bucket.get("role") or "")
            slot = roles.setdefault(
                role, {"input_tokens": 0, "output_tokens": 0, "calls": 0}
            )
            slot["input_tokens"] += int(bucket.get("input_tokens") or 0)
            slot["output_tokens"] += int(bucket.get("output_tokens") or 0)
            slot["calls"] += int(bucket.get("calls") or 0)
    return {
        "input_tokens": inn,
        "output_tokens": out,
        "by_role": [{"role": role, **slot} for role, slot in roles.items()],
    }


def _usage_role_cell(usage: dict | None, role: str) -> str:
    for bucket in (usage or {}).get("by_role") or []:
        if bucket.get("role") != role:
            continue
        inn = int(bucket.get("input_tokens") or 0)
        out = int(bucket.get("output_tokens") or 0)
        n = int(bucket.get("calls") or 0)
        if inn <= 0 and out <= 0:
            return ""
        if n > 1:
            return f"{inn}/{out}×{n}"
        return f"{inn}/{out}"
    return ""


_COST_STEPS: tuple[tuple[str, str], ...] = (
    ("extract", "extract"),
    ("embed", "embed"),
    ("claim_judge", "judge"),
    ("recommend", "recommend"),
)


def _md_cell(value: object) -> str:
    return " ".join(str(value or "").split()).replace("|", "\\|")


def _bucket_cost_usd(bucket: dict) -> float:
    raw = bucket.get("cost_usd")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    inn = int(bucket.get("input_tokens") or 0)
    out = int(bucket.get("output_tokens") or 0)
    model = bucket.get("model")
    if (bucket.get("kind") or "chat") == "embed":
        return inn * embed_usd_per_mtok(model) / 1_000_000
    usd_in, usd_out = chat_usd_per_mtok(model)
    return (inn * usd_in + out * usd_out) / 1_000_000


def _role_cost_usd(usage: dict | None, role: str) -> float:
    total = 0.0
    for bucket in (usage or {}).get("by_role") or []:
        if bucket.get("role") == role:
            total += _bucket_cost_usd(bucket)
    return total


def _fmt_usd(value: float) -> str:
    if value <= 0:
        return ""
    return f"${value:.4f}"


def _price_cell(value: object) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return f"₪{int(number)}"
    return f"₪{number:g}"


def _rec_pick_cell(item: dict) -> str:
    name = item.get("campsite") or ""
    typ = item.get("accommodation_type") or ""
    title = " — ".join(p for p in (name, typ) if p) or str(item.get("campsite_id") or "")
    price = _price_cell(item.get("price_per_night"))
    if price:
        title = f"{title} ({price})"
    return title


def run_one(
    query: str, *, recommender: bool = False
) -> tuple[dict | None, dict | None, dict | None, dict, dict]:
    """Extractor then planner. Optional recommender. Skips the light/cleaner node."""
    with collect_stages() as clock, collect_llm_usage() as usage:
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
        recommend = None
        if recommender and planner is not None:
            result = recommend_from_payload(query, planner)
            recommend = {
                "recommendations": [
                    recommendation_row(row) for row in result.recommendations
                ],
                "empty": result.empty,
                "text": result.text,
            }
        usage_d = (
            usage.report("eval")
            if usage.chat_calls or usage.embed_calls
            else {}
        )
        return extract, planner, recommend, clock.snapshot(), usage_d


def _summarize_planner(planner: dict | None) -> dict:
    if not planner:
        return {}
    return {
        "fits_count": len(planner.get("fits") or []),
        "rejected_count": planner.get("rejected_count"),
        "skipped": planner.get("skipped"),
        "fits": [_summarize_fit(row) for row in planner.get("fits") or []],
        "rejected": [
            _summarize_fit(row) for row in planner.get("rejected") or []
        ],
    }


def _summarize_fit(row: dict) -> dict:
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
                "detail": entry.get("detail"),
            }
        )
    out = {
        "campsite_id": row.get("campsite_id"),
        "campsite": row.get("campsite"),
        "accommodation_type": row.get("accommodation_type"),
        "price_per_night": row.get("price_per_night"),
        "why": why,
    }
    if row.get("booking_url"):
        out["booking_url"] = row["booking_url"]
    if row.get("retrieved"):
        out["retrieved"] = row["retrieved"]
    if row.get("claim_judge"):
        out["claim_judge"] = [
            {
                "query": v.get("query"),
                "satisfies": v.get("satisfies"),
                "satisfy_by": v.get("satisfy_by"),
                "relevant_claims": v.get("relevant_claims"),
                "reason": v.get("reason"),
            }
            for v in row["claim_judge"]
            if isinstance(v, dict)
        ]
    return out


def _fmt_numeric(item: dict) -> str:
    return f"{item.get('field')} {item.get('operator')} {item.get('value')}"


def _fmt_semantic(item: dict | str) -> str:
    if isinstance(item, str):
        return item
    if str(item.get("op") or "").lower() == "or":
        values = " | ".join(str(v) for v in item.get("values") or [])
        locus = item.get("locus") or "site"
        return f"OR({values}) ({locus})"
    query = item.get("query") or ""
    locus = item.get("locus") or "site"
    return f"{query} ({locus})" if query else ""


def _extract_lines(extract: dict | None) -> list[str]:
    if not extract:
        return ["- extract: (none)"]
    lines = ["- extract:"]
    date = extract.get("date") or {}
    if isinstance(date, dict) and date.get("start"):
        lines.append(f"  - date: {date.get('start')} → {date.get('end')}")
    campsite = extract.get("campsite")
    if campsite:
        lines.append(f"  - campsite: {campsite}")
    nums = [
        _fmt_numeric(item)
        for item in extract.get("numeric_constraints") or []
        if isinstance(item, dict)
    ]
    lines.append(f"  - numeric: {'; '.join(nums) if nums else '(none)'}")
    sem = [
        _fmt_semantic(item)
        for item in extract.get("semantic_constraints") or []
        if item
    ]
    sem = [s for s in sem if s]
    lines.append(f"  - semantic: {'; '.join(sem) if sem else '(none)'}")
    return lines


def _planner_queries(row: dict) -> list[str]:
    found: list[str] = []
    extract = row.get("extract") or {}
    for item in extract.get("semantic_constraints") or []:
        if isinstance(item, dict) and str(item.get("op") or "").lower() == "or":
            found.extend(str(v) for v in item.get("values") or [] if v)
        elif isinstance(item, dict) and item.get("query"):
            found.append(str(item["query"]))
        elif isinstance(item, str) and item.strip():
            found.append(item.strip())
    planner = row.get("planner") or {}
    for fit in list(planner.get("fits") or []) + list(planner.get("rejected") or []):
        for entry in fit.get("why") or []:
            if not isinstance(entry, dict):
                continue
            query = entry.get("query")
            if isinstance(query, list):
                found.extend(str(v) for v in query if v)
            elif isinstance(query, str) and query.strip():
                found.append(query.strip())
        for rec in fit.get("retrieved") or []:
            query = rec.get("query")
            if isinstance(query, str) and query.strip():
                found.append(query.strip())
    return list(dict.fromkeys(found))


def _listing_hits(rows: list[dict]) -> list[str]:
    hits: list[str] = []
    for row in rows:
        for entry in row.get("why") or []:
            if not isinstance(entry, dict):
                continue
            query = entry.get("query")
            if isinstance(query, list):
                query = query[0] if query else None
            hit = (
                entry.get("stated_amenity")
                or entry.get("site_amenity")
                or entry.get("claim")
            )
            if query and hit:
                hits.append(f"{query} → {hit}")
    return list(dict.fromkeys(hits))


def _site_trace_lines(fits: list[dict], *, dropped: bool = False) -> list[str]:
    by_site: dict[object, list[dict]] = {}
    order: list[object] = []
    for fit in fits:
        cid = fit.get("campsite_id")
        if cid not in by_site:
            by_site[cid] = []
            order.append(cid)
        by_site[cid].append(fit)
    lines: list[str] = []
    for cid in order:
        rows = by_site[cid]
        name = rows[0].get("campsite") or ""
        prefix = "dropped " if dropped else ""
        lines.append(f"- {prefix}site {cid} {name}".rstrip())
        types: list[str] = []
        for row in rows:
            typ = str(row.get("accommodation_type") or "")
            price = row.get("price_per_night")
            if typ and price is not None:
                types.append(f"{typ} ({price})")
            elif typ:
                types.append(typ)
        if types:
            lines.append("  - types: " + "; ".join(types))
        listing = _listing_hits(rows)
        if listing:
            lines.append("  - listing: " + "; ".join(listing))
        retrieved = rows[0].get("retrieved") or []
        for rec in retrieved:
            query = rec.get("query")
            claims = rec.get("claims") or []
            rules = rec.get("rules") or []
            if claims:
                claim_s = ", ".join(
                    str(c.get("claim"))
                    + (
                        ""
                        if c.get("is_positive") is None
                        else f" pos={c.get('is_positive')}"
                    )
                    for c in claims
                )
            else:
                claim_s = "(none)"
            if rules:
                rule_bits = []
                for rule in rules:
                    bit = str(rule.get("subject") or "")
                    if rule.get("polarity") is not None:
                        bit += f" pol={rule.get('polarity')}"
                    span = rule.get("evidence_span")
                    if span:
                        bit += f" “{span}”"
                    rule_bits.append(bit)
                rule_s = ", ".join(rule_bits)
            else:
                rule_s = "(none)"
            lines.append(f"  - RAG `{query}` claims: {claim_s}")
            lines.append(f"    rules: {rule_s}")
        for verdict in rows[0].get("claim_judge") or []:
            query = verdict.get("query")
            rel = verdict.get("relevant_claims") or []
            rel_s = ", ".join(str(x) for x in rel) or "(none)"
            by = verdict.get("satisfy_by")
            lines.append(
                f"  - judge `{query}` satisfies={verdict.get('satisfies')} "
                f"by={by} relevant: {rel_s}"
            )
            reason = verdict.get("reason")
            if reason:
                lines.append(f"    reason: {reason}")
    return lines


def _case_trace_lines(row: dict) -> list[str]:
    score = row["score"]
    mark = "PASS" if score["ok"] else "FAIL"
    lines = [f"### {row['id']} {mark} — {row['query']}", ""]
    fails = score.get("failures") or []
    if fails:
        for item in fails:
            lines.append(f"- fail: {item}")
    lines.extend(_extract_lines(row.get("extract")))
    usage_s = format_usage_line(row.get("usage"))
    if usage_s:
        lines.append(f"- {usage_s}")
    queries = _planner_queries(row)
    lines.append(
        "- planner queries: " + (", ".join(queries) if queries else "(none)")
    )
    planner = row.get("planner") or {}
    if planner.get("skipped"):
        lines.append(f"- skipped: {planner['skipped']}")
    fits = planner.get("fits") or []
    rejected = planner.get("rejected") or []
    if fits:
        lines.extend(_site_trace_lines(fits))
    if rejected:
        lines.extend(_site_trace_lines(rejected, dropped=True))
    if not fits and not rejected and not planner.get("skipped"):
        lines.append("- fits: (none)")
    rec = row.get("recommend") or {}
    recs = rec.get("recommendations") or []
    if recs or rec.get("empty") or rec.get("text"):
        lines.append("- recommendations:")
        if rec.get("empty"):
            lines.append(f"  - empty: {rec['empty']}")
        for item in recs:
            if not isinstance(item, dict):
                continue
            cid = item.get("campsite_id")
            name = item.get("campsite") or ""
            typ = item.get("accommodation_type") or ""
            start = item.get("start") or ""
            end = item.get("end") or ""
            price = item.get("price_per_night")
            bits = [str(cid)]
            if name:
                bits.append(str(name))
            if typ:
                bits.append(str(typ))
            if start:
                stay = str(start) if not end or end == start else f"{start}→{end}"
                bits.append(stay)
            if price is not None:
                bits.append(str(price))
            lines.append("  - " + " ".join(bits))
            why = item.get("why")
            if why:
                lines.append(f"    why: {why}")
        text = rec.get("text")
        if text:
            lines.append("  - reply:")
            for line in str(text).splitlines() or [""]:
                lines.append(f"    {line}")
    lines.append("")
    return lines


def _stage_cell(item: dict | None) -> str:
    if not item:
        return ""
    seconds = float(item.get("s") or 0)
    n = int(item.get("n") or 0)
    if n <= 0 and seconds <= 0:
        return ""
    if n > 1:
        return f"{seconds:.1f}×{n}"
    return f"{seconds:.1f}"


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
        f"`TRIPPY_TODAY={os.environ.get('TRIPPY_TODAY')}` "
        f"`TRIPPY_JUDGE_COMPACT={int(judge_compact())}` "
        f"`TRIPPY_JUDGE_CONCURRENCY={judge_concurrency()}` "
        f"`TRIPPY_JUDGE_BATCH={int(judge_batch())}` "
        f"`TRIPPY_JUDGE_MODEL={judge_model()}`",
        "",
    ]
    totals = merge_snapshots([r.get("stages") or {} for r in rows])
    totals_s = format_stages(totals)
    if totals_s:
        lines.append(f"- stages: {totals_s}")
    usage_totals = _usage_totals(rows)
    usage_s = format_usage_line(usage_totals)
    if usage_s:
        lines.append(f"- {usage_s}")
    if totals_s or usage_s:
        lines.append("")
    lines.extend(
        [
            "| id | diff | result | s | date | fit sites | failures |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        score = row["score"]
        mark = "PASS" if score["ok"] else "FAIL"
        date = score.get("extract_date") or {}
        date_s = f"{date.get('start', '')}→{date.get('end', '')}" if date else ""
        fails = "; ".join(score.get("failures") or []) or ""
        secs = row.get("seconds")
        secs_s = "" if secs is None else f"{secs:.1f}"
        lines.append(
            f"| {row['id']} | {row['difficulty']} | {mark} | {secs_s} | {date_s} | "
            f"{score.get('fit_sites')} | {fails} |"
        )
    if any("recommend" in r for r in rows):
        lines.extend(
            [
                "",
                "## Recommendations",
                "",
                "| id | query | n | rec | why |",
                "|---|---|---|---|---|",
            ]
        )
        for row in rows:
            rec = row.get("recommend") or {}
            recs = [
                item
                for item in rec.get("recommendations") or []
                if isinstance(item, dict)
            ]
            query = _md_cell(row.get("query"))
            if recs:
                pick = _md_cell(" · ".join(_rec_pick_cell(item) for item in recs))
                why = _md_cell(
                    " · ".join(
                        str(item.get("why") or "") for item in recs if item.get("why")
                    )
                )
                n_s = str(len(recs))
            else:
                pick = "(empty)" if rec else ""
                why = _md_cell(rec.get("empty") or rec.get("text") or "")
                n_s = "0" if rec else ""
            lines.append(
                f"| {row['id']} | {query} | {n_s} | {pick} | {why} |"
            )
    if any(r.get("stages") for r in rows):
        timing_cols = ["id", "s", *STAGE_ORDER]
        lines.extend(
            [
                "",
                "## Timing",
                "",
                "| " + " | ".join(timing_cols) + " |",
                "|" + "|".join(["---"] * len(timing_cols)) + "|",
            ]
        )
        for row in rows:
            snap = row.get("stages") or {}
            secs = row.get("seconds")
            secs_s = "" if secs is None else f"{secs:.1f}"
            cells = [_stage_cell(snap.get(name)) for name in STAGE_ORDER]
            lines.append(f"| {row['id']} | {secs_s} | {' | '.join(cells)} |")
        total_cells = [_stage_cell(totals.get(name)) for name in STAGE_ORDER]
        lines.append(f"| **total** | {wall:.1f} | {' | '.join(total_cells)} |")
    if any(r.get("usage") for r in rows):
        show_recommend = any(
            _usage_role_cell(r.get("usage"), "recommend") for r in rows
        )
        if show_recommend:
            lines.extend(
                [
                    "",
                    "## Tokens",
                    "",
                    "| id | in | out | extract | judge | recommend |",
                    "|---|---|---|---|---|---|",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "## Tokens",
                    "",
                    "| id | in | out | extract | judge |",
                    "|---|---|---|---|---|",
                ]
            )
        for row in rows:
            usage = row.get("usage") or {}
            inn = int(usage.get("input_tokens") or 0)
            out = int(usage.get("output_tokens") or 0)
            inn_s = "" if inn <= 0 and out <= 0 else str(inn)
            out_s = "" if inn <= 0 and out <= 0 else str(out)
            cells = (
                f"| {row['id']} | {inn_s} | {out_s} | "
                f"{_usage_role_cell(usage, 'extract')} | "
                f"{_usage_role_cell(usage, 'claim_judge')}"
            )
            if show_recommend:
                cells += f" | {_usage_role_cell(usage, 'recommend')}"
            lines.append(cells + " |")
        t_in = int(usage_totals.get("input_tokens") or 0)
        t_out = int(usage_totals.get("output_tokens") or 0)
        total = (
            f"| **total** | {t_in} | {t_out} | "
            f"{_usage_role_cell(usage_totals, 'extract')} | "
            f"{_usage_role_cell(usage_totals, 'claim_judge')}"
        )
        if show_recommend:
            total += f" | {_usage_role_cell(usage_totals, 'recommend')}"
        lines.append(total + " |")
    if any(r.get("usage") for r in rows):
        lines.extend(
            [
                "",
                "## Cost",
                "",
                "| id | extract | embed | judge | recommend | total |",
                "|---|---|---|---|---|---|",
            ]
        )
        step_totals = {role: 0.0 for role, _label in _COST_STEPS}
        grand = 0.0
        for row in rows:
            usage = row.get("usage") or {}
            cells: list[str] = []
            row_total = 0.0
            for role, _label in _COST_STEPS:
                amount = _role_cost_usd(usage, role)
                step_totals[role] += amount
                row_total += amount
                cells.append(_fmt_usd(amount))
            grand += row_total
            lines.append(
                f"| {row['id']} | {' | '.join(cells)} | {_fmt_usd(row_total)} |"
            )
        total_cells = [_fmt_usd(step_totals[role]) for role, _label in _COST_STEPS]
        lines.append(
            f"| **total** | {' | '.join(total_cells)} | {_fmt_usd(grand)} |"
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
    if rows:
        lines.extend(["", "## Cases", ""])
        for row in rows:
            lines.extend(_case_trace_lines(row))
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
    parser.add_argument(
        "--model",
        default="",
        help="Instruct model for extractor+judge: 30B or 235B",
    )
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=0,
        help="Parallel claim-judge calls (default 5)",
    )
    parser.add_argument(
        "--judge-compact",
        action="store_true",
        help="Judge returns claim indices and a 4-5 word reason (default on)",
    )
    parser.add_argument(
        "--no-judge-compact",
        action="store_true",
        help="Quoted claim text in the judge JSON (opt out of compact)",
    )
    parser.add_argument(
        "--judge-batch",
        action="store_true",
        help="One judge call per planner case (all site/query jobs)",
    )
    parser.add_argument(
        "--judge-model",
        default="",
        help="Judge model: glm, 30B, 235B, or a full Nebius id",
    )
    parser.add_argument(
        "--recommender",
        action="store_true",
        help="After the planner, pick 1-2 fits and dump the Hebrew rec",
    )
    parser.add_argument(
        "--from-json",
        default="",
        help="Rebuild markdown from a dump; do not run cases",
    )
    args = parser.parse_args(argv)
    spec_path = Path(args.eval)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if args.from_json:
        dump_path = Path(args.from_json)
        dump = json.loads(dump_path.read_text(encoding="utf-8"))
        apply_run_env(spec)
        md = dump_path.with_suffix(".md")
        if args.out_dir:
            md = Path(args.out_dir) / md.name
        write_report(
            md, spec, list(dump.get("cases") or []), float(dump.get("seconds") or 0)
        )
        print(f"report {md}", flush=True)
        return 0
    if not args.no_copy:
        refresh_experiments_from_public()
    apply_run_env(spec)
    if args.model:
        os.environ["TRIPPY_INSTRUCT_MODEL"] = args.model
    if args.judge_concurrency:
        os.environ["TRIPPY_JUDGE_CONCURRENCY"] = str(args.judge_concurrency)
    if args.no_judge_compact:
        os.environ["TRIPPY_JUDGE_COMPACT"] = "0"
    elif args.judge_compact:
        os.environ["TRIPPY_JUDGE_COMPACT"] = "1"
    if args.judge_batch:
        os.environ["TRIPPY_JUDGE_BATCH"] = "1"
    if args.judge_model:
        os.environ["TRIPPY_JUDGE_MODEL"] = args.judge_model
    table = os.environ.get("TRIPPY_AVAILABILITY_TABLE") or "availability"
    print(f"TRIPPY_SCHEMA={os.environ.get('TRIPPY_SCHEMA')}", flush=True)
    print(f"TRIPPY_TODAY={os.environ.get('TRIPPY_TODAY')}", flush=True)
    print(
        f"TRIPPY_INSTRUCT_MODEL={os.environ.get('TRIPPY_INSTRUCT_MODEL') or '235B'}",
        flush=True,
    )
    print(
        f"TRIPPY_JUDGE_CONCURRENCY={judge_concurrency()}",
        flush=True,
    )
    print(
        f"TRIPPY_JUDGE_COMPACT={int(judge_compact())}",
        flush=True,
    )
    print(
        f"TRIPPY_JUDGE_BATCH={int(judge_batch())}",
        flush=True,
    )
    print(
        f"TRIPPY_JUDGE_MODEL={judge_model()}",
        flush=True,
    )
    print(f"recommender={int(args.recommender)}", flush=True)
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
        extract, planner, recommend, stages, usage = run_one(
            query, recommender=args.recommender
        )
        elapsed = time.monotonic() - t0
        score = score_case(case.get("expect") or {}, extract, planner)
        mark = "PASS" if score["ok"] else "FAIL"
        stages_s = format_stages(stages)
        print(
            f"{mark} {elapsed:.1f}s date={score.get('extract_date')} "
            f"fits={score.get('fit_sites')} {score.get('failures')}",
            flush=True,
        )
        if stages_s:
            print(stages_s, flush=True)
        usage_s = format_usage_line(usage)
        if usage_s:
            print(usage_s, flush=True)
        if recommend and recommend.get("text"):
            print(recommend["text"], flush=True)
        row = {
            "id": cid,
            "difficulty": case.get("difficulty"),
            "query": query,
            "seconds": round(elapsed, 1),
            "stages": stages,
            "usage": usage,
            "score": score,
            "extract": extract,
            "planner": _summarize_planner(planner),
        }
        if recommend is not None:
            row["recommend"] = recommend
        rows.append(row)
    wall = time.monotonic() - started
    dump = {
        "eval": spec.get("id"),
        "at": datetime.now().isoformat(timespec="seconds"),
        "seconds": round(wall, 1),
        "pass": sum(1 for r in rows if r["score"]["ok"]),
        "n": len(rows),
        "recommender": bool(args.recommender),
        "cases": rows,
    }
    report_json.write_text(
        json.dumps(dump, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    write_report(report_md, spec, rows, wall)
    passed = dump["pass"]
    print(f"\n{passed}/{len(rows)} pass in {wall:.1f}s", flush=True)
    totals_usage_s = format_usage_line(_usage_totals(rows))
    if totals_usage_s:
        print(totals_usage_s, flush=True)
    print(f"report {report_md}", flush=True)
    print(f"dump   {report_json}", flush=True)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
