"""One readable Markdown report per `scrape-rules` run.

The terminal says everything once, as it scrolls past. This file keeps the part
worth re-reading afterwards, for every page the run touched:

- which extractor terms were merged into an existing subject, with the sentence
  each side was read from -- an over-merge is a wrong answer to "are these the
  same fact?", and the two sentences are what you need to check it;
- which subjects are new to the vocabulary, and from which sentence;
- which statements the upsert refused because their subject already had a row
  in that scope, with both phrasings;
- which terms the resolver dropped outright;
- under each collision, the resolver's diagnosis (extractor or judge, which
  side is right) and what was done: a wrong merge undone by giving the new
  statement its own subject, or the case filed as open in `conflict_cases`;
- what the run cost, by role and model.

One file per run under `reports/rules_ingest/` (git-ignored, like the cost
log), named by the run's start time. `RULES_REPORT_DIR` overrides the folder.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from db.models import QualifierUnit
from source.scraper.amenity_enrichment.llm import LlmUsage
from source.scraper.subjects.resolve import ResolutionTrace, category_label

if TYPE_CHECKING:
    from source.scraper.rules_ingest.db import DroppedRule, ResolvedRule
    from source.scraper.rules_ingest.explain import ConflictExplanation
    from source.scraper.rules_ingest.ingest import SiteReport
    from source.scraper.rules_ingest.resolve_conflicts import ConflictResolution

REPORT_DIR_ENV = "RULES_REPORT_DIR"
DEFAULT_REPORT_DIR = Path("reports") / "rules_ingest"


@dataclass
class SiteRun:
    """One page's outcome: what was written, how long it took, what the resolver did."""

    site: dict
    report: SiteReport
    written: int = 0
    seconds: float = 0.0
    # Set when the page was rolled back (or never fetched); the report still
    # carries whatever the resolver decided before the failure.
    error: str | None = None


def report_path(started_at: datetime, directory: Path | None = None) -> Path:
    folder = directory or Path(os.environ.get(REPORT_DIR_ENV) or DEFAULT_REPORT_DIR)
    return folder / f"{started_at:%Y-%m-%d_%H%M%S}.md"


def write_run_report(
    runs: list[SiteRun],
    usage: LlmUsage,
    *,
    started_at: datetime,
    seconds: float,
    directory: Path | None = None,
) -> Path:
    """Render and write the report; return the file written."""
    path = report_path(started_at, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = render_run_report(runs, usage, started_at=started_at, seconds=seconds)
    path.write_text(text, encoding="utf-8")
    return path


def render_run_report(
    runs: list[SiteRun], usage: LlmUsage, *, started_at: datetime, seconds: float
) -> str:
    kinds: Counter[str] = Counter()
    for run in runs:
        kinds.update(t.kind or "?" for t in _first_traces(run.report).values())
    failed = [r for r in runs if r.error]
    filed = sum(len(r.report.resolutions) for r in runs)
    applied = sum(1 for r in runs for x in r.report.resolutions.values() if x.applied)

    lines = [f"# scrape-rules — {started_at:%Y-%m-%d %H:%M:%S}", ""]
    pages = f"- pages: {len(runs)}"
    if failed:
        pages += f" ({len(failed)} failed)"
    lines += [
        pages,
        f"- rules upserted: {sum(r.written for r in runs)}",
        f"- terms resolved: {sum(kinds.values())} ({_kinds(kinds)})",
        f"- conflicts: {filed} filed for review, {applied} resolved by undoing a merge",
        f"- duration: {_duration(seconds)}",
        f"- cost: ${usage.cost_usd:.4f} "
        f"({usage.chat_calls} chat / {usage.embed_calls} embed calls)",
        "",
        "## Cost by role",
        "",
    ]
    lines += _cost_table(usage)
    for run in runs:
        lines += [""] + _site_section(run)
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ sections
def _site_section(run: SiteRun) -> list[str]:
    site = run.site
    first = _first_traces(run.report)
    kinds = Counter(t.kind or "?" for t in first.values())
    lines = [f"## {site.get('name', '?')} (campsite {site.get('id', '?')})", ""]
    if site.get("url"):
        lines += [f"<{site['url']}>", ""]
    if run.error:
        lines += [f"**Failed, rolled back:** {run.error}", ""]
    lines += [
        f"- rules upserted: {run.written}",
        f"- took: {_duration(run.seconds)}",
        f"- terms resolved: {sum(kinds.values())} ({_kinds(kinds)})",
        "",
    ]

    merged = [t for t in first.values() if t.kind == "merged"]
    lines += [f"### Merged into an existing subject ({len(merged)})", ""]
    if merged:
        lines += [
            "| term | read from | merged into | that subject was first read from |",
            "|---|---|---|---|",
        ]
        for t in merged:
            lines.append(
                f"| `{t.term}` | {_cell(t.context)} | `{t.subject_name}` "
                f"| {_cell(_winner_context(t))} |"
            )
    else:
        lines.append("none")
    lines.append("")

    inserted = [t for t in first.values() if t.kind == "inserted"]
    lines += [f"### New subjects ({len(inserted)})", ""]
    if inserted:
        lines += ["| subject | category | read from |", "|---|---|---|"]
        for t in inserted:
            lines.append(
                f"| `{t.subject_name or t.normalized}` | {category_label(t.category)} "
                f"| {_cell(t.context)} |"
            )
    else:
        lines.append("none")
    lines.append("")

    drops = run.report.drops
    lines += [f"### Refused at the upsert ({len(drops)})", ""]
    if drops:
        lines.append(
            "A subject already had a row in this scope. CONFLICTING means the "
            "two statements disagree, so the surviving row may not say what "
            "the page says."
        )
        lines.append("")
        for index, drop in enumerate(drops):
            lines += _drop_lines(
                drop, first, run.report.explanations.get(index),
                run.report.resolutions.get(index),
            )
    else:
        lines.append("none")
    lines.append("")

    dropped = [t for t in first.values() if t.kind == "dropped"]
    lines += [f"### Dropped by the resolver ({len(dropped)})", ""]
    if dropped:
        lines += [f"- `{t.term}`: {t.outcome}" for t in dropped]
    else:
        lines.append("none")
    return lines


def _drop_lines(
    drop: DroppedRule,
    first: dict[str, ResolutionTrace],
    explanation: ConflictExplanation | None = None,
    resolution: ConflictResolution | None = None,
) -> list[str]:
    name = first.get(drop.kept.term or "")
    subject = name.subject_name if name is not None else f"#{drop.kept.subject_id}"
    lines = [f"**{drop.label}** on `{subject}` (campsite {drop.campsite_id})", ""]
    for role, rule in (("kept", drop.kept), ("dropped", drop.dropped)):
        trace = first.get(rule.term) if rule.term else None
        how = trace.outcome if trace is not None else "(resolved from the page cache)"
        where = f" in _{rule.section_title}_" if rule.section_title else ""
        lines.append(
            f"- **{role}:** `{rule.term}`{where} → {_value(rule)}  "
            f"\n  “{_cell(rule.evidence_span)}”  \n  {how}"
        )
    if explanation is not None:
        lines.append(
            f"- **explainer:** `{explanation.cause}`, right: {explanation.which_is_right}. "
            f"{explanation.explanation}  \n  Fix: {explanation.fix}"
        )
    if resolution is not None:
        lines.append(f"- **resolution:** {resolution.one_line()}")
    lines.append("")
    return lines


# ------------------------------------------------------------------ helpers
def _first_traces(report: SiteReport) -> dict[str, ResolutionTrace]:
    """Each term's first resolution on this page; repeats come from the cache."""
    first: dict[str, ResolutionTrace] = {}
    for trace in report.traces:
        first.setdefault(trace.term, trace)
    return first


def _winner_context(trace: ResolutionTrace) -> str | None:
    for c in trace.candidates:
        if c.name == trace.subject_name:
            return c.context
    return None


def _value(rule: ResolvedRule) -> str:
    if rule.qualifier is None:
        return f"polarity={rule.polarity}"
    try:
        unit = QualifierUnit(int(rule.qualifier_unit)).name.lower()
    except ValueError:
        unit = str(rule.qualifier_unit)
    return f"polarity={rule.polarity} qualifier={rule.qualifier} {unit}"


def _cell(text: str | None) -> str:
    """One Markdown table cell: no pipes, no newlines."""
    if not text:
        return ""
    return " ".join(text.split()).replace("|", "\\|")


def _kinds(kinds: Counter[str]) -> str:
    return ", ".join(f"{n} {kind}" for kind, n in sorted(kinds.items())) or "none"


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} s"
    minutes, rest = divmod(int(round(seconds)), 60)
    return f"{minutes} min {rest} s"


def _cost_table(usage: LlmUsage) -> list[str]:
    rows = usage.by_role()
    if not rows:
        return ["no LLM calls"]
    lines = [
        "| role | model | calls | tokens in | tokens out | USD |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for b in rows:
        lines.append(
            f"| {b.role} | {b.model.rsplit('/', 1)[-1]} | {b.calls} "
            f"| {b.prompt_tokens} | {b.completion_tokens} | {b.cost_usd:.4f} |"
        )
    lines.append(
        f"| **total** | | {usage.chat_calls + usage.embed_calls} "
        f"| {usage.input_tokens} | {usage.output_tokens} | **{usage.cost_usd:.4f}** |"
    )
    return lines
