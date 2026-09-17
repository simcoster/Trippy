"""One Markdown report per `scrape-prices` run.

The terminal scrolls past AST banners and gold lines. This file keeps the
outcome per campsite: stored or failed, which retry ran, the failing gold
/ static lines, and the versioned dump stems (`13_v1.py`) for attempts
that did not pass.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from source.scraper.amenity_enrichment.llm import LlmUsage

REPORT_DIR_ENV = "PRICES_REPORT_DIR"
REPORT_COPY_ENV = "PRICES_REPORT_PATH"
DEFAULT_REPORT_DIR = Path("reports") / "scrape_prices"


@dataclass
class PriceFunctionRun:
    site_id: int
    site_name: str
    url: str = ""
    outcome: str = "skipped"
    n_gold: int = 0
    digest: str = ""
    retry: str | None = None
    skip_reason: str = ""
    failures: list[str] = field(default_factory=list)
    fail_stems: list[str] = field(default_factory=list)
    store_status: str = ""
    scraped_at: datetime | None = None
    updated_at: datetime | None = None


def run_folder(started_at: datetime, directory: Path | None = None) -> Path:
    """One directory per scrape-prices run: dumps and the Markdown report."""
    root = directory or Path(os.environ.get(REPORT_DIR_ENV) or DEFAULT_REPORT_DIR)
    return root / f"{started_at:%Y-%m-%d_%H%M%S}"


def report_path(started_at: datetime, directory: Path | None = None) -> Path:
    folder = directory or Path(os.environ.get(REPORT_DIR_ENV) or DEFAULT_REPORT_DIR)
    return folder / f"{started_at:%Y-%m-%d_%H%M%S}.md"


def write_run_report(
    runs: list[PriceFunctionRun],
    usage: LlmUsage,
    *,
    started_at: datetime,
    seconds: float,
    directory: Path | None = None,
    name: str | None = None,
) -> Path:
    path = report_path(started_at, directory)
    if name is not None:
        path = path.with_name(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_run_report(runs, usage, started_at=started_at, seconds=seconds),
        encoding="utf-8",
    )
    copy = (os.environ.get(REPORT_COPY_ENV) or "").strip()
    if copy:
        dest = Path(copy)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def render_run_report(
    runs: list[PriceFunctionRun],
    usage: LlmUsage,
    *,
    started_at: datetime,
    seconds: float,
) -> str:
    stored = [run for run in runs if run.outcome == "stored"]
    failed = [
        run
        for run in runs
        if run.outcome in {"gold_failed", "ast_failed", "compile_error"}
    ]
    skipped = [run for run in runs if run.outcome in {"skipped", "http_error"}]
    lines = [
        f"# scrape-prices — {started_at:%Y-%m-%d %H:%M:%S}",
        "",
        f"- campsites: {len(runs)}",
        f"- stored: {len(stored)}"
        + (
            f" ({_store_kinds(stored)})"
            if stored
            else ""
        ),
        f"- failed: {len(failed)}",
        f"- skipped: {len(skipped)}",
        f"- duration: {_duration(seconds)}",
        f"- cost: ${usage.cost_usd:.4f} "
        f"({usage.chat_calls} chat / {usage.embed_calls} embed calls)",
        "",
        "## Cost by role",
        "",
    ]
    lines += _cost_table(usage)
    if failed:
        lines += ["", "## Failures", ""]
        for run in failed:
            lines += _site_section(run)
    if stored:
        lines += ["", "## Stored", ""]
        for run in stored:
            lines += _site_section(run)
    if skipped:
        lines += ["", "## Skipped", ""]
        for run in skipped:
            lines += _site_section(run)
    return "\n".join(lines) + "\n"


def stamp(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.isoformat(timespec="seconds")


def _site_section(run: PriceFunctionRun) -> list[str]:
    lines = [f"### {run.site_id}. {run.site_name}", ""]
    lines.append(f"- outcome: {_outcome_label(run)}")
    if run.retry:
        lines.append(f"- retry: {run.retry}")
    if run.n_gold:
        lines.append(f"- gold cases: {run.n_gold}")
    if run.digest:
        lines.append(f"- sha256: `{run.digest[:12]}`")
    if run.scraped_at is not None:
        lines.append(f"- scraped_at: {stamp(run.scraped_at)}")
    if run.updated_at is not None:
        lines.append(f"- updated_at: {stamp(run.updated_at)}")
    if run.skip_reason:
        lines.append(f"- reason: {run.skip_reason}")
    dumps: list[str] = []
    if run.outcome not in {"skipped", "http_error"} or run.fail_stems:
        dumps.append(f"`{run.site_id}.py`")
        dumps.extend(f"`{stem}.py`" for stem in run.fail_stems)
    if dumps:
        lines.append(f"- dumps: {', '.join(dumps)}")
        lines.append(
            "- prompts: "
            + ", ".join(name.replace(".py", ".prompt.txt") for name in dumps)
        )
    lines.append("")
    if run.failures:
        lines += ["Failing tests:", ""]
        for line in run.failures:
            lines.append(f"- {line}")
        lines.append("")
    return lines


def _outcome_label(run: PriceFunctionRun) -> str:
    if run.outcome == "stored":
        return run.store_status or "stored"
    if run.outcome == "gold_failed" and run.store_status:
        return f"gold failed (stored {run.store_status})"
    labels = {
        "gold_failed": "gold failed",
        "ast_failed": "AST / static failed",
        "compile_error": "compile call failed",
        "skipped": "skipped",
        "http_error": "HTTP error",
    }
    return labels.get(run.outcome, run.outcome)


def _store_kinds(runs: list[PriceFunctionRun]) -> str:
    counts: dict[str, int] = {}
    for run in runs:
        key = run.store_status or "stored"
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{n} {name}" for name, n in counts.items())


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
    for block in rows:
        lines.append(
            f"| {block.role} | {block.model.rsplit('/', 1)[-1]} | {block.calls} "
            f"| {block.prompt_tokens} | {block.completion_tokens} | {block.cost_usd:.4f} |"
        )
    lines.append(
        f"| **total** | | {usage.chat_calls + usage.embed_calls} "
        f"| {usage.input_tokens} | {usage.output_tokens} | **{usage.cost_usd:.4f}** |"
    )
    return lines
