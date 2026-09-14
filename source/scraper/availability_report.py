"""Change report for one `scrape-availability` run.

The GitHub Actions Summary is the UI. This module builds the markdown;
`AVAILABILITY_REPORT_PATH` is only how the SSH session copies it there.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from source.scraper.amenity_enrichment.llm import LlmUsage

REPORT_PATH_ENV = "AVAILABILITY_REPORT_PATH"


class NightCountKey(NamedTuple):
    site_id: int
    type_name: str


class PageFingerprint(NamedTuple):
    html_sha256: str
    offers_sha256: str


class VacancyChange(NamedTuple):
    site_id: int
    site_name: str
    start_date: date
    type_name: str
    old_count: int | None
    new_count: int | None


class HttpError(NamedTuple):
    site_id: int
    site_name: str
    start_date: date
    message: str


class LayoutSuspicion(NamedTuple):
    site_id: int
    site_name: str
    start_date: date
    previous_had_offers: bool
    now_empty: bool


class PastNightsDeleted(NamedTuple):
    availability_rows: int
    hash_rows: int


@dataclass
class AvailabilityRun:
    """One scrape-availability run, for the Actions Summary."""

    started_at: datetime
    seconds: float = 0.0
    sites: int = 0
    nights: int = 0
    pages_fetched: int = 0
    pages_skipped: int = 0
    rows_upserted: int = 0
    past_rows_deleted: int = 0
    past_hashes_deleted: int = 0
    changes: list[VacancyChange] = field(default_factory=list)
    unmatched: list[tuple[int, str]] = field(default_factory=list)
    http_errors: list[HttpError] = field(default_factory=list)
    layout_suspicions: list[LayoutSuspicion] = field(default_factory=list)


def html_sha256(html: str) -> str:
    return hashlib.sha256((html or "").encode("utf-8")).hexdigest()


def offers_sha256(aggregated: list[dict]) -> str:
    """Stable digest of aggregated (room_type, room_count); order-independent."""
    rows = sorted(
        (str(offer.get("room_type") or ""), int(offer.get("room_count") or 0))
        for offer in aggregated
    )
    blob = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def should_skip_write(
    stored_offers_sha: str | None, new_offers_sha: str
) -> bool:
    return stored_offers_sha is not None and stored_offers_sha == new_offers_sha


def layout_suspicion(
    *,
    previous_offers_sha: str | None,
    new_aggregated: list[dict],
) -> bool:
    """Parser empty vs previously non-empty (or the reverse)."""
    if previous_offers_sha is None:
        return False
    previous_empty = previous_offers_sha == offers_sha256([])
    now_empty = not new_aggregated
    return previous_empty != now_empty


def diff_night_counts(
    *,
    site_name: str,
    start: date,
    old: dict[NightCountKey, int],
    new: dict[NightCountKey, int],
) -> list[VacancyChange]:
    changes: list[VacancyChange] = []
    keys = set(old) | set(new)
    for key in sorted(keys):
        before = old.get(key)
        after = new.get(key)
        if before == after:
            continue
        changes.append(
            VacancyChange(
                site_id=key.site_id,
                site_name=site_name,
                start_date=start,
                type_name=key.type_name,
                old_count=before,
                new_count=after,
            )
        )
    return changes


def _change_line(change: VacancyChange) -> str:
    day = change.start_date.isoformat()
    label = f"**{change.site_name}** {day} · {change.type_name}"
    if change.old_count is None:
        return f"- {label} · appeared ×{change.new_count}"
    if change.new_count is None:
        return f"- {label} · gone (was {change.old_count})"
    return f"- {label} · {change.old_count} → {change.new_count}"


def render_run_report(run: AvailabilityRun, usage: LlmUsage) -> str:
    cost = usage.report("scrape-availability")
    started = run.started_at.strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# scrape-availability",
        "",
        f"Started: {started}",
        f"Duration: {run.seconds:.1f}s",
        (
            f"Sites: {run.sites} · nights: {run.nights} · "
            f"fetched: {run.pages_fetched} · skipped: {run.pages_skipped} · "
            f"upserted: {run.rows_upserted} · "
            f"dropped past: {run.past_rows_deleted}"
        ),
        "",
        "## Cost",
        "",
        f"${cost['cost_usd']:.6f} · {cost['calls']} call(s)",
        "",
        "## Vacancy changes",
        "",
    ]
    if run.changes:
        lines.extend(_change_line(c) for c in run.changes)
    else:
        lines.append("No vacancy changes.")
    lines.extend(["", "## Skipped", ""])
    lines.append(
        f"{run.pages_skipped} page(s) unchanged (offers hash)."
        if run.pages_skipped
        else "None."
    )
    lines.extend(["", "## Unmatched booking names", ""])
    if run.unmatched:
        for site_id, name in sorted(set(run.unmatched)):
            lines.append(f"- campsite {site_id}: {name}")
    else:
        lines.append("None.")
    lines.extend(["", "## HTTP errors", ""])
    if run.http_errors:
        for err in run.http_errors:
            lines.append(
                f"- **{err.site_name}** {err.start_date.isoformat()}: {err.message}"
            )
    else:
        lines.append("None.")
    lines.extend(["", "## Layout suspicions", ""])
    if run.layout_suspicions:
        for item in run.layout_suspicions:
            if item.now_empty:
                detail = "page had offers yesterday, parser returned none"
            else:
                detail = "page was empty yesterday, parser returned offers"
            lines.append(
                f"- **{item.site_name}** {item.start_date.isoformat()}: {detail}"
            )
    else:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)


def write_run_report(text: str, path: Path | None = None) -> Path | None:
    """Write markdown when `AVAILABILITY_REPORT_PATH` (or `path`) is set."""
    dest = path
    if dest is None:
        raw = os.environ.get(REPORT_PATH_ENV)
        if not raw:
            return None
        dest = Path(raw)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return dest
