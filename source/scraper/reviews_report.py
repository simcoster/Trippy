"""Change report for one `scrape-reviews` run.

The GitHub Actions Summary is the UI. This module builds the markdown;
`REVIEWS_REPORT_PATH` is only how the SSH session copies it there.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

REPORT_PATH_ENV = "REVIEWS_REPORT_PATH"
_PREVIEW_MAX = 80


class NewReview(NamedTuple):
    site_id: int
    site_name: str
    author: str | None
    rating: int | None
    published_at: datetime | None
    preview: str


class SkippedSite(NamedTuple):
    site_id: int
    site_name: str
    reason: str


class FetchError(NamedTuple):
    site_id: int
    site_name: str
    message: str


@dataclass
class ReviewsRun:
    """One scrape-reviews run, for the Actions Summary."""

    started_at: datetime
    seconds: float = 0.0
    sites: int = 0
    reviews_fetched: int = 0
    reviews_inserted: int = 0
    reviews_seen: int = 0
    new_reviews: list[NewReview] = field(default_factory=list)
    skipped: list[SkippedSite] = field(default_factory=list)
    http_errors: list[FetchError] = field(default_factory=list)


def review_preview(text: str | None) -> str:
    blob = " ".join((text or "").split())
    if not blob:
        return "(empty)"
    if len(blob) <= _PREVIEW_MAX:
        return blob
    return blob[: _PREVIEW_MAX - 3] + "..."


def _new_review_line(item: NewReview) -> str:
    author = item.author or "anonymous"
    stars = f"{item.rating}★" if item.rating is not None else "no rating"
    when = (
        item.published_at.date().isoformat()
        if item.published_at is not None
        else "no date"
    )
    return (
        f"- **{item.site_name}** · {author} · {stars} · {when} — {item.preview}"
    )


def render_run_report(run: ReviewsRun) -> str:
    started = run.started_at.strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# scrape-reviews",
        "",
        f"Started: {started}",
        f"Duration: {run.seconds:.1f}s",
        (
            f"Sites: {run.sites} · fetched: {run.reviews_fetched} · "
            f"new: {run.reviews_inserted} · already stored: {run.reviews_seen}"
        ),
        "",
        "## Cost",
        "",
        "$0.000000 · 0 call(s)",
        "",
        "Google Place Details only; no LLM.",
        "",
        "## New reviews",
        "",
    ]
    if run.new_reviews:
        lines.extend(_new_review_line(item) for item in run.new_reviews)
    else:
        lines.append("No new reviews.")
    lines.extend(["", "## Skipped", ""])
    if run.skipped:
        for item in run.skipped:
            lines.append(f"- **{item.site_name}** (campsite {item.site_id}): {item.reason}")
    else:
        lines.append("None.")
    lines.extend(["", "## HTTP / API errors", ""])
    if run.http_errors:
        for err in run.http_errors:
            lines.append(f"- **{err.site_name}**: {err.message}")
    else:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)


def write_run_report(text: str, path: Path | None = None) -> Path | None:
    """Write markdown when `REVIEWS_REPORT_PATH` (or `path`) is set."""
    dest = path
    if dest is None:
        raw = os.environ.get(REPORT_PATH_ENV)
        if not raw:
            return None
        dest = Path(raw)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return dest
