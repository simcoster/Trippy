"""The per-run Markdown report written after `scrape-rules`.

Built from the same `SiteReport` the terminal summary uses, so these tests hand
it traces and drops directly -- no LLM, no database -- and read the file back.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace

from db.models import QualifierUnit
from source.scraper.amenity_enrichment.llm import LlmUsage
from source.scraper.rules_ingest.db import DroppedRule, ResolvedRule
from source.scraper.rules_ingest.ingest import SiteReport
from source.scraper.rules_ingest.report import (
    SiteRun,
    render_run_report,
    report_path,
    write_run_report,
)
from source.scraper.subjects.resolve import Candidate, ResolutionTrace

LATE = (
    "שעות כניסה ויציאה: לנים המבקשים להישאר באתר לאחר השעה 12:00 ועד לסיום "
    "שעות הפעילות בשעה 17:00 נדרשים לתשלום של 50% מדמי כניסת יום לאתר"
)
STARTED = datetime(2026, 9, 4, 18, 46, 5)


def make_report() -> SiteReport:
    report = SiteReport()
    report.traces.append(
        ResolutionTrace(
            term="late_check_out_allowed",
            normalized="late_check_out_allowed",
            category=2,
            context=LATE,
            outcome="nothing near enough to ask. INSERTED as boolean_rule 'late_check_out_allowed'.",
            kind="inserted",
            subject_id=34,
            subject_name="late_check_out_allowed",
        )
    )
    report.traces.append(
        ResolutionTrace(
            term="late_check_out_end_time",
            normalized="late_check_out_end_time",
            category=3,
            context=LATE,
            candidates=[
                Candidate(
                    id=34,
                    name="late_check_out_allowed",
                    distance=-0.906,
                    category=2,
                    context="שעות כניסה ויציאה: first sentence | with a pipe",
                ),
                Candidate(
                    id=33, name="check_out_time", distance=-0.804, category=3, context="x"
                ),
            ],
            outcome="ADJUDICATOR merged into 'late_check_out_allowed'.",
            kind="merged",
            subject_id=34,
            subject_name="late_check_out_allowed",
        )
    )
    report.traces.append(
        ResolutionTrace(
            term="no_dogs_never",
            normalized="no_dogs_never",
            outcome="DROPPED (negative phrasing that cannot be made positive).",
            kind="dropped",
        )
    )
    # The second pass over the same term is a cache hit and leaves no trace.
    report.drops.append(
        DroppedRule(
            "CONFLICTING",
            1,
            kept=ResolvedRule(
                subject_id=34,
                polarity=True,
                term="late_check_out_allowed",
                section_title="שעות כניסה ויציאה",
                evidence_span=LATE.split(": ", 1)[1],
            ),
            dropped=ResolvedRule(
                subject_id=34,
                qualifier=Decimal("17"),
                qualifier_unit=int(QualifierUnit.HOUR_OF_DAY),
                term="late_check_out_end_time",
                section_title="שעות כניסה ויציאה",
                evidence_span=LATE.split(": ", 1)[1],
            ),
        )
    )
    return report


def make_usage() -> LlmUsage:
    usage = LlmUsage()
    usage.add_chat(
        SimpleNamespace(prompt_tokens=1000, completion_tokens=100),
        role="rules_extract",
        model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    )
    usage.add_chat(
        SimpleNamespace(prompt_tokens=500, completion_tokens=5),
        role="merge_judge",
        model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    )
    usage.add_embed(
        SimpleNamespace(prompt_tokens=12), role="embed", model="Qwen/Qwen3-Embedding-8B"
    )
    return usage


def make_runs() -> list[SiteRun]:
    site = {"id": 1, "name": "חניון לילה גן לאומי חורשת טל", "url": "https://x/1"}
    return [SiteRun(site=site, report=make_report(), written=49, seconds=318.3)]


def test_the_report_shows_each_merge_with_both_original_sentences():
    text = render_run_report(make_runs(), make_usage(), started_at=STARTED, seconds=489)
    merged = next(line for line in text.splitlines() if line.startswith("| `late_check_out_end_time`"))
    assert "`late_check_out_allowed`" in merged
    assert "לנים המבקשים להישאר" in merged  # the term's own sentence
    assert "first sentence \\| with a pipe" in merged  # the winner's, pipe escaped
    assert "check_out_time" not in merged  # the rejected candidate is not the winner


def test_the_report_lists_new_subjects_with_category_and_sentence():
    text = render_run_report(make_runs(), make_usage(), started_at=STARTED, seconds=489)
    assert "| `late_check_out_allowed` | boolean_rule | שעות כניסה ויציאה:" in text


def test_the_report_carries_upsert_collisions_with_both_phrasings():
    text = render_run_report(make_runs(), make_usage(), started_at=STARTED, seconds=489)
    assert "**CONFLICTING** on `late_check_out_allowed` (campsite 1)" in text
    assert "**kept:** `late_check_out_allowed`" in text
    assert "**dropped:** `late_check_out_end_time`" in text
    assert "qualifier=17 hour_of_day" in text
    assert "ADJUDICATOR merged into 'late_check_out_allowed'." in text


def test_the_report_lists_resolver_drops():
    text = render_run_report(make_runs(), make_usage(), started_at=STARTED, seconds=489)
    assert "- `no_dogs_never`: DROPPED (negative phrasing" in text


def test_the_report_breaks_cost_down_by_role_and_model():
    text = render_run_report(make_runs(), make_usage(), started_at=STARTED, seconds=489)
    assert "| rules_extract | Qwen3-235B-A22B-Instruct-2507 | 1 | 1000 | 100 |" in text
    assert "| merge_judge | Qwen3-235B-A22B-Instruct-2507 | 1 | 500 | 5 |" in text
    assert "| embed | Qwen3-Embedding-8B | 1 | 12 | 0 |" in text
    assert "| **total** | | 3 | 1512 | 105 |" in text


def test_the_header_summarises_the_run():
    text = render_run_report(make_runs(), make_usage(), started_at=STARTED, seconds=489)
    head = text.split("## Cost by role")[0]
    assert "# scrape-rules — 2026-09-04 18:46:05" in head
    assert "- pages: 1" in head
    assert "- rules upserted: 49" in head
    assert "- terms resolved: 3 (1 dropped, 1 inserted, 1 merged)" in head
    assert "- duration: 8 min 9 s" in head


def test_a_failed_page_is_reported_with_its_error():
    runs = make_runs()
    runs.append(
        SiteRun(
            site={"id": 2, "name": "אכזיב", "url": "https://x/2"},
            report=SiteReport(),
            error="HTTP error: 503",
            seconds=1.2,
        )
    )
    text = render_run_report(runs, make_usage(), started_at=STARTED, seconds=489)
    assert "- pages: 2 (1 failed)" in text
    assert "**Failed, rolled back:** HTTP error: 503" in text
    assert "### Merged into an existing subject (0)\n\nnone" in text


def test_the_file_is_named_by_start_time_in_the_given_folder(tmp_path):
    path = write_run_report(
        make_runs(), make_usage(), started_at=STARTED, seconds=489, directory=tmp_path
    )
    assert path == tmp_path / "2026-09-04_184605.md"
    assert path.read_text(encoding="utf-8").startswith("# scrape-rules — 2026-09-04 18:46:05")


def test_the_folder_comes_from_the_environment_when_set(monkeypatch, tmp_path):
    monkeypatch.setenv("RULES_REPORT_DIR", str(tmp_path / "elsewhere"))
    assert report_path(STARTED) == tmp_path / "elsewhere" / "2026-09-04_184605.md"
    monkeypatch.delenv("RULES_REPORT_DIR")
    assert report_path(STARTED).parts[-3:] == ("reports", "rules_ingest", "2026-09-04_184605.md")
