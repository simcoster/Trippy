"""The conflict explainer: one advisory model call per upsert collision, shown in
the terminal summary and the run report (experiments.md 2026-09-04 §15)."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

from db.models import QualifierUnit
from source.scraper.amenity_enrichment.llm import LlmUsage
from source.scraper.rules_ingest.db import DroppedRule, ResolvedRule
from source.scraper.rules_ingest.explain import (
    EXPLAIN_SYSTEM_PROMPT,
    ConflictExplainerLLMClient,
    ConflictExplanation,
    explain_conflicts,
)
from source.scraper.rules_ingest.ingest import SiteReport
from source.scraper.rules_ingest.report import SiteRun, render_run_report
from source.scraper.subjects.resolve import ResolutionTrace

GROUPS = "תיאום לינה לקבוצות (מעל 80 לנים):"
FAMILIES = "תיאום לינה לקבוצות משפחות וחברים (30-80 לנים המשלמים יחד):"
REPLY = (
    '{"cause": "judge_over_merge", "which_is_right": "both", '
    '"explanation": "Two group types: over 80 (מעל 80) and 30-80 families.", '
    '"fix": "keep family_and_friends_group_min_occupancy separate"}'
)


def make_explainer(content: str) -> ConflictExplainerLLMClient:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=500, completion_tokens=60),
    )
    openai = MagicMock()
    openai.chat.completions.create.return_value = response
    return ConflictExplainerLLMClient(openai)


def make_drop() -> DroppedRule:
    return DroppedRule(
        "CONFLICTING",
        1,
        kept=ResolvedRule(
            subject_id=42, qualifier=Decimal("80"), qualifier_unit=int(QualifierUnit.COUNT),
            term="group_min_occupancy", section_title="מערכת הזמנות", evidence_span=GROUPS,
        ),
        dropped=ResolvedRule(
            subject_id=42, qualifier=Decimal("30"), qualifier_unit=int(QualifierUnit.COUNT),
            term="family_and_friends_group_min_occupancy", section_title="מערכת הזמנות",
            evidence_span=FAMILIES,
        ),
    )


def make_report() -> SiteReport:
    report = SiteReport()
    report.traces.append(ResolutionTrace(
        term="group_min_occupancy", normalized="group_min_occupancy", category=3,
        outcome="nothing near enough to ask. INSERTED as numeric_rule 'group_min_occupancy'.",
        kind="inserted", subject_id=42, subject_name="group_min_occupancy",
    ))
    report.traces.append(ResolutionTrace(
        term="family_and_friends_group_min_occupancy", normalized="family_and_friends_group_min_occupancy",
        category=3, outcome="ADJUDICATOR merged into 'group_min_occupancy'.",
        kind="merged", subject_id=42, subject_name="group_min_occupancy",
    ))
    report.drops.append(make_drop())
    return report


def test_the_prompt_teaches_the_mechanics_the_probe_showed_were_missing():
    assert "alias hit" in EXPLAIN_SYSTEM_PROMPT
    assert "never propose `toilets_count`" in EXPLAIN_SYSTEM_PROMPT
    assert "הונגשו" in EXPLAIN_SYSTEM_PROMPT
    assert "חושה" in EXPLAIN_SYSTEM_PROMPT


def test_both_sides_reach_the_model_with_values_sentences_and_resolver_outcome():
    explainer = make_explainer(REPLY)
    usage = LlmUsage()
    explanation = explainer.explain(
        make_drop(),
        kept_how="nothing near enough to ask. INSERTED",
        dropped_how="ADJUDICATOR merged into 'group_min_occupancy'.",
        usage=usage,
    )
    body = explainer.client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "KEPT: subject name 'group_min_occupancy' (section 'מערכת הזמנות') -> polarity=None qualifier=80 count" in body
    assert "DROPPED: subject name 'family_and_friends_group_min_occupancy'" in body
    assert f"sentence: {FAMILIES!r}" in body
    assert "resolver: ADJUDICATOR merged into 'group_min_occupancy'." in body
    assert explanation == ConflictExplanation(
        "judge_over_merge", "both",
        "Two group types: over 80 (מעל 80) and 30-80 families.",
        "keep family_and_friends_group_min_occupancy separate",
    )
    assert [b.role for b in usage.by_role()] == ["conflict_explainer"]


def test_unknown_cause_and_side_fall_back_and_a_non_json_reply_becomes_the_explanation():
    assert make_explainer('{"cause": "gremlins", "which_is_right": "maybe"}').explain(make_drop()) == ConflictExplanation("other", "neither", "", "")
    assert make_explainer("no json here").explain(make_drop()).explanation == "no json here"


def test_explain_conflicts_fills_the_report_and_survives_a_failed_call():
    report = make_report()
    report.drops.append(make_drop())
    explainer = make_explainer(REPLY)
    explainer.client.chat.completions.create.side_effect = [
        explainer.client.chat.completions.create.return_value,
        RuntimeError("boom"),
    ]
    assert explain_conflicts(report, explainer, verbose=False) == 1
    assert list(report.explanations) == [0]
    assert report.explanations[0].cause == "judge_over_merge"


def test_explain_conflicts_makes_no_call_without_collisions():
    explainer = make_explainer(REPLY)
    assert explain_conflicts(SiteReport(), explainer, verbose=False) == 0
    explainer.client.chat.completions.create.assert_not_called()


def test_the_terminal_summary_and_the_run_report_carry_the_explanation():
    report = make_report()
    explain_conflicts(report, make_explainer(REPLY), verbose=False)
    summary = report.render()
    assert "explainer: judge_over_merge, right: both — Two group types" in summary

    run = SiteRun(site={"id": 1, "name": "x", "url": "https://x"}, report=report, written=1)
    text = render_run_report([run], LlmUsage(), started_at=datetime(2026, 9, 4, 20, 0, 0), seconds=1)
    assert "- **explainer:** `judge_over_merge`, right: both. Two group types: over 80 (מעל 80) and 30-80 families.  \n  Fix: keep family_and_friends_group_min_occupancy separate" in text
