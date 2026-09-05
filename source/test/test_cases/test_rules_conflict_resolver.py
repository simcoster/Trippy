"""The conflict resolver: two actions, one automatic. No database, no model.

`rename_new` undoes a wrong merge by giving the new statement its own subject;
everything else is `none` and the case is filed. `validate` enforces the one
constraint the model is not trusted with (a usable, distinct name), the apply
step is checked SQL-by-SQL against a fake cursor, and the page-level flow is
checked with a mocked resolver.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from source.scraper.amenity_enrichment.llm import LlmUsage
from source.scraper.rules_ingest.db import DroppedRule, ResolvedRule
from source.scraper.rules_ingest.explain import PIPELINE_MECHANICS
from source.scraper.rules_ingest.ingest import SiteReport
from source.scraper.rules_ingest.report import SiteRun, render_run_report
from source.scraper.rules_ingest.resolve_conflicts import (
    ACTIONS,
    RESOLVE_SYSTEM_PROMPT,
    ConflictResolution,
    ConflictResolverLLMClient,
    ResolutionPayload,
    SubjectFacts,
    apply_rename_new,
    file_conflict_case,
    resolve_page_conflicts,
    subject_facts,
    validate,
)
from source.scraper.subjects.resolve import DEFAULT_STORE, ResolutionTrace

RUN_AT = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)
GROUPS = "תיאום לינה לקבוצות (מעל 80 לנים):"
FAMILIES = "תיאום לינה לקבוצות משפחות וחברים (30-80 לנים המשלמים יחד):"
OLD = SubjectFacts("group_min_occupancy", 3, ("group_min_occupancy", "family_and_friends_group_min_occupancy"), 5)


def payload(**kw) -> ResolutionPayload:
    return ResolutionPayload.model_validate(kw)


def make_drop(label="CONFLICTING") -> DroppedRule:
    return DroppedRule(
        label, 1,
        kept=ResolvedRule(42, None, Decimal("80"), 1, GROUPS, "https://x", 0.9, "group_min_occupancy", "מערכת הזמנות"),
        dropped=ResolvedRule(42, None, Decimal("30"), 1, FAMILIES, "https://x", 0.9, "family_and_friends_group_min_occupancy", "מערכת הזמנות"),
    )


class FakeCursor:
    """Answers the resolver's SQL by shape; records everything it was asked."""

    def __init__(self, *, existing_by_name=None, category=3, next_subject_id=77, case_id=9, old_name="group_min_occupancy"):
        self.existing_by_name = existing_by_name
        self.category = category
        self.old_name = old_name
        self.next_subject_id = next_subject_id
        self.case_id = case_id
        self.statements: list[tuple[str, object]] = []
        self._result = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.statements.append((" ".join(sql.split()), params))
        if "WHERE name = %s" in sql:
            self._result = (self.existing_by_name,) if self.existing_by_name else None
        elif "SELECT name, category FROM" in sql:
            self._result = (self.old_name, self.category)
        elif sql.strip().startswith("INSERT INTO subject_vectors"):
            self._result = (self.next_subject_id,)
        elif "INSERT INTO conflict_cases" in sql:
            self._result = (self.case_id,)
        elif "SELECT s.name, s.category, s.aliases" in sql:
            self._result = ("group_min_occupancy", 3, ["group_min_occupancy", "family_and_friends_group_min_occupancy"], 5)
        else:
            self._result = None

    def fetchone(self):
        return self._result

    def matching(self, needle):
        return [s for s in self.statements if needle in s[0]]


def make_conn(cursor: FakeCursor) -> MagicMock:
    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn


def make_embedder():
    embedder = MagicMock()
    embedder.embed.side_effect = lambda texts, **kw: [[0.5] * 4 for _ in texts]
    return embedder


# --- the contract ---------------------------------------------------------------------
def test_only_two_actions_exist_and_the_prompt_says_so():
    assert ACTIONS == ("none", "rename_new")
    assert PIPELINE_MECHANICS in RESOLVE_SYSTEM_PROMPT
    assert "toilets_accessible" in RESOLVE_SYSTEM_PROMPT  # named as the wrong shape
    assert "cannot touch the" in RESOLVE_SYSTEM_PROMPT and "kept row; say so" in RESOLVE_SYSTEM_PROMPT


@pytest.mark.parametrize("raw", ["drop_new", "rename_old", "reassign_kept", "enrich_kept", "", None, "explode"])
def test_every_other_proposed_action_becomes_none(raw):
    assert payload(action=raw).action == "none"


def test_rename_new_is_kept_and_its_name_normalised():
    assert payload(action="RENAME_NEW", new_name="Accessible Toilets").new_name == "accessible_toilets"


# --- validate ---------------------------------------------------------------------------
def test_rename_new_may_reclaim_the_alias_the_merge_created():
    out = validate(payload(action="rename_new"), OLD, "family_and_friends_group_min_occupancy")
    assert out.action == "rename_new"
    assert out.new_name == "family_and_friends_group_min_occupancy"
    assert out.notes  # says the fallback was used


def test_rename_new_to_the_old_name_with_no_other_term_is_filed_as_none():
    out = validate(payload(action="rename_new", new_name="group_min_occupancy"), OLD, "group_min_occupancy")
    assert out.action == "none" and out.new_name is None


def test_none_never_carries_a_name():
    out = validate(payload(action="none", new_name="x", confidence=0.4), OLD, "t")
    assert out.action == "none" and out.new_name is None and out.confidence == 0.4


# --- apply_rename_new: the SQL, in order ---------------------------------------------------
def test_apply_releases_the_alias_creates_the_subject_and_writes_the_row():
    cursor = FakeCursor()
    resolution = ConflictResolution("judge_over_merge", "both", "e", "rename_new", "family_and_friends_group_min_occupancy")
    subject_id = apply_rename_new(make_conn(cursor), make_drop(), resolution, embedder=make_embedder(), store=DEFAULT_STORE)

    assert subject_id == 77
    release = cursor.matching("array_remove")
    assert len(release) == 1 and release[0][1] == {"id": 42, "alias": "family_and_friends_group_min_occupancy"}
    assert "aliases[1] <> %(alias)s" in release[0][0]  # the canonical alias is never removed
    insert = cursor.matching("INSERT INTO subject_vectors")[0]
    assert insert[1]["name"] == "family_and_friends_group_min_occupancy"
    assert insert[1]["category"] == 3  # the old subject's shelf
    assert insert[1]["aliases"] == ["family_and_friends_group_min_occupancy"]
    assert insert[1]["context"] == f"מערכת הזמנות: {FAMILIES}"
    row = cursor.matching("INSERT INTO campsite_rules")[0]
    assert row[1]["subject_id"] == 77 and row[1]["campsite_id"] == 1 and row[1]["qualifier"] == Decimal("30")


def test_apply_joins_an_existing_subject_of_that_name_instead_of_duplicating():
    cursor = FakeCursor(existing_by_name=55, old_name="toilets", category=1)
    resolution = ConflictResolution("extractor_wrong_name", "both", "e", "rename_new", "accessible_toilets")
    drop = DroppedRule("CONFLICTING", 19,
        kept=ResolvedRule(4, True, Decimal("18"), 1, "שירותים (18)", None, None, "toilets", "מה בחניון?"),
        dropped=ResolvedRule(4, True, None, 0, "הונגשו: שירותים", None, None, "toilets", "נגישות"))
    embedder = make_embedder()

    assert apply_rename_new(make_conn(cursor), drop, resolution, embedder=embedder) == 55
    assert not cursor.matching("INSERT INTO subject_vectors")
    embedder.embed.assert_not_called()
    # the bare term IS the old subject's name: it is neither released nor aliased elsewhere
    assert not cursor.matching("array_remove")
    assert not cursor.matching("array_append")
    assert cursor.matching("INSERT INTO campsite_rules")[0][1]["subject_id"] == 55


def test_apply_adds_the_term_as_an_alias_when_it_differs_from_the_new_name():
    cursor = FakeCursor(existing_by_name=55)
    resolution = ConflictResolution("x", "both", "e", "rename_new", "accessible_toilets")
    drop = make_drop()
    apply_rename_new(make_conn(cursor), drop, resolution, embedder=make_embedder())
    assert cursor.matching("array_append")[0][1] == {"id": 55, "alias": "family_and_friends_group_min_occupancy"}


# --- file_conflict_case -------------------------------------------------------------------
def test_filing_copies_both_sides_and_the_verdict():
    cursor = FakeCursor(case_id=9)
    resolution = ConflictResolution("judge_over_merge", "both", "two facts", "rename_new", "family_and_friends_group_min_occupancy", "r", 0.95)
    resolution.applied, resolution.applied_subject_id = True, 77
    case_id = file_conflict_case(make_conn(cursor), make_drop(), resolution, run_at=RUN_AT, kept_how="INSERTED", dropped_how="merged")
    assert case_id == 9
    sql, params = cursor.matching("INSERT INTO conflict_cases")[0]
    assert params["kt"] == "group_min_occupancy" and params["kq"] == Decimal("80") and params["kh"] == "INSERTED"
    assert params["nt"] == "family_and_friends_group_min_occupancy" and params["nq"] == Decimal("30") and params["nh"] == "merged"
    assert params["action"] == "rename_new" and params["applied"] is True and params["applied_subject_id"] == 77
    assert params["status"] == "applied" and params["run_at"] == RUN_AT and params["confidence"] == 0.95


def test_an_unapplied_case_is_filed_open():
    cursor = FakeCursor()
    resolution = ConflictResolution("true_duplicate", "kept", "same fact", "none")
    file_conflict_case(make_conn(cursor), make_drop("duplicate"), resolution, run_at=RUN_AT, kept_how=None, dropped_how=None)
    params = cursor.matching("INSERT INTO conflict_cases")[0][1]
    assert params["status"] == "open" and params["applied"] is False and params["new_name"] is None
    assert params["label"] == "duplicate"


def test_subject_facts_reads_name_category_aliases_and_row_count():
    facts = subject_facts(make_conn(FakeCursor()), 42)
    assert facts == SubjectFacts("group_min_occupancy", 3, ("group_min_occupancy", "family_and_friends_group_min_occupancy"), 5)


# --- the page-level flow --------------------------------------------------------------------
def make_resolver(content: str) -> ConflictResolverLLMClient:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=700, completion_tokens=90),
    )
    openai = MagicMock()
    openai.chat.completions.create.return_value = response
    return ConflictResolverLLMClient(openai)


def make_report() -> SiteReport:
    report = SiteReport()
    report.traces.append(ResolutionTrace(term="group_min_occupancy", normalized="group_min_occupancy", category=3,
        outcome="nothing near enough to ask. INSERTED as numeric_rule 'group_min_occupancy'.", kind="inserted", subject_id=42, subject_name="group_min_occupancy"))
    report.traces.append(ResolutionTrace(term="family_and_friends_group_min_occupancy", normalized="family_and_friends_group_min_occupancy", category=3,
        outcome="ADJUDICATOR merged into 'group_min_occupancy' (confidence 0.95).", kind="merged", subject_id=42, subject_name="group_min_occupancy"))
    report.drops.append(make_drop())
    return report


RENAME = ('{"cause": "judge_over_merge", "which_is_right": "both", "explanation": "Two group types.", '
          '"action": "rename_new", "new_name": "family_and_friends_group_min_occupancy", "rationale": "r", "confidence": 0.95}')


def test_a_wrong_merge_is_undone_filed_as_applied_and_shown_in_both_reports():
    report = make_report()
    cursor = FakeCursor()
    usage = LlmUsage()
    applied = resolve_page_conflicts(make_conn(cursor), report, make_resolver(RENAME), embedder=make_embedder(), run_at=RUN_AT, usage=usage, verbose=False)

    assert applied == 1
    resolution = report.resolutions[0]
    assert resolution.applied and resolution.applied_subject_id == 77 and resolution.case_id == 9
    assert report.explanations[0].cause == "judge_over_merge"
    assert [b.role for b in usage.by_role()] == ["conflict_resolver"]
    assert cursor.matching("INSERT INTO conflict_cases")[0][1]["kh"].startswith("nothing near enough")
    assert "resolution: rename_new -> 'family_and_friends_group_min_occupancy', applied as subject #77 (case #9) [confidence 0.95]. r" in report.render()
    run = SiteRun(site={"id": 1, "name": "x", "url": "https://x"}, report=report, written=1)
    text = render_run_report([run], usage, started_at=datetime(2026, 9, 5, 12, 0), seconds=1)
    assert "- conflicts: 1 filed for review, 1 resolved by undoing a merge" in text
    assert "- **resolution:** rename_new -> 'family_and_friends_group_min_occupancy', applied as subject #77 (case #9)" in text


def test_none_files_the_case_open_and_changes_nothing():
    report = make_report()
    cursor = FakeCursor()
    content = '{"cause": "true_duplicate", "which_is_right": "kept", "explanation": "same fact", "action": "none", "confidence": 0.95}'
    applied = resolve_page_conflicts(make_conn(cursor), report, make_resolver(content), embedder=make_embedder(), run_at=RUN_AT, verbose=False)
    assert applied == 0
    assert not cursor.matching("array_remove") and not cursor.matching("INSERT INTO subject_vectors") and not cursor.matching("INSERT INTO campsite_rules")
    params = cursor.matching("INSERT INTO conflict_cases")[0][1]
    assert params["action"] == "none" and params["status"] == "open"
    assert "resolution: none: filed for review (case #9)" in report.render()


def test_a_failed_model_call_still_files_the_case():
    report = make_report()
    resolver = make_resolver(RENAME)
    resolver.client.chat.completions.create.side_effect = RuntimeError("boom")
    cursor = FakeCursor()
    assert resolve_page_conflicts(make_conn(cursor), report, resolver, embedder=make_embedder(), run_at=RUN_AT, verbose=False) == 0
    params = cursor.matching("INSERT INTO conflict_cases")[0][1]
    assert params["action"] == "none" and "resolver failed: boom" in params["explanation"]


def test_a_failed_apply_files_the_case_unapplied_with_the_error():
    report = make_report()
    cursor = FakeCursor()
    embedder = MagicMock()
    embedder.embed.side_effect = RuntimeError("no embeddings today")
    resolve_page_conflicts(make_conn(cursor), report, make_resolver(RENAME), embedder=embedder, run_at=RUN_AT, verbose=False)
    resolution = report.resolutions[0]
    assert resolution.action == "rename_new" and not resolution.applied
    assert any("apply failed: no embeddings today" in n for n in resolution.notes)
    assert cursor.matching("INSERT INTO conflict_cases")[0][1]["status"] == "open"


def test_no_collisions_means_no_calls():
    resolver = make_resolver(RENAME)
    assert resolve_page_conflicts(MagicMock(), SiteReport(), resolver, embedder=make_embedder(), run_at=RUN_AT, verbose=False) == 0
    resolver.client.chat.completions.create.assert_not_called()
