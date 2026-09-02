"""campsite_rules writes: idempotent upserts and the amenity-id mirror."""

import json
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from db.models import QualifierUnit, SubjectCategory
from source.scraper.rules_ingest.db import (
    ResolvedRule,
    sync_campsite_amenity_ids,
    upsert_campsite_rules,
)
from source.scraper.rules_ingest.llm import _coerce_payload
from source.scraper.rules_ingest.schemas import RuleExtract, RuleStatement


def make_cursor(rows=()):
    cursor = MagicMock()
    cursor.fetchall.return_value = list(rows)
    return cursor


def params_of(cursor, index=0):
    return cursor.execute.call_args_list[index].args[1]


def test_upsert_writes_one_row_per_rule_with_a_null_unit_scope():
    cursor = make_cursor()
    rules = [
        ResolvedRule(
            subject_id=1,
            polarity=False,
            evidence_span="הכניסה לכלבים אסורה",
            source_url="https://x",
            confidence=0.95,
        ),
        ResolvedRule(
            subject_id=2,
            qualifier=Decimal("20.5"),
            qualifier_unit=int(QualifierUnit.HOUR_OF_DAY),
        ),
    ]

    assert upsert_campsite_rules(cursor, campsite_id=5, rules=rules) == 2

    first = params_of(cursor, 0)
    assert first["campsite_id"] == 5
    assert first["accommodation_type_id"] is None  # site-wide
    assert first["subject_id"] == 1
    assert first["polarity"] is False
    assert first["qualifier"] is None
    assert first["evidence_span"] == "הכניסה לכלבים אסורה"

    second = params_of(cursor, 1)
    assert second["qualifier"] == Decimal("20.5")
    assert second["qualifier_unit"] == int(QualifierUnit.HOUR_OF_DAY)


def test_upsert_targets_the_nulls_not_distinct_unique_key():
    cursor = make_cursor()
    upsert_campsite_rules(cursor, campsite_id=5, rules=[ResolvedRule(subject_id=1)])

    sql = cursor.execute.call_args.args[0]
    assert (
        "ON CONFLICT (campsite_id, accommodation_type_id, subject_id) DO UPDATE" in sql
    )
    assert "updated_at = now()" in sql


def test_a_subject_stated_twice_in_one_scope_is_written_once():
    """The unique key allows one row per subject; a repeat would abort the batch."""
    cursor = make_cursor()
    rules = [
        ResolvedRule(subject_id=1, polarity=True),
        ResolvedRule(subject_id=1, polarity=False),
    ]
    assert upsert_campsite_rules(cursor, campsite_id=5, rules=rules) == 1
    assert cursor.execute.call_count == 1


def test_a_unit_scope_is_passed_through_when_given():
    cursor = make_cursor()
    upsert_campsite_rules(
        cursor, campsite_id=5, rules=[ResolvedRule(subject_id=1)], accommodation_type_id=9
    )
    assert params_of(cursor)["accommodation_type_id"] == 9


def test_no_rules_writes_nothing():
    cursor = make_cursor()
    assert upsert_campsite_rules(cursor, campsite_id=5, rules=[]) == 0
    cursor.execute.assert_not_called()


def test_sync_splits_site_amenities_by_polarity():
    cursor = make_cursor(rows=[(11, True), (12, False), (13, None)])

    included, not_included = sync_campsite_amenity_ids(cursor, campsite_id=5)

    assert (included, not_included) == (2, 1)
    written = cursor.execute.call_args.args[1]
    # A bare quantity ("6 fountains") still means the site has the thing.
    assert json.loads(written["amenities"]) == [11, 13]
    assert json.loads(written["not_included"]) == [12]
    assert written["campsite_id"] == 5


def test_sync_reads_only_site_level_amenity_rows():
    cursor = make_cursor(rows=[])
    sync_campsite_amenity_ids(cursor, campsite_id=5)

    select_sql, select_params = cursor.execute.call_args_list[0].args
    assert "cr.accommodation_type_id IS NULL" in select_sql
    assert "sv.category = %(amenity)s" in select_sql
    assert select_params["amenity"] == int(SubjectCategory.AMENITY)


def test_sync_clears_the_arrays_when_a_site_has_no_amenities():
    cursor = make_cursor(rows=[])
    assert sync_campsite_amenity_ids(cursor, campsite_id=5) == (0, 0)
    written = cursor.execute.call_args.args[1]
    assert json.loads(written["amenities"]) == []
    assert json.loads(written["not_included"]) == []


# --- extractor payload tolerance ----------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        '{"statements": []}',
        "[]",  # what the model actually sends for an empty section
        '```json\n[]\n```',
        "Here you go: []",
    ],
)
def test_an_empty_section_parses_however_the_model_words_it(raw):
    assert _coerce_payload(raw) == {"statements": []}


def test_a_bare_list_of_statements_is_wrapped():
    payload = _coerce_payload('[{"subject": "shower", "category": "amenity"}]')
    assert RuleExtract.model_validate(payload).statements[0].subject == "shower"


def test_unparseable_output_still_raises():
    with pytest.raises(ValueError):
        _coerce_payload("no json here at all")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("rule", 2),
        ("rules", 2),
        ("amenity", 1),
        ("amenities", 1),
        (1, 1),
        (2, 2),
    ],
)
def test_statement_category_is_coerced(raw, expected):
    statement = RuleStatement.model_validate({"subject": "x", "category": raw})
    assert statement.category == expected


@pytest.mark.parametrize("raw", [None, "", "nonsense", 7, -1])
def test_an_unusable_category_becomes_none_not_a_guess(raw):
    """None searches every category; a wrong guess would hide the real subject."""
    statement = RuleStatement.model_validate({"subject": "x", "category": raw})
    assert statement.category is None


def test_category_is_optional():
    assert RuleStatement(subject="x").category is None


# --- injected tables ----------------------------------------------------------


def test_rules_are_written_to_the_injected_table():
    cursor = make_cursor()
    upsert_campsite_rules(
        cursor,
        campsite_id=5,
        rules=[ResolvedRule(subject_id=1)],
        table="test_campsite_rules",
    )
    sql = cursor.execute.call_args.args[0]
    assert "INSERT INTO test_campsite_rules" in sql
    assert "INTO campsite_rules" not in sql


def test_the_default_rules_table_is_production():
    cursor = make_cursor()
    upsert_campsite_rules(cursor, campsite_id=5, rules=[ResolvedRule(subject_id=1)])
    assert "INSERT INTO campsite_rules" in cursor.execute.call_args.args[0]


def test_the_mirror_reads_and_writes_the_injected_tables():
    cursor = make_cursor(rows=[(11, True)])
    sync_campsite_amenity_ids(
        cursor,
        campsite_id=5,
        rules_table="test_campsite_rules",
        subjects_table="test_subject_vectors",
    )
    select_sql = cursor.execute.call_args_list[0].args[0]
    assert "FROM test_campsite_rules cr" in select_sql
    assert "JOIN test_subject_vectors sv" in select_sql


@pytest.mark.parametrize("table", ["campsite_rules; DROP TABLE x", "a.b", "Rules"])
def test_a_bad_table_name_is_refused(table):
    with pytest.raises(ValueError):
        upsert_campsite_rules(
            make_cursor(), campsite_id=5, rules=[ResolvedRule(subject_id=1)], table=table
        )
