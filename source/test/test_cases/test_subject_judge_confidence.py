"""The merge judge sees what each side states and its match is gated on confidence.

Both came out of experiments.md 2026-09-04 §10, §12 and §14: with the two
sentences already in view the judge merged a 30-person minimum into an
80-person one; shown the numbers it did not, and every wrong merge it still
made came back below 0.9 while every right answer came back at 0.95.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from source.scraper.subjects.llm import (
    ADJUDICATE_SYSTEM_PROMPT,
    MATCH_MIN_CONFIDENCE,
    Judgement,
    SubjectAdjudicatorLLMClient,
)
from source.scraper.subjects.resolve import format_states


def make_client(content: str) -> SubjectAdjudicatorLLMClient:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=None
    )
    openai = MagicMock()
    openai.chat.completions.create.return_value = response
    return SubjectAdjudicatorLLMClient(openai)


def sent(client: SubjectAdjudicatorLLMClient) -> str:
    return client.client.chat.completions.create.call_args.kwargs["messages"][1]["content"]


# --- the prompt ---------------------------------------------------------------
def test_the_prompt_explains_states_lines_and_asks_for_confidence():
    assert "`states:` line" in ADJUDICATE_SYSTEM_PROMPT
    assert "read from ONE page that state different numbers" in ADJUDICATE_SYSTEM_PROMPT
    assert '"confidence": <number 0..1' in ADJUDICATE_SYSTEM_PROMPT
    assert MATCH_MIN_CONFIDENCE == 0.9


# --- what the judge is shown ----------------------------------------------------
def test_states_lines_are_sent_for_both_sides_when_given():
    client = make_client('{"match": null, "confidence": 0.95}')
    client.pick_match(
        "family_and_friends_group_min_occupancy",
        ["group_min_occupancy"],
        term_context="מערכת הזמנות: 30-80 לנים",
        candidate_contexts={"group_min_occupancy": "מערכת הזמנות: מעל 80 לנים"},
        term_states="qualifier=30 count",
        candidate_states={"group_min_occupancy": "qualifier=80 count (same page)"},
    )
    text = sent(client)
    assert "Term: family_and_friends_group_min_occupancy" in text
    assert "    context: מערכת הזמנות: 30-80 לנים\n    states: qualifier=30 count" in text
    assert "- group_min_occupancy\n    context: מערכת הזמנות: מעל 80 לנים\n    states: qualifier=80 count (same page)" in text


def test_no_states_line_appears_when_none_is_known():
    client = make_client('{"match": "shower", "confidence": 0.95}')
    client.pick_match("showers", ["shower"], term_context="x")
    assert "states:" not in sent(client)


# --- the gate ---------------------------------------------------------------------
def test_a_confident_match_is_returned():
    client = make_client('{"match": "freezers", "confidence": 0.95}')
    sink: list[Judgement] = []
    assert client.pick_match("freezer", ["freezers"], judgement_sink=sink) == "freezers"
    assert sink == [Judgement("freezers", 0.95, True)]


def test_a_match_below_the_gate_is_not_a_match_but_is_reported():
    """The caravan-pitch hookup: answered "match" at 0.3 -- only the gate stopped it."""
    client = make_client('{"match": "electric_hookup_in_caravan_pitch", "confidence": 0.3}')
    sink: list[Judgement] = []
    assert client.pick_match("electric_hookup", ["electric_hookup_in_caravan_pitch"], judgement_sink=sink) is None
    assert sink == [Judgement("electric_hookup_in_caravan_pitch", 0.3, False)]


@pytest.mark.parametrize("confidence", [0.9, 0.95, 1.0, 1.7])
def test_the_gate_is_inclusive_and_confidence_is_clamped(confidence):
    client = make_client(f'{{"match": "shower", "confidence": {confidence}}}')
    assert client.pick_match("showers", ["shower"]) == "shower"


def test_a_reply_without_confidence_is_accepted_as_before():
    client = make_client('{"match": "shower"}')
    sink: list[Judgement] = []
    assert client.pick_match("showers", ["shower"], judgement_sink=sink) == "shower"
    assert sink == [Judgement("shower", None, True)]


def test_a_null_answer_records_its_confidence_too():
    client = make_client('{"match": null, "confidence": 0.95}')
    sink: list[Judgement] = []
    assert client.pick_match("x", ["y"], judgement_sink=sink) is None
    assert sink == [Judgement(None, 0.95, False)]


def test_an_invented_name_is_not_a_match_whatever_the_confidence():
    client = make_client('{"match": "made_up", "confidence": 1.0}')
    sink: list[Judgement] = []
    assert client.pick_match("x", ["y"], judgement_sink=sink) is None
    assert sink[0].match is None


# --- how a statement's value is phrased for the judge ----------------------------------
@pytest.mark.parametrize(
    ("polarity", "qualifier", "unit", "expected"),
    [
        (None, 30, 1, "qualifier=30 count"),
        (None, 16, 2, "qualifier=16 hour_of_day"),
        (None, 20.5, 2, "qualifier=20.5 hour_of_day"),
        (True, None, 0, "polarity=true"),
        (False, None, 0, "polarity=false"),
        (True, 2, 1, "polarity=true qualifier=2 count"),
        (None, 50, 8, "qualifier=50 percent"),
        (None, 3, 0, "qualifier=3"),
        (None, None, 0, None),
    ],
)
def test_format_states(polarity, qualifier, unit, expected):
    assert format_states(polarity, qualifier, unit) == expected


# --- what the judge is told about a candidate's existing rows ------------------------
def test_candidate_states_put_the_current_page_first_and_summarise_the_rest():
    from source.scraper.subjects.resolve import (
        DEFAULT_STORE,
        STATES_SHOWN,
        Candidate,
        _candidate_states,
    )

    cursor = MagicMock()
    cursor.__enter__ = lambda self: self
    cursor.__exit__ = lambda self, *exc: False
    # subject 42 has rows on six campsites; the page being read (19) is not first by id.
    cursor.fetchall.return_value = [
        (42, 1, None, 80, 1), (42, 3, None, 80, 1), (42, 4, None, 80, 1),
        (42, 5, None, 80, 1), (42, 19, None, 80, 1), (42, 20, None, 80, 1),
        (7, 3, True, None, 0),
    ]
    conn = MagicMock()
    conn.cursor.return_value = cursor
    offered = [
        Candidate(id=42, name="group_min_occupancy", distance=-0.9, category=3),
        Candidate(id=7, name="dogs_allowed", distance=-0.8, category=2),
        Candidate(id=9, name="no_rows_yet", distance=-0.8, category=2),
    ]

    states = _candidate_states(conn, offered, 19, DEFAULT_STORE)

    assert states["group_min_occupancy"].startswith("qualifier=80 count (same page); ")
    assert states["group_min_occupancy"].endswith(f"; +{6 - STATES_SHOWN} more")
    assert states["dogs_allowed"] == "polarity=true (campsite 3)"
    assert "no_rows_yet" not in states
    sql, params = cursor.execute.call_args.args
    assert "FROM campsite_rules" in sql and params == ([42, 7, 9],)
