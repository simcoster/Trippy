"""tomorrow / tonight date_intent: schema, resolve_dates, extractor wiring."""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from source.agent.dates import resolve_dates
from source.agent.prompts import EXTRACTOR_SYSTEM_PROMPT

MONDAY = date(2026, 9, 21)
PROMPT_LIVE = (
    "we're looking for a place for 2 adults and 2 kids for tomorrow, "
    "with pools for the kids, maybe with a fridge"
)


def test_prompt_contains_tomorrow_token():
    text = EXTRACTOR_SYSTEM_PROMPT
    assert '"on": "YYYY-MM-DD" | "today" | "tonight" | "tomorrow" | null' in text
    assert '"kind": "on", "on": "tomorrow"' in text
    assert "מחר" in text
    assert PROMPT_LIVE in text
    assert "Not on=\"today\"" in text or "not on=\"today\"" in text.lower()


def test_resolve_tomorrow_is_next_night():
    resolved = resolve_dates(
        kind="on", on="tomorrow", nights=1, today=MONDAY
    )
    assert resolved["windows"] == [{"start": "2026-09-22", "end": "2026-09-23"}]
    assert resolved["truncated"] is False


def test_resolve_tonight_is_today():
    resolved = resolve_dates(
        kind="on", on="tonight", nights=1, today=MONDAY
    )
    assert resolved["windows"] == [{"start": "2026-09-21", "end": "2026-09-22"}]


def test_resolve_today_is_today():
    resolved = resolve_dates(kind="on", on="today", nights=1, today=MONDAY)
    assert resolved["windows"] == [{"start": "2026-09-21", "end": "2026-09-22"}]


def test_resolve_tomorrow_two_nights():
    resolved = resolve_dates(
        kind="on", on="tomorrow", nights=2, today=MONDAY
    )
    assert resolved["windows"] == [{"start": "2026-09-22", "end": "2026-09-24"}]


def test_extractor_resolves_on_tomorrow(monkeypatch):
    from source.agent import graph as agent_graph
    from source.agent.dates import resolve_dates as resolve

    monkeypatch.setattr(agent_graph, "today_il", lambda today=None: MONDAY)
    llm_json = {
        "date_intent": {"kind": "on", "on": "tomorrow", "nights": 1},
        "campsite": None,
        "numeric_constraints": [
            {"field": "party_size", "operator": ">=", "value": 4}
        ],
        "semantic_constraints": [
            {"query": "pools for children", "locus": "site"},
            {"query": "fridge", "locus": "site"},
        ],
    }
    fake_model = MagicMock()
    fake_model.invoke.return_value = AIMessage(content=json.dumps(llm_json))
    monkeypatch.setattr(agent_graph, "extractor_model", fake_model)
    monkeypatch.setattr(
        agent_graph,
        "resolve_dates_tool",
        SimpleNamespace(invoke=lambda args: resolve(**args, today=MONDAY)),
    )

    result = agent_graph.extractor_node(
        {"messages": [HumanMessage(content=PROMPT_LIVE)]}
    )
    payload = json.loads(result["messages"][0].content)
    assert payload["date_intent"] == {
        "kind": "on",
        "on": "tomorrow",
        "nights": 1,
    }
    assert payload["date"] == {"start": "2026-09-22", "end": "2026-09-23"}
    assert payload["date_windows"] == [payload["date"]]
