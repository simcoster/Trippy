"""Extractor emits child_num, child_ages, and planned_exit_time."""

from __future__ import annotations

import json

import pytest
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

pytestmark = pytest.mark.llm


def _extractor_constraints_json(messages: list) -> dict:
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        raw = msg.content
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise AssertionError(
        "extractor_node did not return a JSON constraints AIMessage; "
        f"got {[type(m).__name__ for m in messages]}"
    )


def _extract(prompt: str) -> dict:
    from source.agent.graph import extractor_node

    result = extractor_node({"messages": [HumanMessage(content=prompt)]})
    return _extractor_constraints_json(result["messages"])


def _norm(value: str) -> str:
    return " ".join(str(value).lower().replace("_", " ").replace("-", " ").split())


def _semantic_texts(semantic: list) -> list[str]:
    texts: list[str] = []
    for item in semantic or []:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            if item.get("query"):
                texts.append(str(item["query"]))
            texts.extend(str(v) for v in (item.get("values") or []))
    return texts


def _party_at_least(numeric: list, size: int) -> bool:
    for item in numeric or []:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field") or "").lower()
        if field not in {"party_size", "adults", "guests"}:
            continue
        try:
            value = int(item.get("value"))
        except (TypeError, ValueError):
            continue
        op = str(item.get("operator") or item.get("op") or "").strip()
        if value == size and op in {">=", "=>", "gte"}:
            return True
    return False


def test_extractor_emits_child_ages_in_stated_order():
    """Two adults and children aged 3 then 6 → child_num 2, ages [3, 6]."""
    constraints = _extract("מחפשים מקום למחר, שני מבוגרים ושני ילדים בני 3 ו-6")
    assert constraints.get("child_num") == 2, constraints
    assert constraints.get("child_ages") == [3, 6], constraints
    assert _party_at_least(constraints.get("numeric_constraints") or [], 4), constraints
    leaked = [
        text
        for text in _semantic_texts(constraints.get("semantic_constraints") or [])
        if any(needle in _norm(text) for needle in ("age", "aged", "3", "6", "ילד"))
    ]
    assert not leaked, f"ages belong on child_ages, not semantic: {leaked!r} {constraints!r}"


def test_extractor_emits_child_count_without_inventing_ages():
    """One adult and three kids, no ages → child_num 3 and no child_ages."""
    constraints = _extract("camping tomorrow for one adult and three kids")
    assert constraints.get("child_num") == 3, constraints
    assert not constraints.get("child_ages"), constraints
    assert _party_at_least(constraints.get("numeric_constraints") or [], 4), constraints


def test_extractor_emits_planned_exit_separate_from_entry():
    """Arrive after 18 and leave at 11 → both clocks, exit not folded into entry."""
    constraints = _extract("נכנסים אחרי 18 ויוצאים ב-11")
    assert constraints.get("planned_entry_time") == "18:00", constraints
    assert constraints.get("planned_exit_time") == "11:00", constraints
    leaked = [
        text
        for text in _semantic_texts(constraints.get("semantic_constraints") or [])
        if any(
            needle in _norm(text)
            for needle in ("leave", "exit", "depart", "11", "18", "arriv")
        )
    ]
    assert not leaked, (
        f"clocks belong on planned_entry_time / planned_exit_time, not semantic: "
        f"{leaked!r} {constraints!r}"
    )
