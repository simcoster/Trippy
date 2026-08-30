"""Extractor: Hebrew party + tonight + two nights + AC."""

from __future__ import annotations

import json
from datetime import timedelta

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from source.agent.constraints import today_il

load_dotenv()

PROMPT = "משהו ל3 אנשים החל מהיום ל2 לילות עם מזגן"


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


def _norm(s: str) -> str:
    return " ".join(str(s).lower().replace("_", " ").split())


def _has_air_conditioning(semantic: list) -> bool:
    needles = ("air conditioning", "air conditioner", "ac")
    for item in semantic or []:
        texts: list[str] = []
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            if item.get("query"):
                texts.append(str(item["query"]))
            texts.extend(str(v) for v in (item.get("values") or []))
        if any(n in _norm(t) for t in texts for n in needles):
            return True
    return False


def _party_size_eq_3(numeric: list) -> bool:
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
        op = str(item.get("operator") or item.get("op") or "=").strip()
        if value == 3 and op in {"=", "==", "eq"}:
            return True
    return False


def test_extractor_tonight_two_nights_three_adults_ac():
    """משהו ל3 אנשים החל מהיום ל2 לילות עם מזגן → date + party_size + AC."""
    from source.agent.graph import extractor_node, planner_model

    assert planner_model.temperature == 0

    result = extractor_node({"messages": [HumanMessage(content=PROMPT)]})
    constraints = _extractor_constraints_json(result["messages"])

    today = today_il()
    assert constraints.get("date") == {
        "start": today.isoformat(),
        "end": (today + timedelta(days=2)).isoformat(),
    }, constraints.get("date")
    assert _party_size_eq_3(constraints.get("numeric_constraints") or []), (
        f"expected party_size=3, got {constraints.get('numeric_constraints')!r}"
    )
    assert _has_air_conditioning(constraints.get("semantic_constraints") or []), (
        f"expected air conditioning, got {constraints.get('semantic_constraints')!r}"
    )
    assert not constraints.get("campsite")
    assert "amenities" not in constraints
