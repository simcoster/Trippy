"""Extractor: weekday glued to desert must keep both.

Live miss: לשבוע הבא בחמישי במדבר kept Thursday and dropped desert.
The few-shot is the glued prefix; this prompt also has party, cleanliness,
and shomer shabbat so those must survive too.
"""

from __future__ import annotations

import json
from datetime import date

import pytest
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from source.agent.prompts import EXTRACTOR_SYSTEM_PROMPT

load_dotenv()

MONDAY = date(2026, 9, 14)
NEXT_THURSDAY = date(2026, 9, 24)

PROMPT = (
    "משהו לשבוע הבא בחמישי במדבר ל3 אנשים, חשוב לנו ניקיון. אחד שומר שבת"
)
TRIALS = range(5)


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
    return " ".join(str(s).lower().replace("_", " ").replace("-", " ").split())


def _semantic_texts(semantic: list) -> list[str]:
    texts: list[str] = []
    for item in semantic or []:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            if item.get("query"):
                texts.append(str(item["query"]))
            texts.extend(str(v) for v in (item.get("values") or []))
    return [_norm(t) for t in texts]


def _has_any(texts: list[str], needles: tuple[str, ...]) -> bool:
    return any(n in t for t in texts for n in needles)


def _party_size_at_least_3(numeric: list) -> bool:
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
        if value == 3 and op in {">=", "=>", "gte"}:
            return True
    return False


def _freeze_monday(monkeypatch: pytest.MonkeyPatch) -> None:
    from source.agent import graph as agent_graph

    monkeypatch.setattr(agent_graph, "today_il", lambda today=None: MONDAY)


def _extract(prompt: str) -> dict:
    from source.agent.graph import extractor_model, extractor_node

    assert extractor_model.temperature == 0
    result = extractor_node({"messages": [HumanMessage(content=prompt)]})
    return _extractor_constraints_json(result["messages"])


def test_prompt_keeps_desert_when_glued_to_weekday():
    text = EXTRACTOR_SYSTEM_PROMPT
    assert "בחמישי במדבר" in text
    assert '"query": "desert"' in text
    assert "Never drop a location pref" in text


@pytest.mark.llm
@pytest.mark.parametrize("trial", TRIALS, ids=lambda i: f"t{i + 1}")
def test_extractor_next_thursday_desert_clean_shabbat(trial, monkeypatch):
    _freeze_monday(monkeypatch)
    constraints = _extract(PROMPT)
    semantic = constraints.get("semantic_constraints") or []
    texts = _semantic_texts(semantic)
    intent = constraints.get("date_intent") or {}

    assert str(intent.get("kind") or "").lower() == "weekday", intent
    assert str(intent.get("weekday") or "").lower() == "thursday", intent
    assert str(intent.get("when") or "").lower() == "next", intent
    assert intent.get("nights") == 1, intent

    date_field = constraints.get("date")
    assert isinstance(date_field, dict), f"expected date object, got {date_field!r}"
    assert str(date_field.get("start") or "")[:10] == NEXT_THURSDAY.isoformat(), (
        f"date={date_field!r} date_intent={intent!r}"
    )

    assert _has_any(texts, ("desert", "negev", "מדבר")), (
        f"expected desert in semantic_constraints={semantic!r}"
    )
    assert _has_any(texts, ("clean",)), (
        f"expected cleanliness in semantic_constraints={semantic!r}"
    )
    assert _has_any(texts, ("shabbat", "shabbos", "sabbath")), (
        f"expected shomer shabbat in semantic_constraints={semantic!r}"
    )
    assert _party_size_at_least_3(constraints.get("numeric_constraints") or []), (
        f"expected party_size>=3, got {constraints.get('numeric_constraints')!r}"
    )
    assert not constraints.get("campsite")
    assert "amenities" not in constraints
