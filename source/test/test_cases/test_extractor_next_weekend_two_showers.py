"""Extractor: next weekend + two showers in the room.

Currently fails: amenity *count* and in-unit *locus* are not in the schema.
Stage-2 RAG is boolean (has a shower-ish amenity), so "2 showers in the room"
must not become party_size=2 or a semantic blob like "hot showers".
"""

from __future__ import annotations

import json
from datetime import timedelta

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from source.agent.dates import today_il, weekday_next_iso_week

load_dotenv()

PROMPT = "next weekend, 2 showers in the room"


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
    return texts


def _as_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _locus_is_in_room(item: dict) -> bool:
    blobs = [
        item.get("locus"),
        item.get("in"),
        item.get("location"),
        item.get("field"),
        item.get("query"),
    ]
    text = _norm(" ".join(str(b) for b in blobs if b))
    return any(
        needle in text
        for needle in ("in the room", "in room", "in-unit", "in unit", "private")
    ) or _norm(str(item.get("locus") or "")) in {"room", "unit", "in_room"}


def _structured_in_room_shower_count(constraints: dict) -> int | None:
    """Count of in-unit showers — numeric field or semantic min_count, not a RAG blob."""
    for item in constraints.get("numeric_constraints") or []:
        if not isinstance(item, dict):
            continue
        field = _norm(str(item.get("field") or ""))
        if "shower" not in field:
            continue
        if not (_locus_is_in_room(item) or "room" in field or "unit" in field):
            continue
        n = _as_int(item.get("value"))
        if n is not None:
            return n
    for item in constraints.get("semantic_constraints") or []:
        if not isinstance(item, dict):
            continue
        query = _norm(str(item.get("query") or ""))
        if "shower" not in query:
            continue
        n = _as_int(item.get("min_count") or item.get("count") or item.get("value"))
        if n is None:
            continue
        if _locus_is_in_room(item) or "room" in query or "unit" in query:
            return n
    return None


def _party_size_value(numeric: list) -> int | None:
    for item in numeric or []:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field") or "").lower()
        if field not in {"party_size", "adults", "guests"}:
            continue
        return _as_int(item.get("value"))
    return None


def test_extractor_next_weekend_two_showers_in_the_room():
    """Next weekend is Friday night; two in-unit showers are a counted amenity."""
    from source.agent.graph import extractor_node

    result = extractor_node({"messages": [HumanMessage(content=PROMPT)]})
    constraints = _extractor_constraints_json(result["messages"])
    semantic = constraints.get("semantic_constraints") or []
    texts = [_norm(t) for t in _semantic_texts(semantic)]

    intent = constraints.get("date_intent") or {}
    assert str(intent.get("kind") or "").lower() == "weekend", intent
    assert str(intent.get("when") or "").lower() == "next", intent
    assert intent.get("nights") == 1, intent

    friday = weekday_next_iso_week(today_il(), 4)
    date_field = constraints.get("date")
    assert isinstance(date_field, dict), f"expected date object, got {date_field!r}"
    assert str(date_field.get("start"))[:10] == friday.isoformat(), (
        f"next weekend = Friday night of next ISO week, got date={date_field!r} "
        f"date_intent={intent!r}"
    )
    assert str(date_field.get("end") or "")[:10] == (
        friday + timedelta(days=1)
    ).isoformat(), (
        f"weekend is one Friday night, not Fri–Sat two nights: {date_field!r}"
    )

    leaked_count = [
        t
        for t in texts
        if "shower" in t and any(n in t for n in ("2", "two"))
    ]
    assert not leaked_count, (
        f"shower count belongs on numeric min_count / showers>=2 with in-room locus, "
        f"not semantic RAG: {leaked_count!r} "
        f"(full semantic_constraints={semantic!r})"
    )

    count = _structured_in_room_shower_count(constraints)
    assert count == 2, (
        f"expected structured in-room shower count 2, got {count!r} "
        f"numeric={constraints.get('numeric_constraints')!r} "
        f"semantic={semantic!r}"
    )
    assert _party_size_value(constraints.get("numeric_constraints") or []) is None, (
        f"'2 showers' is not party_size: {constraints.get('numeric_constraints')!r}"
    )
    assert "amenities" not in constraints
    assert not constraints.get("campsite")
    for item in semantic:
        q = item.get("query") if isinstance(item, dict) else item
        assert not (q and "weekend" in str(q).lower()), f"date leaked into semantic: {q!r}"
