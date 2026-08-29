"""Extractor constraint schema tests (date ranges, OR amenities, aliases)."""

from __future__ import annotations

import json
from datetime import date

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from source.agent.constraints import (
    amenity_search_queries,
    next_friday,
    normalize_constraints,
    resolve_relative_date_phrase,
)

load_dotenv()

PROMPT_HE_RUNNING_WATER = "אני רוצה משהו לשישי הבא עם מים זורמים"
PROMPT_SEA_OR_WATER = "next friday, near the sea or some body of water to swim in"


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


def _date_covers_friday(constraints: dict, friday: date) -> bool:
    friday_iso = friday.isoformat()
    raw = constraints.get("date")
    if isinstance(raw, dict):
        start = str(raw.get("start") or "")[:10]
        end = str(raw.get("end") or start)[:10]
        return start <= friday_iso <= end if start and end else friday_iso in (start, end)
    if isinstance(raw, list):
        return friday_iso in {str(d).strip()[:10] for d in raw}
    if isinstance(raw, str):
        return raw.strip()[:10] == friday_iso
    return False


def _amenities_include_running_water(amenities: list) -> bool:
    for item in amenities or []:
        if isinstance(item, str):
            if "running water" in item.lower().replace("_", " "):
                return True
        elif isinstance(item, dict):
            for v in item.get("values") or []:
                if "running water" in str(v).lower().replace("_", " "):
                    return True
    return False


def _find_or_group(amenities: list) -> dict | None:
    for item in amenities or []:
        if isinstance(item, dict) and str(item.get("op", "")).lower() == "or":
            return item
    return None


def _norm_label(s: str) -> str:
    return " ".join(str(s).lower().replace("_", " ").split())


# ---- Unit tests (no LLM) ----


def test_resolve_next_friday_single_night():
    today = date(2026, 8, 27)  # Thursday
    resolved = resolve_relative_date_phrase("next Friday", today=today)
    assert resolved == {"start": "2026-08-28", "end": "2026-08-28"}


def test_normalize_moves_date_out_of_semantic():
    today = date(2026, 8, 27)
    out = normalize_constraints(
        {
            "semantic_constraints": [
                {"query": "available next Friday"},
                {"query": "quiet"},
            ],
            "numeric_constraints": [],
            "amenities": [],
        },
        today=today,
    )
    assert out["date"] == {"start": "2026-08-28", "end": "2026-08-28"}
    queries = [
        (s.get("query") if isinstance(s, dict) else s) for s in out["semantic_constraints"]
    ]
    assert not any(q and "friday" in str(q).lower() for q in queries)
    assert any(q and "quiet" in str(q).lower() for q in queries)


def test_normalize_or_amenities_and_user_text_date():
    today = date(2026, 8, 27)
    out = normalize_constraints(
        {
            "date": None,
            "amenities": [
                {
                    "op": "or",
                    "values": ["near the sea", "near a body of water"],
                }
            ],
            "numeric_constraints": [],
            "semantic_constraints": [],
        },
        today=today,
        user_text=PROMPT_SEA_OR_WATER,
    )
    assert out["date"] == {"start": "2026-08-28", "end": "2026-08-28"}
    group = _find_or_group(out["amenities"])
    assert group is not None
    norms = {_norm_label(v) for v in group["values"]}
    assert "near the sea" in norms
    assert "near a body of water" in norms


def test_amenity_search_queries_or_group_no_query_expand():
    """Planner searches exact OR values; place→type expansion is LLM ingest only."""
    groups = amenity_search_queries(
        [{"op": "or", "values": ["near the sea", "near a body of water"]}]
    )
    assert len(groups) == 1
    norms = {_norm_label(q) for q in groups[0]}
    assert "near the sea" in norms
    assert "near a body of water" in norms
    assert not any("kineret" in n for n in norms)


# ---- Integration tests (LLM extractor) ----


def test_extractor_next_friday_running_water_constraint_schema():
    """Hebrew trip ask → structured date + amenities."""
    from source.agent.graph import extractor_node

    result = extractor_node({"messages": [HumanMessage(content=PROMPT_HE_RUNNING_WATER)]})
    constraints = _extractor_constraints_json(result["messages"])
    friday = next_friday()

    assert _date_covers_friday(constraints, friday), (
        f"expected next Friday {friday.isoformat()} in date={constraints.get('date')!r}"
    )
    assert "amenities" in constraints, (
        f"expected amenities; got keys {sorted(constraints)}: {constraints}"
    )
    assert _amenities_include_running_water(constraints.get("amenities") or []), (
        f"expected running water in amenities={constraints.get('amenities')!r}"
    )
    # Date must not live only in semantic/numeric
    for item in constraints.get("semantic_constraints") or []:
        q = item.get("query") if isinstance(item, dict) else item
        assert not (q and "friday" in str(q).lower()), f"date leaked into semantic: {q!r}"
    for item in constraints.get("numeric_constraints") or []:
        if isinstance(item, dict):
            assert "date" not in str(item.get("field", "")).lower()


def test_extractor_next_friday_sea_or_body_of_water():
    """English sea OR body-of-water + next Friday → date range + OR amenity group."""
    from source.agent.graph import extractor_node

    result = extractor_node({"messages": [HumanMessage(content=PROMPT_SEA_OR_WATER)]})
    constraints = _extractor_constraints_json(result["messages"])
    friday = next_friday()
    friday_iso = friday.isoformat()

    date_field = constraints.get("date")
    assert isinstance(date_field, dict), f"expected date object, got {date_field!r}"
    assert str(date_field.get("start"))[:10] == friday_iso
    assert str(date_field.get("end") or date_field.get("start"))[:10] == friday_iso

    group = _find_or_group(constraints.get("amenities") or [])
    assert group is not None, (
        f"expected one OR amenity group; got {constraints.get('amenities')!r}"
    )
    norms = {_norm_label(v) for v in group.get("values") or []}
    assert any("sea" in n for n in norms), norms
    assert any("body of water" in n or "water" in n for n in norms), norms

    for item in constraints.get("semantic_constraints") or []:
        q = item.get("query") if isinstance(item, dict) else item
        assert not (q and "friday" in str(q).lower()), f"date leaked into semantic: {q!r}"
    assert not any(
        isinstance(n, dict) and "date" in str(n.get("field", "")).lower()
        for n in (constraints.get("numeric_constraints") or [])
    )
