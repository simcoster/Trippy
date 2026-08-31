"""Extractor constraint schema tests (date ranges, OR amenities, aliases)."""

from __future__ import annotations

import json
from datetime import date, timedelta

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from source.agent.constraints import (
    amenity_search_queries,
    claim_recency,
    normalize_constraints,
    semantic_search_queries,
)
from source.agent.dates import resolve_dates, today_il, weekday_next_iso_week

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


def _semantic_includes_running_water(semantic: list) -> bool:
    for item in semantic or []:
        if isinstance(item, str):
            if "running water" in item.lower().replace("_", " "):
                return True
        elif isinstance(item, dict):
            query = str(item.get("query") or "")
            if "running water" in query.lower().replace("_", " "):
                return True
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
    resolved = resolve_dates(
        kind="weekday", weekday="friday", when="next", nights=1, today=today
    )
    assert resolved["windows"] == [{"start": "2026-09-04", "end": "2026-09-05"}]


def test_resolve_this_friday_single_night():
    today = date(2026, 8, 27)  # Thursday
    resolved = resolve_dates(
        kind="weekday", weekday="friday", when="this", nights=1, today=today
    )
    assert resolved["windows"] == [{"start": "2026-08-28", "end": "2026-08-29"}]


def test_normalize_same_day_bumps_end_to_one_night():
    today = date(2026, 8, 27)
    out = normalize_constraints(
        {
            "date": {"start": "2026-08-28", "end": "2026-08-28"},
            "numeric_constraints": [],
            "semantic_constraints": [],
        },
        today=today,
    )
    assert out["date"] == {"start": "2026-08-28", "end": "2026-08-29"}
    assert out["date_windows"] == [{"start": "2026-08-28", "end": "2026-08-29"}]
    assert out["date_truncated"] is False
    assert out["date_notice"] is None
    assert set(out) == {
        "date",
        "date_windows",
        "date_truncated",
        "date_notice",
        "numeric_constraints",
        "semantic_constraints",
    }


def test_normalize_resolves_date_intent():
    today = date(2026, 8, 27)
    out = normalize_constraints(
        {
            "date_intent": {
                "kind": "weekday",
                "weekday": "friday",
                "when": "next",
                "nights": 1,
            },
            "semantic_constraints": [{"query": "quiet"}],
            "numeric_constraints": [],
        },
        today=today,
    )
    assert out["date"] == {"start": "2026-09-04", "end": "2026-09-05"}
    assert "amenities" not in out
    queries = [
        (s.get("query") if isinstance(s, dict) else s) for s in out["semantic_constraints"]
    ]
    assert any(q and "quiet" in str(q).lower() for q in queries)


def test_normalize_folds_amenities_into_semantic():
    today = date(2026, 8, 27)
    out = normalize_constraints(
        {
            "date_intent": {
                "kind": "weekday",
                "weekday": "friday",
                "when": "next",
                "nights": 1,
            },
            "amenities": [
                {
                    "op": "or",
                    "values": ["near the sea", "near a body of water"],
                }
            ],
            "numeric_constraints": [],
            "semantic_constraints": [{"query": "quiet"}],
        },
        today=today,
    )
    assert out["date"] == {"start": "2026-09-04", "end": "2026-09-05"}
    assert "amenities" not in out
    group = _find_or_group(out["semantic_constraints"])
    assert group is not None
    norms = {_norm_label(v) for v in group["values"]}
    assert "near the sea" in norms
    assert "near a body of water" in norms
    queries = [
        (s.get("query") if isinstance(s, dict) else s)
        for s in out["semantic_constraints"]
    ]
    assert any(q and "quiet" in str(q).lower() for q in queries)


def test_amenity_search_queries_or_group_no_query_expand():
    """Planner searches exact OR values; place→type expansion is LLM ingest only."""
    groups = semantic_search_queries(
        [{"op": "or", "values": ["near the sea", "near a body of water"]}]
    )
    assert groups == amenity_search_queries(
        [{"op": "or", "values": ["near the sea", "near a body of water"]}]
    )
    assert len(groups) == 1
    norms = {_norm_label(q) for q in groups[0]}
    assert "near the sea" in norms
    assert "near a body of water" in norms
    assert not any("kineret" in n for n in norms)


def test_claim_recency_iso_and_days_ago():
    day, days_ago = claim_recency("2026-05-22", today=date(2026, 8, 30))
    assert day == "2026-05-22"
    assert days_ago == 100


def test_claim_recency_missing():
    assert claim_recency(None, today=date(2026, 8, 30)) == (None, None)
    assert claim_recency("", today=date(2026, 8, 30)) == (None, None)


def test_semantic_evidence_payload_empty_queries():
    from source.agent.graph import _semantic_evidence_payload

    out = _semantic_evidence_payload([])
    assert out["stated_amenities"] == []
    assert out["review_claims"] == []
    assert "stated_amenity_hits" not in out
    assert "query" in out


# ---- Integration tests (LLM extractor) ----


def test_extractor_next_friday_running_water_constraint_schema():
    """Hebrew trip ask → structured date + semantic running water."""
    from source.agent.graph import extractor_node

    result = extractor_node({"messages": [HumanMessage(content=PROMPT_HE_RUNNING_WATER)]})
    constraints = _extractor_constraints_json(result["messages"])
    friday = weekday_next_iso_week(today_il(), 4)

    assert _date_covers_friday(constraints, friday), (
        f"expected next Friday {friday.isoformat()} in date={constraints.get('date')!r}"
    )
    assert "amenities" not in constraints, (
        f"amenities should be folded into semantic_constraints: {constraints}"
    )
    assert _semantic_includes_running_water(
        constraints.get("semantic_constraints") or []
    ), (
        f"expected running water in semantic_constraints="
        f"{constraints.get('semantic_constraints')!r}"
    )
    # Date must not live only in semantic/numeric
    for item in constraints.get("semantic_constraints") or []:
        q = item.get("query") if isinstance(item, dict) else item
        assert not (q and "friday" in str(q).lower()), f"date leaked into semantic: {q!r}"
    for item in constraints.get("numeric_constraints") or []:
        if isinstance(item, dict):
            assert "date" not in str(item.get("field", "")).lower()


def test_extractor_next_friday_sea_or_body_of_water():
    """English sea OR body-of-water + next Friday → date range + OR semantic group."""
    from source.agent.graph import extractor_node

    result = extractor_node({"messages": [HumanMessage(content=PROMPT_SEA_OR_WATER)]})
    constraints = _extractor_constraints_json(result["messages"])
    friday = weekday_next_iso_week(today_il(), 4)
    friday_iso = friday.isoformat()

    date_field = constraints.get("date")
    assert isinstance(date_field, dict), f"expected date object, got {date_field!r}"
    assert str(date_field.get("start"))[:10] == friday_iso
    assert str(date_field.get("end") or date_field.get("start"))[:10] == (
        friday + timedelta(days=1)
    ).isoformat()

    group = _find_or_group(constraints.get("semantic_constraints") or [])
    assert group is not None, (
        f"expected one OR semantic group; got "
        f"{constraints.get('semantic_constraints')!r}"
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
