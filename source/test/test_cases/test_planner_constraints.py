"""Desired planner constraint shape for date + amenity Hebrew queries.

Fails until the planner emits ``{date: [...], amenities: [...]}`` instead of
``semantic_constraints`` / ``numeric_constraints``.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

PROMPT = "אני רוצה משהו לשישי הבא עם מים זורמים"


def next_friday(today: date | None = None) -> date:
    """Upcoming Friday; if today is Friday, the Friday one week later."""
    today = today or date.today()
    days_ahead = (4 - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return today + timedelta(days=days_ahead)


def _planner_constraints_json(messages: list) -> dict:
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
        "planner_node did not return a JSON constraints AIMessage; "
        f"got {[type(m).__name__ for m in messages]}"
    )


def test_planner_next_friday_running_water_constraint_schema():
    """Hebrew trip ask → structured date + amenities (not semantic_constraints)."""
    from source.agent.graph import planner_node

    result = planner_node({"messages": [HumanMessage(content=PROMPT)]})
    constraints = _planner_constraints_json(result["messages"])

    expected_friday = next_friday().isoformat()  # YYYY-MM-DD

    assert "date" in constraints, (
        "expected key 'date' with next Friday ISO date(s); "
        f"got keys {sorted(constraints)}: {constraints}"
    )
    assert "amenities" in constraints, (
        "expected key 'amenities' (e.g. ['running water']); "
        f"got keys {sorted(constraints)}: {constraints}"
    )

    dates = constraints["date"]
    amenities = constraints["amenities"]
    assert isinstance(dates, list) and dates, f"date must be a non-empty list: {dates!r}"
    assert isinstance(amenities, list) and amenities, (
        f"amenities must be a non-empty list: {amenities!r}"
    )

    date_norm = {str(d).strip()[:10] for d in dates}
    assert expected_friday in date_norm, (
        f"expected next Friday {expected_friday} in date={dates!r}"
    )

    amenity_norm = {str(a).strip().lower().replace("_", " ") for a in amenities}
    assert "running water" in amenity_norm, (
        f"expected amenities to include 'running water'; got {amenities!r}"
    )
