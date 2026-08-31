"""Extractor: summer weather + stargazing + Saturday afternoon one-night stay.

Currently fails: stay length and Saturday-afternoon arrival leak into
semantic_constraints. Weather + stargazing belong there; one night is the
date window; Saturday afternoon arrival is a policy/check-in search (schema TBD).
"""

from __future__ import annotations

import json
from datetime import timedelta

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from source.agent.dates import today_il, weekday_next_iso_week, weekday_this_iso_week

load_dotenv()

PROMPT_HE_SUMMER_STARS = (
    "מקום עם מזג אוויר נחמד בקיץ שאפשר לראות בו כוכבים "
    "ואפשר להגיע בשבת בצהריים ללילה אחד עד ראשון"
)


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


def _upcoming_saturday():
    today = today_il()
    this_sat = weekday_this_iso_week(today, 5)
    if this_sat >= today:
        return this_sat
    return weekday_next_iso_week(today, 5)


def test_extractor_nice_weather_stars_saturday_afternoon_one_night():
    """Weather + stargazing are semantic; one night is the date; Saturday PM is not RAG."""
    from source.agent.graph import extractor_node

    result = extractor_node({"messages": [HumanMessage(content=PROMPT_HE_SUMMER_STARS)]})
    constraints = _extractor_constraints_json(result["messages"])
    semantic = constraints.get("semantic_constraints") or []
    texts = [_norm(t) for t in _semantic_texts(semantic)]

    assert any("weather" in t for t in texts), (
        f"expected summer weather in semantic_constraints={semantic!r}"
    )
    assert any("star" in t for t in texts), (
        f"expected stargazing in semantic_constraints={semantic!r}"
    )

    leaked = [
        t
        for t in texts
        if any(
            needle in t
            for needle in (
                "one night",
                "1 night",
                "single night",
                "overnight",
                "nights",
            )
        )
    ]
    assert not leaked, (
        f"stay length belongs on date_intent.nights, not semantic: {leaked!r} "
        f"(full semantic_constraints={semantic!r})"
    )

    policy_leaked = [
        t
        for t in texts
        if any(
            needle in t
            for needle in (
                "saturday afternoon",
                "arrive saturday",
                "arrival",
                "check in",
                "afternoon",
            )
        )
    ]
    assert not policy_leaked, (
        f"Saturday afternoon arrival is a policy/check-in search, not semantic RAG: "
        f"{policy_leaked!r} (full semantic_constraints={semantic!r})"
    )

    intent = constraints.get("date_intent") or {}
    assert str(intent.get("kind") or "").lower() == "weekday", intent
    assert str(intent.get("weekday") or "").lower() == "saturday", intent
    assert intent.get("nights") == 1, intent
    assert not intent.get("horizon_days"), intent

    saturday = _upcoming_saturday()
    date_field = constraints.get("date")
    assert isinstance(date_field, dict), f"expected date object, got {date_field!r}"
    assert str(date_field.get("start"))[:10] == saturday.isoformat(), (
        f"date={date_field!r} date_intent={intent!r}"
    )
    assert str(date_field.get("end") or "")[:10] == (
        saturday + timedelta(days=1)
    ).isoformat(), (
        f"one night Sat→Sun, not a two-night window: {date_field!r} "
        f"date_intent={intent!r}"
    )
    assert "amenities" not in constraints
    assert not constraints.get("campsite")
