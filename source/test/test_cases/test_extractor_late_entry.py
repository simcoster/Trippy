"""Extractor: late arrival after 19:00 is planned_entry_time, not semantic."""

from __future__ import annotations

import json

import pytest
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

pytestmark = pytest.mark.llm

PROMPT = (
    "מקום ל4 אנשים ביום רביעי בעוד שבועיים עם בריכות לילדים "
    "2 מבוגרים ו2 ילדים ואפשר להיכנס אחרי 19"
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


def test_extractor_late_entry_after_19_is_planned_entry_time():
    """ואפשר להיכנס אחרי 19 → planned_entry_time 19:00, not a semantic query."""
    from source.agent.graph import extractor_node

    result = extractor_node({"messages": [HumanMessage(content=PROMPT)]})
    constraints = _extractor_constraints_json(result["messages"])
    assert constraints.get("planned_entry_time") == "19:00", constraints

    leaked = [
        t
        for t in _semantic_texts(constraints.get("semantic_constraints") or [])
        if any(
            needle in _norm(t)
            for needle in ("arriv", "check in", "entry", "after 19", "19:00")
        )
    ]
    assert not leaked, (
        f"late arrival belongs on planned_entry_time, not semantic: {leaked!r} "
        f"(full={constraints!r})"
    )
