"""Filter funnel shown after the recommended stays."""

from __future__ import annotations

from typing import Any


def query_is_hebrew(query: str) -> bool:
    """True when the query has more Hebrew letters than Latin ones."""
    hebrew = 0
    latin = 0
    for char in query or "":
        if "\u0590" <= char <= "\u05ff":
            hebrew += 1
        elif ("A" <= char <= "Z") or ("a" <= char <= "z"):
            latin += 1
    return hebrew > latin


def _site_list(step: dict[str, Any]) -> list[str]:
    raw = step.get("sites")
    if not isinstance(raw, list):
        return []
    return [str(name).strip() for name in raw if str(name).strip()]


def _bracket(sites: list[str]) -> str:
    return "[" + ", ".join(sites) + "]"


def _price_sentence(sites: list[str], *, hebrew: bool) -> str:
    listed = _bracket(sites)
    count = len(sites)
    if hebrew:
        if count == 1:
            return f"אתר נוסף עם זמינות {listed} אבל הוא מחוץ לטווח המחיר."
        return f"{count} אתרים נוספים עם זמינות {listed} אבל הם מחוץ לטווח המחיר."
    if count == 1:
        return f"1 other site has availability {listed} but it is outside the price range."
    return (
        f"{count} other sites have availability {listed} "
        "but they are outside the price range."
    )


def _missing_sentence(sites: list[str], query: str, *, hebrew: bool) -> str:
    listed = _bracket(sites)
    count = len(sites)
    if hebrew:
        if count == 1:
            return f"אתר נוסף עם זמינות {listed} אבל אין לו {query}."
        return f"{count} אתרים נוספים עם זמינות {listed} אבל אין להם {query}."
    if count == 1:
        return f"1 other site has availability {listed} but it doesn't have {query}."
    return f"{count} other sites have availability {listed} but they don't have {query}."


def _step_line(step: dict[str, Any], *, hebrew: bool) -> str:
    sites = _site_list(step)
    if not sites:
        return ""
    stage = step.get("stage")
    if stage == "price":
        return _price_sentence(sites, hebrew=hebrew)
    if stage == "missing":
        query = str(step.get("query") or "").strip()
        if not query:
            return ""
        return _missing_sentence(sites, query, hebrew=hebrew)
    return ""


def render_why_not(steps: list[dict[str, Any]] | None, *, hebrew: bool) -> str:
    """Named sites that were available, and why each group was left out."""
    lines: list[str] = []
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        text = _step_line(step, hebrew=hebrew)
        if text:
            lines.append(text)
    return "\n".join(lines)
