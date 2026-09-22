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


def _availability_line(count: int, *, hebrew: bool) -> str:
    if hebrew:
        if count == 1:
            return "אתר קמפינג אחד עם זמינות"
        return f"{count} אתרי קמפינג עם זמינות"
    if count == 1:
        return "1 campsite has availability"
    return f"{count} campsites have availability"


def _price_line(count: int, *, hebrew: bool) -> str:
    if hebrew:
        return f"{count} בטווח המחיר"
    if count == 1:
        return "1 is in the price range"
    return f"{count} are in the price range"


def _missing_line(count: int, query: str, *, hebrew: bool) -> str:
    if hebrew:
        return f"{count} בלי {query} כמו שביקשת"
    if count == 1:
        return f"1 doesn't have {query} like you requested"
    return f"{count} don't have {query} like you requested"


def _step_line(step: dict[str, Any], *, hebrew: bool) -> str:
    stage = step.get("stage")
    count = step.get("count")
    if isinstance(count, bool) or not isinstance(count, int):
        return ""
    if stage == "availability":
        return _availability_line(count, hebrew=hebrew)
    if stage == "price":
        return _price_line(count, hebrew=hebrew)
    if stage == "missing":
        query = str(step.get("query") or "").strip()
        if not query:
            return ""
        return _missing_line(count, query, hebrew=hebrew)
    return ""


def render_why_not(steps: list[dict[str, Any]] | None, *, hebrew: bool) -> str:
    """Arrow chain: availability, then price, then each missed request."""
    lines: list[str] = []
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        text = _step_line(step, hebrew=hebrew)
        if text:
            lines.append(text)
    if len(lines) < 2:
        return ""
    head = "למה לא" if hebrew else "Why not"
    body = [lines[0]]
    body.extend(f"→ {line}" for line in lines[1:])
    return head + "\n" + "\n".join(body)
