"""Filter funnel shown after the recommended stays."""

from __future__ import annotations

from typing import Any, NamedTuple


class _WhyGroup(NamedTuple):
    stage: str
    query: str


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


def _names(sites: list[str], *, hebrew: bool) -> str:
    if len(sites) == 1:
        return f"[{sites[0]}]"
    joiner = " ו" if hebrew else " and "
    return "[" + joiner.join(sites) + "]"


def _requested_price(constraints: dict[str, Any] | None, *, hebrew: bool) -> str:
    """The price bound the user asked for, in the reply's language."""
    numeric = (constraints or {}).get("numeric_constraints")
    if not isinstance(numeric, list):
        return ""
    for item in numeric:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field") or "").lower()
        if field not in {"price_per_night", "price", "cost"}:
            continue
        try:
            value = float(item.get("value"))
        except (TypeError, ValueError):
            return ""
        amount = str(int(value)) if value.is_integer() else f"{value:g}"
        op = str(item.get("operator") or "<=")
        if hebrew:
            if op in {">=", "=>"}:
                return f"מ-{amount} ₪"
            if op in {"=", "=="}:
                return f"{amount} ₪"
            return f"עד {amount} ₪"
        if op in {">=", "=>"}:
            return f"at least {amount} NIS"
        if op in {"=", "=="}:
            return f"{amount} NIS"
        return f"up to {amount} NIS"
    return ""


def _price_sentence(sites: list[str], *, hebrew: bool, bound: str) -> str:
    count = len(sites)
    limit = f" ({bound})" if bound else ""
    if count > 2:
        if hebrew:
            return f"{count} מקומות לינה מחוץ לטווח המחיר{limit}."
        return f"{count} campsite slots are outside the price range{limit}."
    listed = _names(sites, hebrew=hebrew)
    if hebrew:
        if count == 1:
            return f"מקום לינה אחד {listed} מחוץ לטווח המחיר{limit}."
        return f"{count} מקומות לינה {listed} מחוץ לטווח המחיר{limit}."
    if count == 1:
        return f"1 campsite slot {listed} is outside the price range{limit}."
    return f"{count} campsite slots {listed} are outside the price range{limit}."


def _missing_sentence(sites: list[str], query: str, *, hebrew: bool) -> str:
    count = len(sites)
    if count > 2:
        if hebrew:
            return f"{count} אתרים בלי אינדיקציה על {query} ."
        return f"{count} campsites don't have an indication of {query} ."
    listed = _names(sites, hebrew=hebrew)
    if hebrew:
        note = f"בלי אינדיקציה על {query}"
        if count == 1:
            return f"אתר אחד {listed} {note}."
        return f"{count} אתרים {listed} {note}."
    if count == 1:
        return f"1 campsite {listed} doesn't have an indication of {query}."
    return f"{count} campsites {listed} don't have an indication of {query}."


def _rule_sentence(sites: list[str], query: str, *, hebrew: bool) -> str:
    count = len(sites)
    if count > 2:
        if hebrew:
            return f"{count} אתרים עם כלל שלא מתיר {query}."
        return f"{count} campsites have a rule that doesn't allow {query}."
    listed = _names(sites, hebrew=hebrew)
    if hebrew:
        if count == 1:
            return f"אתר אחד {listed} עם כלל שלא מתיר {query}."
        return f"{count} אתרים {listed} עם כלל שלא מתיר {query}."
    if count == 1:
        return f"1 campsite {listed} has a rule that doesn't allow {query}."
    return f"{count} campsites {listed} have a rule that doesn't allow {query}."


def _step_line(
    step: dict[str, Any], *, hebrew: bool, price_bound: str
) -> str:
    sites = _site_list(step)
    if not sites:
        return ""
    stage = step.get("stage")
    if stage == "price":
        return _price_sentence(sites, hebrew=hebrew, bound=price_bound)
    if stage == "missing":
        query = str(step.get("query") or "").strip()
        if not query:
            return ""
        return _missing_sentence(sites, query, hebrew=hebrew)
    if stage == "rule":
        query = str(step.get("query") or "").strip()
        if not query:
            return ""
        return _rule_sentence(sites, query, hebrew=hebrew)
    return ""


def render_why_not(
    steps: list[dict[str, Any]] | None,
    *,
    hebrew: bool,
    constraints: dict[str, Any] | None = None,
) -> str:
    """Named sites that were available, and why each group was left out."""
    bound = _requested_price(constraints, hebrew=hebrew)
    lines: list[str] = []
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        text = _step_line(step, hebrew=hebrew, price_bound=bound)
        if text:
            lines.append(text)
    return "\n".join(lines)


def _rule_forbids(row: dict, query: str) -> bool:
    """A stored rule for this ask whose polarity is false."""
    raw = row.get("campsite_rules")
    rules: list = []
    if isinstance(raw, dict):
        rules = list(raw.get(query) or [])
    elif isinstance(raw, list):
        rules = list(raw)
    return any(
        isinstance(rule, dict) and rule.get("polarity") is False for rule in rules
    )


def fold_unverified_into_why_not(
    payload: dict[str, Any],
    *,
    kept: list[dict],
    dropped: list[dict],
) -> None:
    """Sites the judge removes join why-not.

    why_not is built before the judge. A loose amenity hit can put a
    campsite in fits, and the judge can then drop it. A polarity-false
    rule for that ask is a rule line; otherwise it is a missing-amenity
    line. Price misses stay on the quote's price line.
    """
    if not dropped:
        return
    still_fit = {
        int(row["campsite_id"])
        for row in kept
        if row.get("campsite_id") is not None
    }
    steps = [
        dict(step)
        for step in (payload.get("why_not") or [])
        if isinstance(step, dict)
    ]
    named = {
        str(name)
        for step in steps
        for name in (step.get("sites") or [])
    }
    by_group: dict[_WhyGroup, list[str]] = {}
    seen: dict[_WhyGroup, set[int]] = {}
    for row in dropped:
        cid = row.get("campsite_id")
        if cid is None:
            continue
        cid = int(cid)
        if cid in still_fit:
            continue
        name = str(row.get("campsite") or "").strip() or str(cid)
        if name in named:
            continue
        for verdict in row.get("claim_judge") or []:
            if not isinstance(verdict, dict) or verdict.get("satisfies"):
                continue
            query = str(verdict.get("query") or "").strip()
            if not query:
                continue
            group = _WhyGroup(
                "rule" if _rule_forbids(row, query) else "missing",
                query,
            )
            if cid in seen.setdefault(group, set()):
                continue
            seen[group].add(cid)
            by_group.setdefault(group, []).append(name)
            named.add(name)
            break
    if not by_group:
        return
    for group, names in by_group.items():
        for step in steps:
            if step.get("stage") == group.stage and step.get("query") == group.query:
                sites = list(step.get("sites") or [])
                sites.extend(name for name in names if name not in sites)
                step["sites"] = sites
                step["count"] = len(sites)
                break
        else:
            steps.append(
                {
                    "stage": group.stage,
                    "count": len(names),
                    "query": group.query,
                    "sites": names,
                }
            )
    payload["why_not"] = steps
