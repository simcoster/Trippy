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


def _grouped(sites: list[str], reason: str, *, hebrew: bool, one: str) -> str:
    count = len(sites)
    if count > 2:
        if hebrew:
            return f"{count} אתרים אומרים {reason}."
        return f"{count} campsites say {reason}."
    listed = _names(sites, hebrew=hebrew)
    if hebrew:
        if count == 1:
            return f"אתר אחד {listed} אומר {one}."
        return f"{count} אתרים {listed} אומרים {reason}."
    if count == 1:
        return f"1 campsite {listed} says that {one}."
    return f"{count} campsites {listed} say that {reason}."


def _price_sentence(sites: list[str], *, hebrew: bool) -> str:
    if hebrew:
        reason = "שהם מחוץ לטווח המחיר"
        one = "שהוא מחוץ לטווח המחיר"
    else:
        reason = "they are outside the price range"
        one = "it is outside the price range"
    return _grouped(sites, reason, hebrew=hebrew, one=one)


def _missing_sentence(sites: list[str], query: str, *, hebrew: bool) -> str:
    if hebrew:
        reason = f"שאין להם {query}"
        one = f"שאין לו {query}"
    else:
        reason = f"they don't have {query}"
        one = f"it doesn't have {query}"
    return _grouped(sites, reason, hebrew=hebrew, one=one)


def _rule_sentence(sites: list[str], query: str, *, hebrew: bool) -> str:
    if hebrew:
        reason = f"שכלל באתר לא מתיר {query}"
        one = reason
    else:
        reason = f"a rule doesn't allow {query}"
        one = reason
    return _grouped(sites, reason, hebrew=hebrew, one=one)


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
    if stage == "rule":
        query = str(step.get("query") or "").strip()
        if not query:
            return ""
        return _rule_sentence(sites, query, hebrew=hebrew)
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
