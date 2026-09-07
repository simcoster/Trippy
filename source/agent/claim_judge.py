"""Planner claim/rule judge: relevant evidence + satisfies, one 235B call."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from source.agent import search
from source.agent.planner import CLAIM_EVIDENCE_LIMIT
from source.scraper.amenity_enrichment.llm import (
    QWEN_INSTRUCT_MODEL,
    LlmUsage,
    _parse_json_payload,
    make_nebius_openai_client,
)

logger = logging.getLogger(__name__)

CLAIM_JUDGE_SYSTEM = """
You judge review claims and official campsite rules for ONE campsite against
one guest request. This feed is for a planner: two decisions, not one.

The claims were retrieved with a loose vector gate (−0.6), top 5. MOST are
probably unrelated. Distance does not mean relevance. Each claim has
is_positive (true = praise / feature present / allowed; false = complaint /
missing / forbidden). Use that polarity.

Official campsite_rules are also retrieved by embedding. MOST are unrelated
(tents and cabins will show up for a desert query). A rule has subject,
polarity (true = allowed/provided, false = forbidden/not provided, null =
quantity), and the source sentence. Use polarity. Do not infer from the
campsite name.

1. relevant_claims: every claim that is actually about the request, including
   complaints and forbiddens. Keep all of those. Do not keep passing mentions
   or a different fact that happens to share a word.
   - "Pets are not allowed" IS relevant to "pet friendly" (it is about pets).
   - "Staff is friendly" / "family-friendly" is NOT relevant to "pet friendly".
   - "Campfires are allowed" is NOT relevant to "desert".
   - "despite being in the desert" IS relevant to "in the desert".
   - "desert animals on the drive" is wildlife, not that the site is in the
     desert — not relevant.
   - Quote claim text exactly as given. Empty list if none are about it.

2. satisfies: true only if the guest would get what they asked for, from at
   least one relevant claim with matching polarity OR one campsite rule that
   grants the request.
   - Feature / vibe / permission → need is_positive true on a relevant claim,
     or a rule whose polarity is true (provided/allowed).
   - "Pets are not allowed" does NOT satisfy "pet friendly".
   - dogs_allowed with polarity false does NOT satisfy "pet friendly".
   - A tent/cabin/room rule does NOT satisfy "desert".
   - "No electricity at the tent" does NOT satisfy "electricity".
   - A concessive aside counts ("despite being in the desert" satisfies
     "in the desert").
   - Limited coverage still satisfies ("electricity is available, though it
     does not reach every spot"; electric_hookup polarity true).

Examples:
Request "pet friendly". Claim "Pets are not allowed at the site."
is_positive=false. Rule dogs_allowed polarity=false.
→ {"relevant_claims": ["Pets are not allowed at the site."],
   "satisfies": false, "satisfy_by": null,
   "reason": "mentions pets but forbids them; official rule forbids dogs"}

Request "pet friendly". Claim "The staff is friendly." is_positive=true.
Rule dogs_allowed polarity=false.
→ {"relevant_claims": [], "satisfies": false, "satisfy_by": null,
   "reason": "staff-friendly is not about pets; rule forbids dogs"}

Request "desert". Claim "Campfires are allowed." is_positive=true.
Rule tent polarity=true.
→ {"relevant_claims": [], "satisfies": false, "satisfy_by": null,
   "reason": "unrelated claims and rules"}

Request "in the desert". Claim "The tent is clean despite being in the
desert with winds." is_positive=true. Rule tent polarity=true.
→ {"relevant_claims": ["The tent is clean despite being in the desert with winds."],
   "satisfies": true, "satisfy_by": "claim",
   "reason": "concessive aside states the site is in the desert"}

Request "electricity". Claim "Electricity is available, though it does
not reach every spot." is_positive=true. Rule electric_hookup polarity=true.
→ {"relevant_claims": ["Electricity is available, though it does not reach every spot."],
   "satisfies": true, "satisfy_by": "both",
   "reason": "feature present; official hookup; limited coverage is a caveat"}

Output JSON only:
{"relevant_claims": [str], "satisfies": bool,
 "satisfy_by": "claim" | "rule" | "both" | null, "reason": str}
""".strip()


def _norm(text: str) -> str:
    return " ".join((text or "").split())


def _why_is_claim_only(entry: dict[str, Any]) -> bool:
    return bool(entry.get("claim")) and not entry.get("stated_amenity") and not entry.get(
        "site_amenity"
    )


def _why_query(entry: dict[str, Any]) -> str | None:
    query = entry.get("query")
    if isinstance(query, list):
        return str(query[0]) if query else None
    if isinstance(query, str) and query.strip():
        return query
    return None


def judge_site_request(
    *,
    query: str,
    campsite: str,
    claims: list[dict[str, Any]],
    rules: list[dict[str, Any]],
    usage: LlmUsage | None = None,
) -> dict[str, Any]:
    """One 235B call: which claims are relevant, and whether the site satisfies."""
    if not claims and not rules:
        return {
            "relevant_claims": [],
            "satisfies": False,
            "satisfy_by": None,
            "reason": "no claims or rules",
        }
    user = json.dumps(
        {
            "request": query,
            "campsite": campsite,
            "note": (
                "Most claims and rules are probably not about the request. "
                "relevant_claims can be forbiddens; satisfies is stricter."
            ),
            "claims": [
                {"claim": c.get("claim"), "is_positive": c.get("is_positive")}
                for c in claims
            ],
            "campsite_rules": [
                {
                    "subject": r.get("subject"),
                    "category": r.get("category"),
                    "polarity": r.get("polarity"),
                    "qualifier": r.get("qualifier"),
                    "evidence_span": r.get("evidence_span"),
                }
                for r in rules
                if not r.get("error")
            ],
        },
        ensure_ascii=False,
    )
    client = make_nebius_openai_client()
    response = client.chat.completions.create(
        model=QWEN_INSTRUCT_MODEL,
        temperature=0,
        max_tokens=600,
        messages=[
            {"role": "system", "content": CLAIM_JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    if usage is not None:
        usage.add_chat(response.usage, role="claim_judge", model=QWEN_INSTRUCT_MODEL)
    raw = (response.choices[0].message.content or "").strip()
    try:
        parsed = _parse_json_payload(raw)
    except (json.JSONDecodeError, ValueError):
        logger.warning("claim_judge unparseable for %s %r: %s", campsite, query, raw[:200])
        parsed = {
            "relevant_claims": [],
            "satisfies": False,
            "satisfy_by": None,
            "reason": f"unparseable: {raw[:200]}",
        }
    rel = parsed.get("relevant_claims") or []
    if not isinstance(rel, list):
        rel = []
    return {
        "relevant_claims": [str(x) for x in rel],
        "satisfies": bool(parsed.get("satisfies")),
        "satisfy_by": parsed.get("satisfy_by"),
        "reason": str(parsed.get("reason") or ""),
    }


def _rules_by_site(
    query: str,
    campsite_ids: list[int],
    *,
    search_rules: Callable[..., list[dict[str, Any]]] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    fetch = search_rules or search.search_campsite_rules
    kwargs: dict[str, Any] = {
        "limit": CLAIM_EVIDENCE_LIMIT,
        "campsite_ids": campsite_ids,
    }
    if search_rules is None:
        kwargs["embedding"] = search._query_vec_literal(query)
    hits = fetch(query, **kwargs)
    by_site: dict[int, list[dict[str, Any]]] = {int(i): [] for i in campsite_ids}
    for hit in hits:
        if hit.get("error"):
            continue
        cid = int(hit["campsite_id"])
        if cid in by_site:
            by_site[cid].append(hit)
    return by_site


def apply_claim_rule_judgements(
    payload: dict[str, Any],
    *,
    judge: Callable[..., dict[str, Any]] | None = None,
    search_rules: Callable[..., list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Filter claim-only fits that do not satisfy; keep relevant claims as evidence.

    A stated amenity still wins: the judge never vetoes an official listing.
    """
    fits = list(payload.get("fits") or [])
    if not fits:
        return payload
    judge_fn = judge or judge_site_request
    usage = LlmUsage()
    site_ids = list(dict.fromkeys(int(f["campsite_id"]) for f in fits))
    cache: dict[tuple[int, str], dict[str, Any]] = {}
    rules_cache: dict[str, dict[int, list[dict[str, Any]]]] = {}
    kept: list[dict[str, Any]] = []
    extra_rejected: list[dict[str, Any]] = []

    for fit in fits:
        cid = int(fit["campsite_id"])
        why = list(fit.get("why") or [])
        claims = list(fit.get("review_claims") or [])
        if not claims and not any(_why_is_claim_only(w) for w in why):
            kept.append(fit)
            continue
        queries = list(
            dict.fromkeys(
                q
                for q in [_why_query(w) for w in why]
                + [c.get("query") for c in claims]
                if isinstance(q, str) and q
            )
        )
        verdicts: list[dict[str, Any]] = []
        relevant_norm: set[str] = set()
        drop = False
        drop_reason = ""
        for query in queries:
            key = (cid, query)
            if key not in cache:
                if query not in rules_cache:
                    rules_cache[query] = _rules_by_site(
                        query, site_ids, search_rules=search_rules
                    )
                q_claims = [c for c in claims if c.get("query") == query] or claims
                cache[key] = judge_fn(
                    query=query,
                    campsite=str(fit.get("campsite") or ""),
                    claims=q_claims,
                    rules=rules_cache[query].get(cid) or [],
                    usage=usage,
                )
            verdict = cache[key]
            verdicts.append({"query": query, **verdict})
            relevant_norm.update(_norm(t) for t in verdict.get("relevant_claims") or [])
        for entry in why:
            if not _why_is_claim_only(entry):
                continue
            query = _why_query(entry)
            if query is None:
                continue
            verdict = cache.get((cid, query))
            if verdict is not None and not verdict.get("satisfies"):
                drop = True
                drop_reason = str(verdict.get("reason") or "claim_not_verified")
                break
        evidence = [c for c in claims if _norm(str(c.get("claim") or "")) in relevant_norm]
        fit = dict(fit)
        if evidence:
            fit["review_claims"] = evidence
        elif "review_claims" in fit:
            del fit["review_claims"]
        if verdicts:
            fit["claim_judge"] = verdicts
        if drop:
            why_out = list(why) + [{"reason": "claim_not_verified", "detail": drop_reason}]
            extra_rejected.append({**fit, "why": why_out})
            continue
        kept.append(fit)

    payload = dict(payload)
    payload["fits"] = kept
    rejected = list(payload.get("rejected") or [])
    payload["rejected"] = extra_rejected + rejected
    payload["rejected_count"] = int(payload.get("rejected_count") or 0) + len(
        extra_rejected
    )
    if usage.chat_calls:
        logger.info(
            "claim_judge %s calls $%.4f",
            usage.chat_calls,
            usage.cost_usd,
        )
    return payload
