"""Planner claim/rule judge: relevant evidence + satisfies, one 235B call."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from source.agent.planner import CLAIM_EVIDENCE_LIMIT
from source.agent.timing import record_stage, stage
from source.scraper.amenity_enrichment.llm import (
    GLM_INSTRUCT_MODEL,
    QWEN_INSTRUCT_30B_MODEL,
    QWEN_INSTRUCT_MODEL,
    LlmUsage,
    _parse_json_payload,
    collected_llm_usage,
    instruct_chat_model,
    make_nebius_openai_client,
)

logger = logging.getLogger(__name__)

CLAIM_JUDGE_SYSTEM = """
You judge review claims and official campsite rules for ONE campsite against
one guest request. This feed is for a planner: two decisions, not one.

The claims were retrieved with a loose vector gate (−0.6), top 5. Official
listing rows (amenities/rules) were retrieved at −0.7. MOST hits of both
kinds are unrelated. Distance does not mean relevance. Do not infer from
the campsite name.

Each claim has is_positive (true = praise / feature present / allowed;
false = complaint / missing / forbidden). Use that polarity.

A rule has subject, polarity (true = allowed/provided, false =
forbidden/not provided, null = quantity), and the source sentence. A
tent/cabin/room/hut subject is lodging, not a location or a vibe.
electric_stove / kettle is cooking, not campsite electricity.
A caravan-bay /water hookup (עמדת חניה לקרוואן) only serves a
guest who brought a caravan; it does not satisfy electricity for a
tent/room stay. electric_outlet in a bungalow/room DOES. Site-wide
נקודות חשמל and a PITCH tent with חיבור חשמל DO (limited coverage is a
caveat, not a no).
area:* / region:* claims name the parks.org.il area. They satisfy a
request for that area or region by name (north, Negev, Dead Sea,
Western Galilee). They do not satisfy "near the sea" / beach / ליד הים
just because the slug or Hebrew contains sea/ים. Dead Sea and Kinneret
are named places, not the Mediterranean or Red Sea coast. A review
about a beach or ים התיכון does.

1. relevant_claims: every claim that is actually about the request, including
   complaints and forbiddens. Keep all of those even when satisfies is true
   — nos are evidence for the recommender, not a planner veto. Do not keep
   passing mentions or a different fact that happens to share a word.
   - "Pets are not allowed" IS relevant to "pet friendly" (it is about pets).
   - "Staff is friendly" / "family-friendly" is NOT relevant to "pet friendly".
   - "Campfires are allowed" is NOT relevant to "desert".
   - "despite being in the desert" IS relevant to "in the desert".
   - "desert animals on the drive" is wildlife, not that the site is in the
     desert — not relevant.
   - "region:dead-sea" / "ארץ ים המלח" is NOT relevant to "near the sea"
     (shared word, different place).
   - Quote claim text exactly as given. Empty list if none are about it.

2. satisfies: true iff a relevant claim says yes OR a campsite rule
   grants the request. A no never decides this. Complaints,
   is_positive=false, polarity-false rules, and "not provided" do not veto
   a yes; leave satisfies true and keep those nos in relevant_claims.
   satisfies is false only when there is no granting claim and no granting
   rule (unrelated hits, or only nos).
   - Feature / vibe / permission → is_positive true on a relevant claim,
     or a rule whose polarity is true (provided/allowed).
   - "Pets are not allowed" does NOT itself satisfy "pet friendly".
   - dogs_allowed with polarity false does NOT itself satisfy "pet friendly".
   - A tent/cabin/room rule does NOT satisfy "desert" or "quiet".
   - electric_stove does NOT satisfy "electricity".
   - A caravan-bay hookup does NOT satisfy "electricity" without a caravan.
   - "No electricity at the tent" does NOT itself satisfy "electricity". If a
     rule grants electricity, satisfies is still true.
   - A concessive aside counts ("despite being in the desert" satisfies
     "in the desert").
   - Limited coverage still satisfies ("electricity is available, though it
     does not reach every spot"; electric_hookup polarity true).
   - area:north DOES satisfy "in the north". region:negev DOES satisfy
     "desert". region:dead-sea does NOT satisfy "near the sea".

Examples:
Request "pet friendly". Claim "Pets are not allowed at the site."
is_positive=false. Rule dogs_allowed polarity=false.
→ {"relevant_claims": ["Pets are not allowed at the site."],
   "satisfies": false, "satisfy_by": null,
   "reason": "no granting claim or rule; only forbids"}

Request "pet friendly". Claim "The staff is friendly." is_positive=true.
Rule dogs_allowed polarity=false.
→ {"relevant_claims": [], "satisfies": false, "satisfy_by": null,
   "reason": "staff-friendly is not about pets; no granting pet rule"}

Request "desert". Claim "Campfires are allowed." is_positive=true.
Rule tent polarity=true.
→ {"relevant_claims": [], "satisfies": false, "satisfy_by": null,
   "reason": "unrelated claims and rules"}

Request "in the desert". Claim "The tent is clean despite being in the
desert with winds." is_positive=true. Rule tent polarity=true.
→ {"relevant_claims": ["The tent is clean despite being in the desert with winds."],
   "satisfies": true, "satisfy_by": "claim",
   "reason": "concessive aside states the site is in the desert"}

Request "fridge". Claim "The kitchen fridge needs more shelves."
is_positive=true. Rule refrigerator polarity=true, evidence "מקררים (3)".
→ {"relevant_claims": ["The kitchen fridge needs more shelves."],
   "satisfies": true, "satisfy_by": "rule",
   "reason": "official listing provides fridges; complaint is a caveat"}

Request "electricity". Claim "No electricity at the tent." is_positive=false.
Rule electric_hookup polarity=true.
→ {"relevant_claims": ["No electricity at the tent."],
   "satisfies": true, "satisfy_by": "rule",
   "reason": "official hookup grants it; the complaint is a caveat"}

Request "electricity". Claim "Electricity is available, though it does
not reach every spot." is_positive=true. Rule electric_hookup polarity=true.
→ {"relevant_claims": ["Electricity is available, though it does not reach every spot."],
   "satisfies": true, "satisfy_by": "both",
   "reason": "feature present; official hookup; limited coverage is a caveat"}

Request "near the sea". Claim "region:dead-sea" is_positive=true.
Claim "ארץ ים המלח" is_positive=true.
→ {"relevant_claims": [], "satisfies": false, "satisfy_by": null,
   "reason": "Dead Sea region is a named place, not the sea coast"}

Request "near the sea". Claim "Access to the beach is accessible all
the way to the water." is_positive=true.
→ {"relevant_claims": ["Access to the beach is accessible all the way to the water."],
   "satisfies": true, "satisfy_by": "claim",
   "reason": "beach access is the sea coast"}

Output JSON only:
{"relevant_claims": [str], "satisfies": bool,
 "satisfy_by": "claim" | "rule" | "both" | null, "reason": str}
""".strip()

CLAIM_JUDGE_COMPACT_SUFFIX = """
COMPACT OUTPUT. The examples above quote claim strings in relevant_claims;
do not copy that shape. Each claim in the user JSON has i. relevant is
those i values (ints), never the claim text. reason is 4-5 English words
maximum.

Examples:
Request "pet friendly". Claim i=0 "Pets are not allowed at the site."
→ {"relevant": [0], "satisfies": false, "satisfy_by": null,
   "reason": "only forbids pets"}

Request "in the desert". Claim i=0 "The tent is clean despite being in
the desert with winds."
→ {"relevant": [0], "satisfies": true, "satisfy_by": "claim",
   "reason": "concessive desert aside"}

Request "electricity". Claim i=0 "No electricity at the tent."
Rule electric_hookup polarity=true.
→ {"relevant": [0], "satisfies": true, "satisfy_by": "rule",
   "reason": "hookup grants electricity"}

Request "near the sea". Claim i=0 "region:dead-sea". Claim i=1
"ארץ ים המלח".
→ {"relevant": [], "satisfies": false, "satisfy_by": null,
   "reason": "Dead Sea not coast"}

Output JSON only:
{"relevant": [int], "satisfies": bool,
 "satisfy_by": "claim" | "rule" | "both" | null, "reason": str}
""".strip()

CLAIM_JUDGE_BATCH_SUFFIX = """
BATCH. The user JSON is jobs[], one object per (campsite, request) with i.
Judge EACH job independently with the same rules above. Do not let one
job's rules or claims decide another.

Return JSON only:
{"judgements": [
  {"i": 0, "relevant": [int], "satisfies": bool,
   "satisfy_by": "claim" | "rule" | "both" | null, "reason": str},
  ...
]}
One object per job, same i, same order. reason is 4-5 English words.
""".strip()


def _norm(text: str) -> str:
    return " ".join((text or "").split())


def _why_query(entry: dict[str, Any]) -> str | None:
    query = entry.get("query")
    if isinstance(query, list):
        return str(query[0]) if query else None
    if isinstance(query, str) and query.strip():
        return query
    return None


def _compact_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for claim in claims:
        text = claim.get("claim")
        if not text:
            continue
        out.append(
            {"claim": text, "is_positive": claim.get("is_positive")}
        )
    return out


def _compact_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rule in rules:
        if rule.get("error"):
            continue
        row = {
            "subject": rule.get("subject"),
            "polarity": rule.get("polarity"),
            "evidence_span": rule.get("evidence_span"),
        }
        if rule.get("qualifier") is not None:
            row["qualifier"] = rule.get("qualifier")
        out.append(row)
    return out


_USAGE_LOCK = threading.Lock()


def judge_concurrency() -> int:
    raw = (os.environ.get("TRIPPY_JUDGE_CONCURRENCY") or "5").strip()
    try:
        n = int(raw)
    except ValueError:
        return 5
    return max(1, n)


def judge_compact() -> bool:
    raw = (os.environ.get("TRIPPY_JUDGE_COMPACT") or "1").strip().casefold()
    return raw not in {"0", "false", "no", "off"}


def judge_batch() -> bool:
    raw = (os.environ.get("TRIPPY_JUDGE_BATCH") or "0").strip().casefold()
    return raw in {"1", "true", "yes", "on"}


def judge_model() -> str:
    """Judge model. `TRIPPY_JUDGE_MODEL` (glm / 30B / a full id) else instruct."""
    raw = (os.environ.get("TRIPPY_JUDGE_MODEL") or "").strip()
    key = raw.casefold()
    if not raw:
        return instruct_chat_model()
    if key in {"glm", "glm-5.2", "glm52"}:
        return GLM_INSTRUCT_MODEL
    if key in {"30b", "little", "small"}:
        return QWEN_INSTRUCT_30B_MODEL
    if key in {"235b", "big"}:
        return QWEN_INSTRUCT_MODEL
    return raw


def _judge_system() -> str:
    if judge_compact():
        return CLAIM_JUDGE_SYSTEM + "\n\n" + CLAIM_JUDGE_COMPACT_SUFFIX
    return CLAIM_JUDGE_SYSTEM


def _no_think_kwargs() -> dict[str, Any]:
    return {
        "reasoning_effort": "none",
        "extra_body": {
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "thinking": {"type": "disabled"},
        },
    }


def _judge_chat(api: Any, **kwargs: Any) -> Any:
    extra = _no_think_kwargs()
    try:
        return api.chat.completions.create(**kwargs, **extra)
    except Exception:
        logger.warning("judge no-think extra_body rejected; retrying reasoning_effort")
        try:
            return api.chat.completions.create(
                **kwargs,
                reasoning_effort="none",
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception:
            return api.chat.completions.create(**kwargs, reasoning_effort="none")


def _empty_verdict() -> dict[str, Any]:
    return {
        "relevant_claims": [],
        "satisfies": False,
        "satisfy_by": None,
        "reason": "no claims or rules",
    }


def _unparseable_verdict(raw: str) -> dict[str, Any]:
    return {
        "relevant_claims": [],
        "satisfies": False,
        "satisfy_by": None,
        "reason": f"unparseable: {raw[:200]}",
    }


def _verdict_from_parsed(
    parsed: dict[str, Any],
    claim_rows: list[dict[str, Any]],
    *,
    compact: bool,
) -> dict[str, Any]:
    return {
        "relevant_claims": _relevant_claim_texts(
            parsed, claim_rows, compact=compact
        ),
        "satisfies": bool(parsed.get("satisfies")),
        "satisfy_by": parsed.get("satisfy_by"),
        "reason": str(parsed.get("reason") or ""),
    }


def _as_claim_index(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _relevant_claim_texts(
    parsed: dict[str, Any],
    claim_rows: list[dict[str, Any]],
    *,
    compact: bool,
) -> list[str]:
    """Map a judge payload to claim strings. Compact uses indices into claim_rows."""
    if not compact:
        rel = parsed.get("relevant_claims") or []
        if not isinstance(rel, list):
            return []
        return [str(x) for x in rel]
    raw = parsed.get("relevant")
    if raw is None:
        raw = parsed.get("relevant_claims") or []
    if not isinstance(raw, list):
        return []
    texts: list[str] = []
    seen: set[str] = set()
    for item in raw:
        idx = _as_claim_index(item)
        if idx is not None:
            if 0 <= idx < len(claim_rows):
                text = str(claim_rows[idx].get("claim") or "")
                key = _norm(text)
                if text and key not in seen:
                    seen.add(key)
                    texts.append(text)
            continue
        if isinstance(item, str) and item.strip():
            key = _norm(item)
            if key not in seen:
                seen.add(key)
                texts.append(item)
    return texts


def judge_site_request(
    *,
    query: str,
    campsite: str,
    claims: list[dict[str, Any]],
    rules: list[dict[str, Any]],
    usage: LlmUsage | None = None,
    client: Any | None = None,
    time_stage: bool = True,
) -> dict[str, Any]:
    """One instruct-model call: which claims are relevant, and whether the site satisfies."""
    if not claims and not rules:
        return {
            "relevant_claims": [],
            "satisfies": False,
            "satisfy_by": None,
            "reason": "no claims or rules",
        }
    compact = judge_compact()
    claim_rows = [
        {"claim": c.get("claim"), "is_positive": c.get("is_positive")}
        for c in claims
    ]
    if compact:
        claim_rows = [row for row in claim_rows if row.get("claim")]
        payload_claims = [{"i": i, **row} for i, row in enumerate(claim_rows)]
        note = (
            "Most claims and rules are probably not about the request. "
            "Always use both lists. satisfies is true if any source "
            "grants; nos belong in relevant (their i values) and do not veto."
        )
    else:
        payload_claims = claim_rows
        note = (
            "Most claims and rules are probably not about the request. "
            "Always use both lists. satisfies is true if any source "
            "grants; nos belong in relevant_claims and do not veto."
        )
    user = json.dumps(
        {
            "request": query,
            "campsite": campsite,
            "note": note,
            "claims": payload_claims,
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
    model = judge_model()
    api = client or make_nebius_openai_client()
    messages = [
        {"role": "system", "content": _judge_system()},
        {"role": "user", "content": user},
    ]
    max_tokens = 200 if compact else 600
    create = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if time_stage:
        with stage("judge"):
            response = api.chat.completions.create(**create)
    else:
        response = api.chat.completions.create(**create)
    if usage is not None:
        with _USAGE_LOCK:
            usage.add_chat(response.usage, role="claim_judge", model=model)
    raw = (response.choices[0].message.content or "").strip()
    try:
        parsed = _parse_json_payload(raw)
    except (json.JSONDecodeError, ValueError):
        logger.warning("claim_judge unparseable for %s %r: %s", campsite, query, raw[:200])
        return _unparseable_verdict(raw)
    return _verdict_from_parsed(parsed, claim_rows, compact=compact)


# Bound at import so a test patch of `judge_site_request` does not look like live Nebius.
_LIVE_JUDGE = judge_site_request


def _queries_for_fit(fit: dict[str, Any]) -> list[str]:
    why = list(fit.get("why") or [])
    claims = list(fit.get("review_claims") or [])
    return list(
        dict.fromkeys(
            q
            for q in [_why_query(w) for w in why]
            + [c.get("query") for c in claims]
            if isinstance(q, str) and q
        )
    )


def _rules_from_fit(fit: dict[str, Any], query: str) -> list[dict[str, Any]]:
    raw = fit.get("campsite_rules")
    if isinstance(raw, dict):
        return list(raw.get(query) or [])
    if isinstance(raw, list):
        return list(raw)
    return []


def _rules_by_site(
    query: str,
    campsite_ids: list[int],
    *,
    search_rules: Callable[..., list[dict[str, Any]]],
) -> dict[int, list[dict[str, Any]]]:
    hits = search_rules(query, limit=CLAIM_EVIDENCE_LIMIT, campsite_ids=campsite_ids)
    by_site: dict[int, list[dict[str, Any]]] = {int(i): [] for i in campsite_ids}
    for hit in hits:
        if hit.get("error"):
            continue
        cid = int(hit["campsite_id"])
        if cid in by_site:
            by_site[cid].append(hit)
    return by_site


def _claim_rows_for_job(
    claims: list[dict[str, Any]], *, compact: bool
) -> list[dict[str, Any]]:
    rows = [
        {"claim": c.get("claim"), "is_positive": c.get("is_positive")}
        for c in claims
    ]
    if compact:
        return [row for row in rows if row.get("claim")]
    return rows


def _parse_batch_rows(raw: str, n: int) -> list[dict[str, Any] | None]:
    parsed = _parse_json_payload(raw)
    rows = parsed.get("judgements")
    if not isinstance(rows, list):
        rows = parsed.get("judgments")
    if not isinstance(rows, list):
        raise ValueError("no judgements list")
    out: list[dict[str, Any] | None] = [None] * n
    for row in rows:
        if not isinstance(row, dict):
            continue
        idx = row.get("i")
        if isinstance(idx, int) and 0 <= idx < n:
            out[idx] = row
    if all(item is None for item in out) and len(rows) == n:
        for idx, row in enumerate(rows):
            if isinstance(row, dict):
                out[idx] = row
    return out


def _run_judge_jobs_batch(
    pending: dict[tuple[int, str], dict[str, Any]],
    *,
    usage: LlmUsage,
) -> dict[tuple[int, str], dict[str, Any]]:
    compact = judge_compact()
    model = judge_model()
    cache: dict[tuple[int, str], dict[str, Any]] = {}
    live_keys: list[tuple[int, str]] = []
    claim_rows_by_i: list[list[dict[str, Any]]] = []
    jobs_payload: list[dict[str, Any]] = []
    note = (
        "Most claims and rules are probably not about the request. "
        "Always use both lists. satisfies is true if any source "
        "grants; nos belong in relevant (their i values) and do not veto."
    )
    for key, job in pending.items():
        claims = list(job.get("claims") or [])
        rules = [r for r in (job.get("rules") or []) if not r.get("error")]
        if not claims and not rules:
            cache[key] = _empty_verdict()
            continue
        rows = _claim_rows_for_job(claims, compact=compact)
        live_keys.append(key)
        claim_rows_by_i.append(rows)
        payload_claims = (
            [{"i": j, **row} for j, row in enumerate(rows)] if compact else rows
        )
        jobs_payload.append(
            {
                "i": len(live_keys) - 1,
                "request": job["query"],
                "campsite": job["campsite"],
                "claims": payload_claims,
                "campsite_rules": [
                    {
                        "subject": r.get("subject"),
                        "category": r.get("category"),
                        "polarity": r.get("polarity"),
                        "qualifier": r.get("qualifier"),
                        "evidence_span": r.get("evidence_span"),
                    }
                    for r in rules
                ],
            }
        )
    if not live_keys:
        return cache
    user = json.dumps({"note": note, "jobs": jobs_payload}, ensure_ascii=False)
    system = _judge_system() + "\n\n" + CLAIM_JUDGE_BATCH_SUFFIX
    api = make_nebius_openai_client()
    started = time.perf_counter()
    response = _judge_chat(
        api,
        model=model,
        temperature=0,
        max_tokens=2000,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    record_stage("judge", time.perf_counter() - started, calls=1)
    usage.add_chat(response.usage, role="claim_judge", model=model)
    raw = (response.choices[0].message.content or "").strip()
    try:
        parsed_rows = _parse_batch_rows(raw, len(live_keys))
    except (json.JSONDecodeError, ValueError):
        logger.warning("claim_judge batch unparseable: %s", raw[:200])
        parsed_rows = [None] * len(live_keys)
    for i, key in enumerate(live_keys):
        row = parsed_rows[i]
        if row is None:
            cache[key] = _unparseable_verdict(raw)
            continue
        cache[key] = _verdict_from_parsed(row, claim_rows_by_i[i], compact=compact)
    return cache


def _run_judge_jobs(
    pending: dict[tuple[int, str], dict[str, Any]],
    *,
    judge_fn: Callable[..., dict[str, Any]],
    usage: LlmUsage,
    live: bool,
) -> dict[tuple[int, str], dict[str, Any]]:
    cache: dict[tuple[int, str], dict[str, Any]] = {}
    if not pending:
        return cache
    if live and judge_batch() and judge_fn is _LIVE_JUDGE:
        return _run_judge_jobs_batch(pending, usage=usage)
    workers = judge_concurrency() if live else 1
    extra: dict[str, Any] = {}
    if live and workers > 1 and judge_fn is _LIVE_JUDGE:
        extra["client"] = make_nebius_openai_client()
        extra["time_stage"] = False
    if workers <= 1:
        for key, job in pending.items():
            cache[key] = judge_fn(
                query=job["query"],
                campsite=job["campsite"],
                claims=job["claims"],
                rules=job["rules"],
                usage=usage,
            )
        return cache
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(
                judge_fn,
                query=job["query"],
                campsite=job["campsite"],
                claims=job["claims"],
                rules=job["rules"],
                usage=usage,
                **extra,
            ): key
            for key, job in pending.items()
        }
        for fut in as_completed(futs):
            cache[futs[fut]] = fut.result()
    record_stage("judge", time.perf_counter() - started, calls=len(pending))
    return cache


def apply_claim_rule_judgements(
    payload: dict[str, Any],
    *,
    judge: Callable[..., dict[str, Any]] | None = None,
    search_rules: Callable[..., list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Filter fits the judge says do not satisfy; keep relevant claims as evidence.

    Claims and official rules are already on the fit (planner retrieve).
    The judge does not embed or search. Listing hits at amenity −0.7 are
    recall (tent-as-desert, stove-as-electricity). The judge sifts those
    too, not only claim-only why (experiments.md 2026-09-07 §8). Nos in
    relevant_claims stay on the survivor for the recommender; they do not
    veto a granting rule or claim.
    """
    fits = list(payload.get("fits") or [])
    if not fits:
        return payload
    judge_fn = judge or judge_site_request
    usage = LlmUsage()
    site_ids = list(dict.fromkeys(int(f["campsite_id"]) for f in fits))
    rules_cache: dict[str, dict[int, list[dict[str, Any]]]] = {}
    pending: dict[tuple[int, str], dict[str, Any]] = {}
    kept: list[dict[str, Any]] = []
    extra_rejected: list[dict[str, Any]] = []

    def _rules_for_query(fit: dict[str, Any], query: str, cid: int) -> list[dict[str, Any]]:
        if search_rules is not None:
            if query not in rules_cache:
                rules_cache[query] = _rules_by_site(
                    query, site_ids, search_rules=search_rules
                )
            return rules_cache[query].get(cid) or []
        return _rules_from_fit(fit, query)

    for fit in fits:
        cid = int(fit["campsite_id"])
        claims = list(fit.get("review_claims") or [])
        queries = _queries_for_fit(fit)
        for query in queries:
            key = (cid, query)
            if key in pending:
                continue
            q_claims = [c for c in claims if c.get("query") == query] or claims
            pending[key] = {
                "query": query,
                "campsite": str(fit.get("campsite") or ""),
                "claims": q_claims,
                "rules": _rules_for_query(fit, query, cid),
            }

    cache = _run_judge_jobs(
        pending, judge_fn=judge_fn, usage=usage, live=judge is None
    )

    for fit in fits:
        cid = int(fit["campsite_id"])
        why = list(fit.get("why") or [])
        claims = list(fit.get("review_claims") or [])
        queries = _queries_for_fit(fit)
        if not queries:
            kept.append(fit)
            continue
        verdicts: list[dict[str, Any]] = []
        retrieved: list[dict[str, Any]] = []
        relevant_norm: set[str] = set()
        drop = False
        drop_reason = ""
        for query in queries:
            key = (cid, query)
            q_claims = [c for c in claims if c.get("query") == query] or claims
            q_rules = _rules_for_query(fit, query, cid)
            retrieved.append(
                {
                    "query": query,
                    "claims": _compact_claims(q_claims),
                    "rules": _compact_rules(q_rules),
                }
            )
            verdict = cache[key]
            verdicts.append({"query": query, **verdict})
            relevant_norm.update(_norm(t) for t in verdict.get("relevant_claims") or [])
        for entry in why:
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
        if retrieved:
            fit["retrieved"] = retrieved
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
    sink = collected_llm_usage()
    if sink is not None:
        sink.merge(usage)
    return payload
