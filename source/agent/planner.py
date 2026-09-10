"""Two-stage planner: vacancies then amenity intersection. Not a LangGraph node."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, TypeAlias

from source.agent import search
from source.agent.constraints import (
    ROOM_LOCUS,
    party_size_from_numeric,
    semantic_locus_groups,
)
from source.agent.dates import DATE_TRUNCATED_NOTICE, MAX_DATE_WINDOWS

AMENITY_MATCH_MAX_DISTANCE = -0.7
# Claims are whole sentences, so they sit further from a short query than a
# curated amenity label does — a looser gate than the amenity lane. −0.6 is
# recall; the 235B claim/rule judge (claim_judge.py) is precision.
# Amenity −0.7 is also recall (outlets/hookups at −0.75/−0.70, and tent-as-
# desert at −0.719). The same judge sifts listing hits (experiments.md
# 2026-09-07 §8).
CLAIM_MATCH_MAX_DISTANCE = -0.6
CLAIM_RECENCY_HALF_LIFE_DAYS = 365
CLAIM_EVIDENCE_LIMIT = 5
REJECTED_SAMPLE_LIMIT = 5

_WhyRows: TypeAlias = list[dict[str, Any]]
_ClaimsBySite: TypeAlias = dict[str, _WhyRows]
_RulesByCampsite: TypeAlias = dict[int, _WhyRows]
_RulesByQuery: TypeAlias = dict[str, _RulesByCampsite]


class _SlotKey(NamedTuple):
    """One vacant unit: a campsite plus an accommodation type."""

    campsite_id: str
    accommodation_type_id: int


def _slot_key(slot: dict) -> _SlotKey:
    return _SlotKey(
        campsite_id=str(slot["campsite_id"]),
        accommodation_type_id=int(slot["accommodation_type_id"]),
    )


@dataclass
class _SemanticWhy:
    """Retrieve result for vacant slots: who matched, who missed, evidence."""

    why_by_slot: dict[_SlotKey, _WhyRows]
    reject_why_by_slot: dict[_SlotKey, _WhyRows]
    claims_by_site: _ClaimsBySite
    rules_by_query: _RulesByQuery


def _semantic_evidence_payload(queries: list[str], *, limit: int = 5) -> dict:
    """Official amenity names + dated review claims for the recommender."""
    if not queries:
        return {
            "query": "",
            "stated_amenities": [],
            "review_claims": [],
        }
    stated_amenities: list[str] = []
    review_claims: list[dict] = []
    seen_amenities: set[str] = set()
    seen_claims: set[tuple] = set()
    for query in queries:
        vec = search._query_vec_literal(query)
        for hit in search.search_stated_amenities(query, limit=limit, embedding=vec):
            if hit.get("error"):
                continue
            label = str(hit.get("amenity") or "").strip()
            if label and label not in seen_amenities:
                seen_amenities.add(label)
                stated_amenities.append(label)
        for hit in search.search_review_claims(query, limit=limit, embedding=vec):
            if hit.get("error"):
                continue
            label = str(hit.get("claim") or "").strip()
            if not label:
                continue
            rec = {
                "claim": label,
                "date": hit.get("date"),
                "days_ago": hit.get("days_ago"),
                "campsite_id": hit.get("campsite_id"),
            }
            key = (rec["claim"], rec["date"], rec["campsite_id"])
            if key in seen_claims:
                continue
            seen_claims.add(key)
            review_claims.append(rec)
    return {
        "query": queries[0] if len(queries) == 1 else queries,
        "stated_amenities": stated_amenities,
        "review_claims": review_claims,
    }


def _amenity_hit_matches(hit: dict) -> bool:
    if hit.get("error") or hit.get("accommodation_type_id") is None:
        return False
    dist = hit.get("distance")
    if dist is None:
        return True
    return float(dist) <= AMENITY_MATCH_MAX_DISTANCE


def _site_amenity_hit_matches(hit: dict) -> bool:
    if hit.get("error") or hit.get("campsite_id") is None:
        return False
    dist = hit.get("distance")
    if dist is None:
        return True
    return float(dist) <= AMENITY_MATCH_MAX_DISTANCE


def _claim_hit_within_threshold(hit: dict) -> bool:
    if hit.get("error") or hit.get("campsite_id") is None:
        return False
    dist = hit.get("distance")
    if dist is None:
        return True
    return float(dist) <= CLAIM_MATCH_MAX_DISTANCE


def _claim_hit_satisfies(hit: dict) -> bool:
    """Only a positive claim can satisfy a constraint — negatives never veto."""
    return hit.get("is_positive") is True and _claim_hit_within_threshold(hit)


def _claim_weight(days_ago: Any) -> float:
    """Recency weight; an undated claim counts as exactly one half-life old."""
    try:
        days = float(days_ago)
    except (TypeError, ValueError):
        return 0.5
    return 0.5 ** (max(days, 0.0) / CLAIM_RECENCY_HALF_LIFE_DAYS)


def _claim_score(hits: list[dict[str, Any]]) -> float:
    """Positive claims lift, negative claims sink, both decayed by age."""
    score = 0.0
    for hit in hits:
        polarity = hit.get("is_positive")
        if polarity is None or not _claim_hit_within_threshold(hit):
            continue
        score += (1.0 if polarity else -1.0) * _claim_weight(hit.get("days_ago"))
    return score


def _claim_evidence(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Closest claims, capped, with both polarities represented when they exist."""
    ranked = sorted(
        hits,
        key=lambda h: (
            float(h["distance"]) if h.get("distance") is not None else 0.0
        ),
    )
    chosen = ranked[:CLAIM_EVIDENCE_LIMIT]
    rest = ranked[CLAIM_EVIDENCE_LIMIT:]
    polarities = {h.get("is_positive") for h in chosen if h.get("is_positive") is not None}
    if len(polarities) == 1 and chosen:
        missing = not polarities.pop()
        swap = next((h for h in rest if h.get("is_positive") is missing), None)
        if swap is not None:
            chosen = chosen[:-1] + [swap]
    return [
        {
            "query": hit.get("query"),
            "claim": hit.get("claim"),
            "date": hit.get("date"),
            "days_ago": hit.get("days_ago"),
            "is_positive": hit.get("is_positive"),
        }
        for hit in chosen
    ]


def _semantic_why_by_slot(
    slots: list[dict],
    semantic_constraints: list,
) -> _SemanticWhy:
    """AND-groups → matched/rejected why per slot, plus claims and
    official rules by site.

    A group is satisfied for a slot when any lane hits: the unit's stated
    amenities, the campsite's site-wide amenities (site locus only), or a
    positive review claim about that campsite. For a site-locus ask the
    claim and site-amenity lanes always both retrieve; satisfaction is OR.
    Official `campsite_rules` use the same query vector as claims so the
    judge does not embed or retrieve.
    """
    keys = list(dict.fromkeys(_slot_key(s) for s in slots))
    type_ids = list(dict.fromkeys(int(s["accommodation_type_id"]) for s in slots))
    site_ids = list(dict.fromkeys(int(s["campsite_id"]) for s in slots))
    claims_by_site: _ClaimsBySite = {str(sid): [] for sid in site_ids}
    rules_by_query: _RulesByQuery = {}
    if not keys:
        return _SemanticWhy(
            why_by_slot={},
            reject_why_by_slot={},
            claims_by_site=claims_by_site,
            rules_by_query=rules_by_query,
        )
    groups = semantic_locus_groups(semantic_constraints)
    if not groups:
        return _SemanticWhy(
            why_by_slot={key: [] for key in keys},
            reject_why_by_slot={},
            claims_by_site=claims_by_site,
            rules_by_query=rules_by_query,
        )

    why: dict[_SlotKey, _WhyRows] = {key: [] for key in keys}
    missing: dict[_SlotKey, _WhyRows] = {key: [] for key in keys}
    matching = set(keys)
    seen_claims: set[tuple] = set()

    for group in groups:
        is_room = group["locus"] == ROOM_LOCUS
        by_type: dict[int, dict[str, Any]] = {}
        by_site: dict[str, dict[str, Any]] = {}
        by_claim: dict[str, dict[str, Any]] = {}
        for query in group["queries"]:
            vec = search._query_vec_literal(query)
            if query not in rules_by_query:
                rules_by_query[query] = _rules_hits_by_site(
                    query, vec, site_ids
                )
            for hit in search.search_stated_amenities(
                query,
                limit=max(len(type_ids), 1),
                embedding=vec,
                accommodation_type_ids=type_ids,
            ):
                if not _amenity_hit_matches(hit):
                    continue
                tid = int(hit["accommodation_type_id"])
                by_type.setdefault(
                    tid,
                    {
                        "query": query,
                        "stated_amenity": hit.get("amenity"),
                        "distance": hit.get("distance"),
                    },
                )
            for hit in search.search_review_claims(
                query,
                limit=CLAIM_EVIDENCE_LIMIT,
                embedding=vec,
                campsite_ids=site_ids,
            ):
                if hit.get("error"):
                    continue
                cid = str(hit.get("campsite_id") or "")
                if cid not in claims_by_site:
                    continue
                key = (cid, hit.get("claim"), hit.get("date"))
                if key not in seen_claims:
                    seen_claims.add(key)
                    claims_by_site[cid].append({**hit, "query": query})
                if _claim_hit_satisfies(hit):
                    by_claim.setdefault(
                        cid,
                        {
                            "query": query,
                            "claim": hit.get("claim"),
                            "date": hit.get("date"),
                            "days_ago": hit.get("days_ago"),
                            "distance": hit.get("distance"),
                            "is_positive": True,
                        },
                    )
            if is_room:
                continue
            # Claim and site-amenity always both retrieve. A positive claim
            # must not skip the listing; satisfaction is either lane.
            for hit in search.search_site_amenities(
                query,
                limit=max(len(site_ids), 1),
                embedding=vec,
                campsite_ids=site_ids,
            ):
                if not _site_amenity_hit_matches(hit):
                    continue
                by_site.setdefault(
                    str(hit["campsite_id"]),
                    {
                        "query": query,
                        "site_amenity": hit.get("amenity"),
                        "distance": hit.get("distance"),
                    },
                )
        for key in keys:
            # Official evidence first, guest reviews last.
            hit = (
                by_type.get(key.accommodation_type_id)
                or by_site.get(key.campsite_id)
                or by_claim.get(key.campsite_id)
            )
            if hit is not None:
                why[key].append({**hit, "locus": "room"} if is_room else hit)
            else:
                miss: dict[str, Any] = {
                    "reason": (
                        "missing_room_amenity" if is_room else "missing_stated_amenity"
                    ),
                    "query": group["label"],
                }
                if is_room:
                    miss["locus"] = "room"
                missing[key].append(miss)
                matching.discard(key)
    return _SemanticWhy(
        why_by_slot={key: why[key] for key in keys if key in matching},
        reject_why_by_slot={
            key: missing[key] for key in keys if key not in matching
        },
        claims_by_site=claims_by_site,
        rules_by_query=rules_by_query,
    )


def _rules_hits_by_site(
    query: str, vec: str, site_ids: list[int]
) -> _RulesByCampsite:
    by_site: _RulesByCampsite = {int(i): [] for i in site_ids}
    for hit in search.search_campsite_rules(
        query,
        limit=CLAIM_EVIDENCE_LIMIT,
        embedding=vec,
        campsite_ids=site_ids,
    ):
        if hit.get("error"):
            continue
        cid = int(hit["campsite_id"])
        if cid in by_site:
            by_site[cid].append(hit)
    return by_site


def _rules_for_fit(
    rules_by_query: _RulesByQuery,
    campsite_id: int,
) -> dict[str, _WhyRows]:
    return {
        query: list(by_site.get(int(campsite_id)) or [])
        for query, by_site in rules_by_query.items()
    }


def _named_site_ids(name: str) -> tuple[list[int], dict[str, Any] | None]:
    hits = search.lookup_campsite_by_name(name)
    if not hits:
        return [], {"error": "No campsite matched that name", "query": name}
    if hits[0].get("error"):
        return [], {"error": hits[0]["error"], "query": name}
    ids = [
        int(hit["hotel_id"])
        for hit in hits
        if hit.get("hotel_id") is not None
    ]
    if not ids:
        return [], {"error": "No campsite matched that name", "query": name}
    return ids, None


def _stay_windows(constraints_json: dict) -> list[dict]:
    """Stay ranges to search, capped at MAX_DATE_WINDOWS."""
    raw = constraints_json.get("date_windows")
    windows: list[dict] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and item.get("start"):
                windows.append(item)
    if not windows:
        date_range = constraints_json.get("date")
        if isinstance(date_range, dict) and date_range.get("start"):
            windows.append(date_range)
    return windows[:MAX_DATE_WINDOWS]


def planner_fits_payload(constraints_json: dict) -> dict[str, Any]:
    numeric = constraints_json.get("numeric_constraints") or []
    semantic = constraints_json.get("semantic_constraints") or []
    payload: dict[str, Any] = {
        "fits": [],
        "rejected": [],
        "rejected_count": 0,
        "constraints": constraints_json,
    }
    windows = _stay_windows(constraints_json)
    if not windows:
        payload["skipped"] = "no_date"
        return payload
    if constraints_json.get("date_notice"):
        payload["date_notice"] = constraints_json["date_notice"]
    elif constraints_json.get("date_truncated"):
        payload["date_notice"] = DATE_TRUNCATED_NOTICE

    site_id: int | list[int] | None = None
    named = constraints_json.get("campsite")
    if named:
        site_ids, error = _named_site_ids(str(named))
        if error:
            payload["error"] = error["error"]
            payload["query"] = error.get("query")
            return payload
        site_id = site_ids if len(site_ids) > 1 else site_ids[0]

    slots: list[dict] = []
    query_records: list[Any] = []
    for window in windows:
        part = search.search_open_slots(
            date_range=window,
            site_id=site_id,
            party_size=party_size_from_numeric(numeric),
            numeric_constraints=numeric,
        )
        record = search._LAST_OPEN_SLOTS_QUERY
        if not isinstance(record, dict):
            record = {"date_range": window}
        else:
            record = {**record, "date_range": record.get("date_range") or window}
        query_records.append(record)
        if part and part[0].get("error"):
            payload["error"] = part[0]["error"]
            payload["open_slots_query"] = (
                query_records[0] if len(query_records) == 1 else query_records
            )
            return payload
        slots.extend(part)
    payload["open_slots_query"] = (
        query_records[0] if len(query_records) == 1 else query_records
    )

    found = _semantic_why_by_slot(slots, semantic)
    fits: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    def _slot_row(slot: dict, why: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "campsite_id": slot["campsite_id"],
            "campsite": slot["campsite"],
            "accommodation_type_id": int(slot["accommodation_type_id"]),
            "accommodation_type": slot["accommodation_type"],
            "start": slot["start"],
            "end": slot["end"],
            "room_count": slot.get("room_count"),
            "max_occupancy": slot.get("max_occupancy"),
            "occupancy_unknown": slot.get("occupancy_unknown"),
            "price_per_night": slot.get("price_per_night"),
            "why": why,
        }

    for slot in slots:
        key = _slot_key(slot)
        if key in found.why_by_slot:
            fits.append(_slot_row(slot, found.why_by_slot[key]))
        else:
            rejected.append(
                _slot_row(
                    slot,
                    found.reject_why_by_slot.get(key)
                    or [{"reason": "semantic_mismatch"}],
                )
            )

    # Claims rank the survivors and supply evidence; they never veto a fit.
    for fit in fits:
        hits = found.claims_by_site.get(str(fit["campsite_id"])) or []
        fit["score"] = _claim_score(hits)
        evidence = _claim_evidence(hits)
        if evidence:
            fit["review_claims"] = evidence
        rules = _rules_for_fit(found.rules_by_query, int(fit["campsite_id"]))
        if rules:
            fit["campsite_rules"] = rules
    # Stable, so equal scores keep the vacancy SQL's start_date / type ordering.
    fits.sort(key=lambda f: f["score"], reverse=True)

    payload["fits"] = fits
    payload["rejected"] = rejected[:REJECTED_SAMPLE_LIMIT]
    payload["rejected_count"] = len(rejected)
    return payload


_planner_fits_payload = planner_fits_payload
