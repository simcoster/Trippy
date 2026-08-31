"""Two-stage planner: vacancies then amenity intersection. Not a LangGraph node."""

from __future__ import annotations

from typing import Any

from source.agent import search
from source.agent.constraints import party_size_from_numeric, semantic_search_queries
from source.agent.dates import DATE_TRUNCATED_NOTICE, MAX_DATE_WINDOWS

AMENITY_MATCH_MAX_DISTANCE = -0.8
REJECTED_SAMPLE_LIMIT = 5


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


def _semantic_why_by_type(
    type_ids: list[int],
    semantic_constraints: list,
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
    """AND-groups of amenity queries → (matched why, rejected why) by type id."""
    unique_ids = list(dict.fromkeys(int(x) for x in type_ids))
    if not unique_ids:
        return {}, {}
    groups = semantic_search_queries(semantic_constraints)
    if not groups:
        return {tid: [] for tid in unique_ids}, {}

    why_by_type: dict[int, list[dict[str, Any]]] = {tid: [] for tid in unique_ids}
    missing_by_type: dict[int, list[dict[str, Any]]] = {tid: [] for tid in unique_ids}
    matching = set(unique_ids)
    for queries in groups:
        group_hits: dict[int, dict[str, Any]] = {}
        for query in queries:
            vec = search._query_vec_literal(query)
            hits = search.search_stated_amenities(
                query,
                limit=max(len(unique_ids), 1),
                embedding=vec,
                accommodation_type_ids=unique_ids,
            )
            for hit in hits:
                if not _amenity_hit_matches(hit):
                    continue
                tid = int(hit["accommodation_type_id"])
                if tid not in group_hits:
                    group_hits[tid] = {
                        "query": query,
                        "stated_amenity": hit.get("amenity"),
                        "distance": hit.get("distance"),
                    }
        matching &= set(group_hits)
        label = queries[0] if len(queries) == 1 else list(queries)
        for tid in unique_ids:
            if tid in group_hits:
                why_by_type[tid].append(group_hits[tid])
            else:
                missing_by_type[tid].append(
                    {
                        "reason": "missing_stated_amenity",
                        "query": label,
                    }
                )
    return (
        {tid: why_by_type[tid] for tid in matching},
        {tid: missing_by_type[tid] for tid in unique_ids if tid not in matching},
    )


def _review_claims_for_sites(
    site_ids: set[str],
    semantic_constraints: list,
    *,
    per_query_limit: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    if not site_ids:
        return {}
    by_site: dict[str, list[dict[str, Any]]] = {sid: [] for sid in site_ids}
    seen: set[tuple] = set()
    for queries in semantic_search_queries(semantic_constraints):
        for query in queries:
            vec = search._query_vec_literal(query)
            for hit in search.search_review_claims(
                query, limit=per_query_limit, embedding=vec
            ):
                if hit.get("error"):
                    continue
                cid = str(hit.get("campsite_id") or "")
                if cid not in by_site:
                    continue
                rec = {
                    "query": query,
                    "claim": hit.get("claim"),
                    "date": hit.get("date"),
                    "days_ago": hit.get("days_ago"),
                }
                key = (cid, rec["claim"], rec["date"])
                if key in seen:
                    continue
                seen.add(key)
                by_site[cid].append(rec)
    return by_site


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

    type_ids = [int(s["accommodation_type_id"]) for s in slots]
    why_by_type, reject_why_by_type = _semantic_why_by_type(type_ids, semantic)
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
        tid = int(slot["accommodation_type_id"])
        if tid in why_by_type:
            fits.append(_slot_row(slot, why_by_type[tid]))
        else:
            rejected.append(
                _slot_row(
                    slot,
                    reject_why_by_type.get(tid)
                    or [{"reason": "semantic_mismatch"}],
                )
            )

    if fits and semantic:
        site_keys = {str(f["campsite_id"]) for f in fits}
        claims = _review_claims_for_sites(site_keys, semantic)
        for fit in fits:
            extra = claims.get(str(fit["campsite_id"])) or []
            if extra:
                fit["review_claims"] = extra

    payload["fits"] = fits
    payload["rejected"] = rejected[:REJECTED_SAMPLE_LIMIT]
    payload["rejected_count"] = len(rejected)
    return payload


_planner_fits_payload = planner_fits_payload
