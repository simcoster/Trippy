"""Score one planner-eval case against gold `expect`. No LLM."""

from __future__ import annotations

from typing import Any


def _fit_sites(planner: dict[str, Any] | None) -> set[int]:
    sites: set[int] = set()
    for row in (planner or {}).get("fits") or []:
        cid = row.get("campsite_id")
        if cid is not None:
            sites.add(int(cid))
    return sites


def _fit_types(planner: dict[str, Any] | None, *, site: int | None = None) -> list[str]:
    names: list[str] = []
    for row in (planner or {}).get("fits") or []:
        if site is not None and int(row.get("campsite_id") or 0) != site:
            continue
        names.append(str(row.get("accommodation_type") or ""))
    return names


def _extract_date(extract: dict[str, Any] | None) -> dict[str, str] | None:
    if not extract:
        return None
    date = extract.get("date")
    if isinstance(date, dict) and date.get("start"):
        return {
            "start": str(date.get("start")),
            "end": str(date.get("end") or ""),
        }
    windows = extract.get("date_windows") or []
    if windows and isinstance(windows[0], dict) and windows[0].get("start"):
        return {
            "start": str(windows[0].get("start")),
            "end": str(windows[0].get("end") or ""),
        }
    return None


def _party_size(extract: dict[str, Any] | None) -> int | None:
    for item in (extract or {}).get("numeric_constraints") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("field") or "") != "party_size":
            continue
        try:
            return int(item.get("value"))
        except (TypeError, ValueError):
            return None
    return None


def _loci(extract: dict[str, Any] | None) -> set[str]:
    found: set[str] = set()
    for item in (extract or {}).get("semantic_constraints") or []:
        if not isinstance(item, dict):
            continue
        locus = item.get("locus")
        if isinstance(locus, str) and locus.strip():
            found.add(locus.strip())
    return found


def _has_or_group(extract: dict[str, Any] | None, values: list[str]) -> bool:
    needles = {v.casefold() for v in values}
    for item in (extract or {}).get("semantic_constraints") or []:
        if not isinstance(item, dict):
            continue
        raw = item.get("values") or item.get("query")
        if item.get("op") == "or" and isinstance(raw, list):
            got = {str(v).casefold() for v in raw}
            if needles <= got or got <= needles:
                return True
            if needles & got:
                return True
    return False


def _why_has_fridge(planner: dict[str, Any] | None) -> bool:
    for row in (planner or {}).get("fits") or []:
        for entry in row.get("why") or []:
            if not isinstance(entry, dict):
                continue
            amenity = str(entry.get("site_amenity") or entry.get("stated_amenity") or "")
            if "refrigerat" in amenity:
                return True
    return False


def _price_for_site(planner: dict[str, Any] | None, site: int) -> list[float]:
    prices: list[float] = []
    for row in (planner or {}).get("fits") or []:
        if int(row.get("campsite_id") or 0) != site:
            continue
        raw = row.get("price_per_night")
        if raw is None:
            continue
        try:
            prices.append(float(raw))
        except (TypeError, ValueError):
            continue
    return prices


def score_case(
    expect: dict[str, Any],
    extract: dict[str, Any] | None,
    planner: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return {ok, failures, extract_date, fit_sites}."""
    failures: list[str] = []
    planner = planner or {}
    extract = extract or {}
    fit_sites = _fit_sites(planner)
    extract_date = _extract_date(extract)

    skipped = planner.get("skipped")
    want_skip = expect.get("current_planner")

    want_party = expect.get("party_size")
    if want_party is not None:
        got_party = _party_size(extract)
        if got_party != int(want_party):
            failures.append(f"party_size {got_party} != {want_party}")

    if want_skip == "skipped_no_date":
        if skipped != "no_date":
            failures.append(f"expected skipped=no_date, got {skipped!r}")
        return {
            "ok": not failures,
            "failures": failures,
            "extract_date": extract_date,
            "fit_sites": sorted(fit_sites),
        }

    want_date = expect.get("date")
    if isinstance(want_date, dict) and want_date.get("start"):
        if extract_date is None:
            failures.append("extract has no date")
        elif extract_date["start"] != str(want_date["start"]) or extract_date[
            "end"
        ] != str(want_date.get("end") or ""):
            failures.append(
                f"date {extract_date} != {want_date['start']}…{want_date.get('end')}"
            )

    want_locus = expect.get("locus")
    if want_locus:
        if want_locus not in _loci(extract):
            failures.append(f"locus {want_locus!r} missing in extract")

    or_values = expect.get("or")
    if isinstance(or_values, list) and or_values:
        if not _has_or_group(extract, [str(v) for v in or_values]):
            failures.append(f"extract missing OR group {or_values}")

    for site in expect.get("must_include_sites") or []:
        if int(site) not in fit_sites:
            failures.append(f"missing site {site}")

    for site in expect.get("must_exclude_sites") or []:
        if int(site) in fit_sites:
            failures.append(f"unexpected site {site}")

    types = _fit_types(planner)
    for needle in expect.get("must_include_types") or []:
        if not any(needle in name for name in types):
            failures.append(f"missing type {needle!r}")

    for needle in expect.get("must_exclude_types") or []:
        if any(needle in name for name in types):
            failures.append(f"unexpected type {needle!r}")

    if expect.get("why_fridge"):
        if not _why_has_fridge(planner):
            failures.append("fits why has no refrigerator listing")

    quotes = expect.get("quoted_price_per_night") or {}
    if isinstance(quotes, dict):
        for site_s, want in quotes.items():
            site = int(site_s)
            prices = _price_for_site(planner, site)
            if not any(abs(p - float(want)) < 0.51 for p in prices):
                failures.append(
                    f"site {site} price {prices} does not include {want}"
                )

    cap = expect.get("max_price_per_night")
    if cap is not None:
        for row in planner.get("fits") or []:
            raw = row.get("price_per_night")
            if raw is None:
                continue
            if float(raw) > float(cap) + 0.51:
                failures.append(
                    f"site {row.get('campsite_id')} price {raw} > {cap}"
                )

    return {
        "ok": not failures,
        "failures": failures,
        "extract_date": extract_date,
        "fit_sites": sorted(fit_sites),
    }
