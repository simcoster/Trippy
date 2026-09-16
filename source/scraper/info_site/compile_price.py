"""Compile parks.org.il rate cards into one AST-checked quote() per site."""

from __future__ import annotations

import re
from typing import NamedTuple

from openai import OpenAI

from source.price_sandbox.ast_check import source_sha256
from source.price_sandbox.execute import eval_quote_inprocess
from source.price_sandbox.gold import GoldCase, gold_for_url, prices_close
from source.scraper.amenity_enrichment.llm import (
    QWEN_INSTRUCT_MODEL,
    LlmUsage,
    make_nebius_openai_client,
)
from source.scraper.info_site.db import resolve_listing_ids
from source.scraper.info_site.match_listing import InfoWebsiteNameMatcher
from source.scraper.info_site.parse import GatheredRateRow
from source.scraper.rules_ingest.lodging import fetch_panel
from source.scraper.rules_ingest.sections import (
    VISITOR_INFO_TITLE,
    parse_visitor_info_panel,
)

QUOTE_SIGNATURE = """
def quote(
    lodging: str,
    adults_num: int,
    child_num: int = 0,
    child_ages: tuple = (),
    guest_type: str = "רגיל",
    is_weekend_or_holiday: bool = False,
    planned_entry_time: str | None = None,
    planned_exit_time: str | None = None,
) -> tuple:
    ...
    return price, explanation
""".strip()

SYSTEM_PROMPT = f"""You write one Python function that quotes one night at one Israeli campsite.

Output Python only, no markdown fences, no commentary. The module may `import math`
and must define exactly this function (same name, same parameters):

{QUOTE_SIGNATURE}

Rules:
- Return (price_float, explanation_str). Price is ILS for one night.
- Copy the supplied LODGINGS and GUEST_TYPES tuples into the module as constants.
  If `lodging` is not in LODGINGS, raise ValueError. If `guest_type` is not in
  GUEST_TYPES, raise ValueError. Compare with `==` / `in` against those exact
  strings. Never invent English names (no "tent"). Never use startswith to
  guess a product.
- `lodging` is already the canonical catalog name (info-site lodging panel).
  Rate-card rows in the user message use that same name.
- `guest_type` is the rate-card tab (רגיל, מנוי, חייל, …), not adult vs child.
  Adult vs child vs toddler is decided from `adults_num` / `child_num` /
  `child_ages` against the rows for that lodging + guest_type.
- Child ages: under 5 are toddlers (free on tent/person rates unless a row says
  otherwise). Ages 5 up to 14 pay the child rate. 14 and up pay adult.
- `adults_num` is the adult headcount. `child_num` is the child headcount
  (toddlers included). Do not deduce `child_num` from `len(child_ages)`.
- Occupancy, extra-person, and group-threshold counts use
  `adults_num + child_num`. If an age is missing, treat that child as a
  paying child (5–14).
- Group (קבוצה) is not a caller flag. If the card has a קבוצה guest_type and
  party size meets that site's threshold, use those rows instead of the
  caller's guest_type. Sites differ (30 vs 80, …). Below the threshold, keep
  the caller's guest_type.
- `is_weekend_or_holiday` selects אמצע שבוע vs סופי שבוע וחגים unit rates.
  Per-person tent rows with no weekday split apply every night.
- Per-unit lodging (bungalow, room, חושה) ignores party size except extra-person
  surcharges (תוספת מבוגר / תוספת ילד / included occupancy).
- `planned_exit_time` after 12:00 on a weekend: add the matching
  תוספת יציאה מאוחרת row when one exists for that product.
- Look rates up with nested dicts and subscript (`RATES[lodging][guest_type]`),
  not `next(...)`. Use only the rate rows and visitor-info pricing rules
  supplied. Do not invent tariffs. Ignore non-price visitor rules (dogs,
  music, glass).
- Only `import math` (or `from math import …`). No other imports, no files, no
  network. Build the explanation like:
  "2 adults [76] + 2 children [58] [ages 5,7] + 1 toddler [free] (age 4); Matmon"
"""

_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.S | re.I)


class CompileRateRow(NamedTuple):
    lodging: str
    guest_type: str
    label: str
    price: float
    notes: str | None


class CompileQuoteDraft(NamedTuple):
    source: str
    user_prompt: str


def extract_python_source(text: str) -> str:
    blob = (text or "").strip()
    fenced = _FENCE_RE.search(blob)
    if fenced:
        return fenced.group(1).strip()
    return blob


def gather_visitor_info_text(site_url: str, html: str) -> str:
    """AJAX `מידע למבקר` body, or empty when the tab is missing."""
    try:
        panel = fetch_panel(site_url, html, title=VISITOR_INFO_TITLE)
    except Exception:
        return ""
    if not panel:
        return ""
    sections = parse_visitor_info_panel(panel, source_url=site_url)
    return "\n\n".join(section.text for section in sections if section.text)


def _canonical_label(raw_label: str, lodging: str) -> str:
    if lodging and lodging in raw_label:
        return raw_label
    if lodging:
        return f"{lodging} — {raw_label}"
    return raw_label


def match_compile_rows(
    gathered: list[GatheredRateRow],
    names: list[tuple[int, str]],
    *,
    matcher: InfoWebsiteNameMatcher | None = None,
    usage: LlmUsage | None = None,
) -> list[CompileRateRow]:
    """Resolve each published label to canonical catalog names (same matcher as list_prices)."""
    id_to_name = dict(names)
    cache: dict[str, list[int]] = {}
    matched: list[CompileRateRow] = []
    for row in gathered:
        if row.raw_label not in cache:
            ids, _confidence = resolve_listing_ids(
                row.raw_label,
                names,
                full_label=row.raw_label,
                matcher=matcher,
                usage=usage,
            )
            cache[row.raw_label] = ids
        for name_id in cache[row.raw_label]:
            lodging = id_to_name.get(name_id) or ""
            if not lodging:
                continue
            matched.append(
                CompileRateRow(
                    lodging=lodging,
                    guest_type=row.rate_class,
                    label=_canonical_label(row.raw_label, lodging),
                    price=row.price,
                    notes=row.notes,
                )
            )
    return matched


def _py_tuple(values: list[str]) -> str:
    if not values:
        return "()"
    inner = ",\n    ".join(repr(value) for value in values)
    return "(\n    " + inner + ",\n)"


def _rows_payload(rows: list[CompileRateRow]) -> str:
    lines = []
    for row in rows:
        note = f" | notes: {row.notes}" if row.notes else ""
        lines.append(
            f"- lodging={row.lodging!r} guest_type={row.guest_type!r} "
            f"label={row.label!r} price={row.price}{note}"
        )
    return "\n".join(lines)


def compile_user_prompt(
    *,
    site_name: str,
    lodgings: list[str],
    guest_types: list[str],
    rows: list[CompileRateRow],
    visitor_info: str,
) -> str:
    return (
        f"Campsite: {site_name}\n\n"
        f"LODGINGS = {_py_tuple(lodgings)}\n\n"
        f"GUEST_TYPES = {_py_tuple(guest_types)}\n\n"
        f"Rate-card rows (lodging is the canonical catalog name):\n"
        f"{_rows_payload(rows)}\n\n"
        f"Visitor-info panel (use pricing rules only):\n"
        f"{visitor_info.strip() or '(none)'}\n"
    )


def compile_quote_source(
    *,
    rows: list[CompileRateRow],
    lodgings: list[str],
    guest_types: list[str],
    visitor_info: str,
    site_name: str,
    client: OpenAI | None = None,
    usage: LlmUsage | None = None,
    model: str | None = None,
) -> CompileQuoteDraft:
    """One 235B call. Returns the model's Python; the caller AST-checks it."""
    llm = client or make_nebius_openai_client()
    chosen = model or QWEN_INSTRUCT_MODEL
    user = compile_user_prompt(
        site_name=site_name,
        lodgings=lodgings,
        guest_types=guest_types,
        rows=rows,
        visitor_info=visitor_info,
    )
    response = llm.chat.completions.create(
        model=chosen,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    if usage is not None:
        usage.add_chat(
            response.usage, role="price_function_compile", model=chosen
        )
    source = extract_python_source(response.choices[0].message.content or "")
    return CompileQuoteDraft(source=source, user_prompt=user)


def run_gold_tests(source: str, cases: list[GoldCase]) -> list[str]:
    """Return human-readable failures; empty means all cases matched."""
    failures: list[str] = []
    for index, case in enumerate(cases, start=1):
        label = case.note or f"case {index}"
        try:
            result = eval_quote_inprocess(source, case.params)
        except Exception as exc:
            failures.append(f"{label}: {exc}")
            continue
        if not prices_close(result.price, case.expected_price):
            detail = case.explanation or ""
            if detail:
                detail = f" ({detail})"
            failures.append(
                f"{label}: got {result.price} expected {case.expected_price}{detail}"
            )
    return failures


def gold_cases_for_site(*, url: str) -> list[GoldCase] | None:
    return gold_for_url(url)


def digest_source(source: str) -> str:
    return source_sha256(source)
