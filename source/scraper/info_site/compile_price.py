"""Compile parks.org.il rate cards into one AST-checked quote() per site."""

from __future__ import annotations

import re

from openai import OpenAI

from source.price_sandbox.ast_check import compile_quote, source_sha256
from source.price_sandbox.execute import eval_quote_inprocess
from source.price_sandbox.gold import GoldCase, gold_for_url, prices_close
from source.scraper.amenity_enrichment.llm import (
    QWEN_INSTRUCT_MODEL,
    LlmUsage,
    make_nebius_openai_client,
)
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
    child_ages: tuple = (),
    is_matmon_sub: bool = False,
    is_soldier: bool = False,
    is_active_reserve: bool = False,
    is_senior: bool = False,
    is_student: bool = False,
    is_disabled_idf: bool = False,
    is_group: bool = False,
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
- `lodging` is the product name (tent, bungalow, חושה, caravan pitch, staff room).
  Match it to the rate-card rows. If nothing matches, raise ValueError.
- Child ages: under 5 are toddlers (free on tent/person rates unless a row says
  otherwise). Ages 5 up to 14 pay the child rate. 14 and up pay adult.
- `adults_num` is the adult headcount and is separate from `child_ages`.
- Discount flags select the matching rate-card tab (מנוי, חייל, מילואים, קבוצה,
  אזרח ותיק, סטודנט, נכה צה"ל). Default tab is רגיל. If two flags are true,
  prefer the more specific discount that exists on the card.
- `is_weekend_or_holiday` selects אמצע שבוע vs סופי שבוע וחגים unit rates.
  Per-person tent rows with no weekday split apply every night.
- Per-unit lodging (bungalow, room, חושה) ignores party size except extra-person
  surcharges (תוספת מבוגר / תוספת ילד / included occupancy).
- `planned_exit_time` after 12:00 on a weekend: add the matching
  תוספת יציאה מאוחרת row when one exists for that product.
- Use only the rate rows and visitor-info pricing rules supplied. Do not invent
  tariffs. Ignore non-price visitor rules (dogs, music, glass).
- Only `import math` (or `from math import …`). No other imports, no files, no
  network. Build the explanation like:
  "2 adults [76] + 2 children [58] [ages 5,7] + 1 toddler [free] (age 4); Matmon"
"""

_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.S | re.I)


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


def _rows_payload(rows: list[GatheredRateRow]) -> str:
    lines = []
    for row in rows:
        note = f" | notes: {row.notes}" if row.notes else ""
        lines.append(
            f"- tab={row.rate_class!r} label={row.raw_label!r} "
            f"price={row.price}{note}"
        )
    return "\n".join(lines)


def compile_quote_source(
    *,
    rows: list[GatheredRateRow],
    visitor_info: str,
    site_name: str,
    client: OpenAI | None = None,
    usage: LlmUsage | None = None,
    model: str | None = None,
) -> str:
    """One 235B call. Raises PriceFunctionError if the reply is not valid quote()."""
    llm = client or make_nebius_openai_client()
    chosen = model or QWEN_INSTRUCT_MODEL
    user = (
        f"Campsite: {site_name}\n\nRate-card rows:\n{_rows_payload(rows)}\n\n"
        f"Visitor-info panel (use pricing rules only):\n"
        f"{visitor_info.strip() or '(none)'}\n"
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
    compile_quote(source)
    return source


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
            failures.append(
                f"{label}: got {result.price} expected {case.expected_price}"
            )
    return failures


def gold_cases_for_site(*, url: str) -> list[GoldCase] | None:
    return gold_for_url(url)


def digest_source(source: str) -> str:
    return source_sha256(source)
