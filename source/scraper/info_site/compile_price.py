"""Compile parks.org.il rate cards into one AST-checked quote() per site."""

from __future__ import annotations

import ast
import re
from typing import NamedTuple

from openai import OpenAI

from source.price_sandbox.ast_check import source_sha256
from source.price_sandbox.gold import GoldCase, gold_for_url, run_cases
from source.price_sandbox.params import strip_type_quotes
from source.scraper.amenity_enrichment.llm import (
    QWEN_INSTRUCT_MODEL,
    LlmUsage,
    make_nebius_openai_client,
)
from source.scraper.info_site.db import listing_match_is_confident, resolve_listing_ids
from source.scraper.info_site.match_listing import InfoWebsiteNameMatcher
from source.scraper.info_site.parse import GatheredRateRow
from source.scraper.rules_ingest.lodging import fetch_panel
from source.scraper.rules_ingest.sections import (
    VISITOR_INFO_TITLE,
    parse_visitor_info_panel,
)

# parks.org.il tableTab for the occupancy schedule. Not a GuestType identity.
GROUP_TAB = "קבוצה"

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

Output Python only, no markdown fences, no commentary. The module may
`import math`, `from enum import Enum` (or StrEnum), define Enum classes, and
must define exactly this function (same name, same parameters):

{QUOTE_SIGNATURE}

Rules:
- Return (price_float, explanation_str). Price is ILS for one night.
- Define `class Lodging(Enum)` and `class GuestType(Enum)` first. Member
  *values* are the exact Hebrew strings from the user message (quotation
  marks already stripped — never write `"`, `'`, or gershayim inside a
  member value). Member *names* are ASCII identifiers (TENT, HUSHA,
  REGULAR, MATMON, …).
- Parse once at the top of quote(), rebinding the parameters:
  `lodging = Lodging(lodging)` and `guest_type = GuestType(guest_type)`.
  An unknown string raises ValueError (do not catch it). After that, compare
  members (`lodging is Lodging.TENT`, `guest_type is GuestType.MATMON`).
  Never compare the raw Hebrew strings, never `in` a tuple of Hebrew, never
  startswith.
- Look rates up with nested dicts keyed by those enum members, then by
  these English field names only: `adult`, `child`, `weekday`, `weekend`,
  `late_exit`, `extra_adult`, `extra_child`. Interpret each rate-card
  `label` now, while writing the function — bake the number into the
  matching field. Do not store Hebrew labels, do not keep a list of
  `{{"label", "price"}}` rows, do not write `"מבוגר" in …` / `"ילד" in …`
  / `"תוספת" in …`, do not startswith/endswith. Hebrew lives only as
  enum member values and inside the explanation string.
- Example shape (numbers from the user message, not these):
  `RATES[Lodging.TENT][GuestType.REGULAR] = {{"adult": 76.0, "child": 58.0}}`
  `RATES[Lodging.HUSHA][GuestType.REGULAR] = {{"weekday": 350.0, "weekend": 450.0, "late_exit": 225.0}}`
  Then `rates = RATES[lodging][guest_type]`; tent uses `rates["adult"]`
  and `rates.get("child", rates["adult"])`; a unit uses
  `rates["weekend"] if is_weekend_or_holiday else rates["weekday"]`
  and adds `rates.get("late_exit")` when the late-exit condition holds.
- `guest_type` is the rate-card tab (רגיל, מנוי, חייל, …), not adult vs child.
  Adult vs child vs toddler is decided from `adults_num` / `child_num` /
  `child_ages` using those named fields.
- Child ages: under 5 are toddlers (free on tent/person rates unless a row says
  otherwise). Ages 5 up to 14 pay the child rate. 14 and up pay adult.
- `adults_num` is the adult headcount. `child_num` is the child headcount
  (toddlers included). Do not deduce `child_num` from `len(child_ages)`.
- Occupancy, extra-person, and group-threshold counts use
  `adults_num + child_num`. If an age is missing, treat that child as a
  paying child (5–14).
- GuestType is identity only (רגיל, מנוי, חייל, …). Never define
  GuestType.GROUP, never `GuestType("קבוצה")`. קבוצה is an occupancy
  override on a separate schedule in the user message, not a tab the
  caller can pass.
- Read that site's threshold from the occupancy-override notes (sites
  differ: 30 vs 80, …) and bake it as a number (`GROUP_MIN = 30`).
  Store those numbers as `GROUP_RATES[lodging] = {{"adult": …, "child": …}}`,
  not under `RATES[lodging][GuestType.GROUP]`.
- After parsing identity, if `adults_num + child_num >= GROUP_MIN` and
  that lodging has an occupancy schedule, use GROUP_RATES for that
  lodging and ignore identity — Matmon, soldier, every other GuestType
  included. Below the threshold, keep `RATES[lodging][guest_type]`.
- `is_weekend_or_holiday` selects אמצע שבוע vs סופי שבוע וחגים unit rates.
  Per-person tent rows with no weekday split apply every night.
- Per-unit lodging (bungalow, room, חושה) ignores party size except extra-person
  surcharges (תוספת מבוגר / תוספת ילד / included occupancy).
- `planned_exit_time` after 12:00 on a weekend: add the matching
  תוספת יציאה מאוחרת row when one exists for that product.
- Use only the rate rows and visitor-info pricing rules supplied. Do not invent
  tariffs. Ignore non-price visitor rules (dogs, music, glass).
- Only `import math` and `from enum import Enum` (or StrEnum). No other
  imports, no files, no network, no try/except. Build the explanation like:
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


_STRING_SCAN_METHODS = frozenset({"startswith", "endswith", "find"})


def _const_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def runtime_string_scan_hits(source: str) -> list[str]:
    """Hebrew labels were interpreted at compile time; quote() must not scan them."""
    tree = ast.parse(source)
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for op, right in zip(node.ops, node.comparators, strict=True):
                if not isinstance(op, (ast.In, ast.NotIn)):
                    continue
                sample = _const_str(node.left) or _const_str(right)
                if sample is None:
                    continue
                hits.append(
                    f"line {node.lineno}: string membership {sample!r}"
                )
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr not in _STRING_SCAN_METHODS:
                continue
            hits.append(
                f"line {node.lineno}: call to {node.func.attr}()"
            )
    return hits


def _constant_bool(node: ast.AST) -> bool | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return None


def _body_always_leaves(stmts: list[ast.stmt]) -> bool:
    return any(_always_leaves(stmt) for stmt in stmts)


def _always_leaves(stmt: ast.stmt) -> bool:
    if isinstance(stmt, (ast.Return, ast.Raise)):
        return True
    if not isinstance(stmt, ast.If):
        return False
    flag = _constant_bool(stmt.test)
    if flag is True:
        return _body_always_leaves(stmt.body)
    if flag is False:
        return _body_always_leaves(stmt.orelse)
    if not stmt.orelse:
        return False
    return _body_always_leaves(stmt.body) and _body_always_leaves(stmt.orelse)


def _scan_unreachable_body(stmts: list[ast.stmt], hits: list[str]) -> None:
    terminated = False
    for stmt in stmts:
        if terminated:
            hits.append(f"line {stmt.lineno}: unreachable")
            continue
        _scan_unreachable_stmt(stmt, hits)
        if _always_leaves(stmt):
            terminated = True


def _scan_unreachable_stmt(stmt: ast.stmt, hits: list[str]) -> None:
    if isinstance(stmt, ast.If):
        flag = _constant_bool(stmt.test)
        if flag is False:
            for nested in stmt.body:
                hits.append(f"line {nested.lineno}: unreachable")
            _scan_unreachable_body(stmt.orelse, hits)
            return
        if flag is True:
            _scan_unreachable_body(stmt.body, hits)
            for nested in stmt.orelse:
                hits.append(f"line {nested.lineno}: unreachable")
            return
        _scan_unreachable_body(stmt.body, hits)
        _scan_unreachable_body(stmt.orelse, hits)
        return
    if isinstance(stmt, ast.For):
        _scan_unreachable_body(stmt.body, hits)
        _scan_unreachable_body(stmt.orelse, hits)
        return
    if isinstance(stmt, ast.FunctionDef):
        _scan_unreachable_body(stmt.body, hits)


def unreachable_code_hits(source: str) -> list[str]:
    """Syntactic dead code: after return/raise, `if False`, both-branch return.

    No dataflow. `if lodging is TENT` twice is not flagged.
    """
    tree = ast.parse(source)
    hits: list[str] = []
    for stmt in tree.body:
        _scan_unreachable_stmt(stmt, hits)
    return hits


def guest_type_must_not_be_group_hits(source: str) -> list[str]:
    """GuestType is identity tabs; קבוצה is an occupancy override."""
    tree = ast.parse(source)
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "GuestType":
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                    continue
                target = stmt.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                value = _const_str(stmt.value)
                if target.id == "GROUP" or value == GROUP_TAB:
                    hits.append(
                        f"line {stmt.lineno}: GuestType must not include "
                        f"{target.id}={value!r}"
                    )
        elif (
            isinstance(node, ast.Attribute)
            and node.attr == "GROUP"
            and isinstance(node.value, ast.Name)
            and node.value.id == "GuestType"
        ):
            hits.append(f"line {node.lineno}: GuestType.GROUP is not allowed")
    return hits


def static_compile_hits(source: str) -> list[str]:
    """Allowlist-adjacent rejects that do not need to execute quote()."""
    return (
        runtime_string_scan_hits(source)
        + unreachable_code_hits(source)
        + guest_type_must_not_be_group_hits(source)
    )


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
    """Canonical lodging rows whose listing match is confident.

    A tab is a guest_type only if its labels match catalog lodging. Low
    confidence (rental extras, events) is dropped, not forced onto a product.
    """
    id_to_name = dict(names)
    cache: dict[str, tuple[list[int], float | None]] = {}
    matched: list[CompileRateRow] = []
    skipped_labels: set[str] = set()
    for row in gathered:
        if row.raw_label not in cache:
            cache[row.raw_label] = resolve_listing_ids(
                row.raw_label,
                names,
                full_label=row.raw_label,
                matcher=matcher,
                usage=usage,
                force=False,
            )
        ids, confidence = cache[row.raw_label]
        if not ids or not listing_match_is_confident(confidence):
            if row.raw_label not in skipped_labels:
                skipped_labels.add(row.raw_label)
                score = "none" if confidence is None else f"{confidence:.2f}"
                print(f"      skip non-lodging {score}: {row.raw_label!r}")
            continue
        for name_id in ids:
            lodging = strip_type_quotes(id_to_name.get(name_id) or "")
            if not lodging:
                continue
            matched.append(
                CompileRateRow(
                    lodging=lodging,
                    guest_type=strip_type_quotes(row.rate_class),
                    label=_canonical_label(row.raw_label, lodging),
                    price=row.price,
                    notes=row.notes,
                )
            )
    return matched


class CompilePromptBuckets(NamedTuple):
    identity_rows: list[CompileRateRow]
    group_rows: list[CompileRateRow]
    identity_guest_types: list[str]


def partition_compile_rows(
    rows: list[CompileRateRow],
    guest_types: list[str],
) -> CompilePromptBuckets:
    identity_rows = [row for row in rows if row.guest_type != GROUP_TAB]
    group_rows = [row for row in rows if row.guest_type == GROUP_TAB]
    identity_guest_types = [name for name in guest_types if name != GROUP_TAB]
    if not identity_guest_types:
        identity_guest_types = list(
            dict.fromkeys(row.guest_type for row in identity_rows)
        )
    return CompilePromptBuckets(
        identity_rows=identity_rows,
        group_rows=group_rows,
        identity_guest_types=identity_guest_types,
    )


def _rows_payload(rows: list[CompileRateRow], *, include_guest_type: bool) -> str:
    lines = []
    for row in rows:
        note = f" | notes: {row.notes}" if row.notes else ""
        guest = (
            f" guest_type={strip_type_quotes(row.guest_type)!r}"
            if include_guest_type
            else ""
        )
        lines.append(
            f"- lodging={strip_type_quotes(row.lodging)!r}{guest} "
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
    buckets = partition_compile_rows(rows, guest_types)
    lodging_lines = "\n".join(
        f"- {strip_type_quotes(name)}" for name in lodgings
    )
    guest_lines = "\n".join(
        f"- {strip_type_quotes(name)}" for name in buckets.identity_guest_types
    )
    identity_block = _rows_payload(
        buckets.identity_rows, include_guest_type=True
    )
    parts = [
        f"Campsite: {site_name}\n",
        f"class Lodging(Enum) — member value is exactly:\n{lodging_lines}\n",
        "class GuestType(Enum) — member value is exactly "
        f"(identity tabs only; never {GROUP_TAB}):\n{guest_lines}\n",
        "Rate-card rows (lodging / guest_type are those Hebrew values):\n"
        f"{identity_block}\n",
    ]
    if buckets.group_rows:
        group_block = _rows_payload(
            buckets.group_rows, include_guest_type=False
        )
        parts.append(
            "Occupancy override — not a GuestType. Read GROUP_MIN from the "
            "notes. When adults_num + child_num >= GROUP_MIN and that lodging "
            "appears here, use these rates and ignore identity (including "
            f"מנוי). Do not put {GROUP_TAB} on GuestType.\n"
            f"{group_block}\n"
        )
    parts.append(
        "Visitor-info panel (use pricing rules only):\n"
        f"{visitor_info.strip() or '(none)'}\n"
    )
    return "\n".join(parts)


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
    return run_cases(source, cases)


def gold_cases_for_site(*, url: str) -> list[GoldCase] | None:
    return gold_for_url(url)


def digest_source(source: str) -> str:
    return source_sha256(source)
