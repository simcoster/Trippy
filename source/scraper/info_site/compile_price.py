"""Compile parks.org.il rate cards into one AST-checked quote() per site."""

from __future__ import annotations

import ast
import re
from typing import NamedTuple

from openai import OpenAI

from source.price_sandbox.ast_check import compile_quote, source_sha256
from source.price_sandbox.execute import eval_quote_inprocess
from source.price_sandbox.gold import GoldCase, gold_for_url, prices_close, run_cases
from source.price_sandbox.params import strip_type_quotes
from source.scraper.amenity_enrichment.llm import (
    QWEN_INSTRUCT_MODEL,
    LlmUsage,
    make_nebius_openai_client,
    nebius_chat_create,
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
  member value). Member *names* are ASCII identifiers (TENT, CABIN,
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
- Example shape (invented numbers — copy rates from the user message only).
  Two kinds of schedule; never mix their keys on one lodging:
  per-person (לינת שטח, pitch, group) — `adult` / `child`:
  `RATES[Lodging.TENT][GuestType.REGULAR] = {{"adult": 10.0, "child": 8.0}}`
  per-unit (חושה, bungalow, room, family tent, mahal) — `weekday` /
  `weekend` / `late_exit`, then extras:
  `RATES[Lodging.CABIN][GuestType.REGULAR] = {{"weekday": 100.0, "weekend": 120.0, "late_exit": 50.0, "extra_adult": 15.0}}`
  After `rates = RATES[lodging][guest_type]`, a per-person lodging uses
  `rates["adult"]` and `rates.get("child", rates["adult"])`. A per-unit
  lodging has no `adult` key — use
  `rates["weekend"] if is_weekend_or_holiday else rates["weekday"]`
  and add `rates.get("late_exit")` when the late-exit condition holds.
  Do not read `rates["adult"]` on a unit dict.
- `guest_type` is the rate-card tab (רגיל, מנוי, חייל, …), not adult vs child.
  Adult vs child vs toddler is decided from `adults_num` / `child_num` /
  `child_ages` using those named fields. Soldier, Matmon, miluim, student,
  senior, and disabled change **per-person** rates only. Per-unit lodging
  (חושה, bungalow, family tent, …) ignores `guest_type` unless that unit
  has its own identity rows.
- Child ages: under 5 are toddlers (free on tent/person rates unless a row says
  otherwise). Ages 5 up to 14 pay the child rate. 14 and up pay adult.
- `adults_num` is the adult headcount. `child_num` is the child headcount
  (toddlers included). Do not deduce `child_num` from `len(child_ages)`.
- Occupancy, extra-person, and group-threshold counts use
  `adults_num + child_num`. If an age is missing, treat that child as
  age 10 (a paying child, 5–14).
- GuestType is identity only (רגיל, מנוי, חייל, …). Never define
  GuestType.GROUP, never `GuestType("קבוצה")`. קבוצה is an occupancy
  override on a separate schedule in the user message, not a tab the
  caller can pass.
- Read that site's threshold from the occupancy-override notes and bake
  it as a number (`GROUP_MIN`). "מעל X לנים" and "X ומעלה" mean X
  people and up: `GROUP_MIN` is X, not X+1. Sites differ; do not assume
  a threshold from this prompt. Store those numbers as
  `GROUP_RATES[lodging] = {{"adult": …, "child": …}}`,
  not under `RATES[lodging][GuestType.GROUP]`.
- After parsing identity, if `adults_num + child_num >= GROUP_MIN` and
  that lodging has an occupancy schedule, use GROUP_RATES for that
  lodging and ignore identity — Matmon, soldier, every other GuestType
  included. Below the threshold, keep `RATES[lodging][guest_type]`.
- `is_weekend_or_holiday` selects אמצע שבוע vs סופי שבוע וחגים unit rates.
  Per-person tent rows with no weekday split apply every night.
- Per-person lodging (לינת שטח, pitch) and GROUP_RATES are headcount ×
  `adult` / `child`. Per-unit lodging (חושה, bungalow, room, hut,
  family tent, mahal) is one price for the whole unit
  (`weekday` / `weekend`); party size does not multiply that price,
  only extra-person surcharges (תוספת מבוגר / תוספת ילד / תוספת אדם).
  `guest_type` does not apply unless that unit has identity rows.
  Included occupancy is the עד N on that unit's own price row (a family
  tent row "עד N לנים" → N people at the unit price). A תוספת אדם row is
  the N+1st guest, not folded into the unit. A cap in the notes
  ("עד M לנים") is a maximum, not included occupancy.
- Two published sizes of the same kind of lodging are different Lodging
  members when both appear in the Lodging list. Quote each member's unit
  price from that member's own rows. Do not price the larger size as
  extras on the smaller unit, and do not copy a number from this prompt.
- Count free under-5s in one name and reuse it (`toddler_count`).
- `planned_exit_time` after 12:00 on a weekend: add the matching
  תוספת יציאה מאוחרת row when one exists for that product.
- Use only the rate rows and visitor-info pricing rules supplied. Do not invent
  tariffs. Ignore non-price visitor rules (dogs, music, glass).
- Only `import math` and `from enum import Enum` (or StrEnum). No other
  imports, no files, no network. try/except is allowed; do not catch the
  enum constructor. Build the explanation like:
  "2 adults [10] + 2 children [8] [ages 5,7] + 1 toddler [free] (age 4); Matmon"
"""

FIX_SYSTEM_PROMPT = f"""You fix one Python quote() module for an Israeli campsite.

Output Python only, no markdown fences, no commentary. Keep the same
quote() signature, Lodging/GuestType enums, and rate numbers already in
the function. Only repair the listed errors (syntax, undefined names,
AST/static violations). Do not add tariffs. Do not guess prices.

The module may `import math`, `from enum import Enum` (or StrEnum), and
must define exactly this function:

{QUOTE_SIGNATURE}
"""

OCCUPANCY_RETRY_PREAMBLE = """
A previous attempt used the wrong included occupancy, or priced two
published sizes as extras on one unit. Included occupancy is the עד N
on that unit's own price row; a תוספת אדם / תוספת מבוגר / תוספת ילד
row is the N+1st guest; a cap in the notes is a maximum, not included
occupancy. "מעל X לנים" means GROUP_MIN is X, not X+1. Two published
sizes are different Lodging members when both appear in the Lodging
list. Reread the rate-card rows above. Do not invent numbers that are
not on those rows.
""".strip()


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


class GoldMiss(NamedTuple):
    note: str
    lodging: str
    kind: str
    message: str


class CompileVerdict(NamedTuple):
    ok: bool
    retry: str | None
    stage: str
    log_lines: list[str]
    retry_errors: list[str]


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
    if isinstance(stmt, ast.While):
        _scan_unreachable_body(stmt.body, hits)
        _scan_unreachable_body(stmt.orelse, hits)
        return
    if isinstance(stmt, ast.FunctionDef):
        _scan_unreachable_body(stmt.body, hits)
        return
    if isinstance(stmt, ast.Try):
        _scan_unreachable_body(stmt.body, hits)
        for handler in stmt.handlers:
            _scan_unreachable_body(handler.body, hits)
        _scan_unreachable_body(stmt.orelse, hits)
        _scan_unreachable_body(stmt.finalbody, hits)
        return


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
            "notes (מעל X לנים means X and up, not X+1). When "
            "adults_num + child_num >= GROUP_MIN and that lodging "
            "appears here, use these rates and ignore identity (including "
            f"מנוי). Do not put {GROUP_TAB} on GuestType.\n"
            f"{group_block}\n"
        )
    parts.append(
        "Visitor-info panel (use pricing rules only):\n"
        f"{visitor_info.strip() or '(none)'}\n"
    )
    return "\n".join(parts)


def _complete_python(
    *,
    system: str,
    user: str,
    role: str,
    client: OpenAI | None = None,
    usage: LlmUsage | None = None,
    model: str | None = None,
) -> CompileQuoteDraft:
    llm = client or make_nebius_openai_client()
    chosen = model or QWEN_INSTRUCT_MODEL
    response = nebius_chat_create(
        llm,
        usage=usage,
        role=role,
        model=chosen,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    source = extract_python_source(response.choices[0].message.content or "")
    return CompileQuoteDraft(source=source, user_prompt=user)


def occupancy_retry_suffix(misses: list[GoldMiss]) -> str:
    """Rate-card regenerate hint. Lodging + case note only; no prices."""
    lines = [OCCUPANCY_RETRY_PREAMBLE, "", "Missed cases:"]
    for miss in misses:
        lines.append(f"- {miss.note} (lodging={miss.lodging})")
    return "\n".join(lines)


def inspect_gold(source: str, cases: list[GoldCase]) -> list[GoldMiss]:
    misses: list[GoldMiss] = []
    for index, case in enumerate(cases, start=1):
        note = case.note or f"case {index}"
        lodging = case.params.lodging
        try:
            result = eval_quote_inprocess(source, case.params)
        except Exception as exc:
            misses.append(
                GoldMiss(note=note, lodging=lodging, kind="exception", message=str(exc))
            )
            continue
        if not prices_close(result.price, case.expected_price):
            misses.append(
                GoldMiss(note=note, lodging=lodging, kind="price", message="")
            )
    return misses


def assess_compiled_source(source: str, cases: list[GoldCase]) -> CompileVerdict:
    """AST, static checks, then gold. retry is fix, regen, or None if ok."""
    try:
        compile_quote(source)
    except Exception as exc:
        text = str(exc)
        return CompileVerdict(
            ok=False,
            retry="fix",
            stage="allowlist",
            log_lines=[text],
            retry_errors=[text],
        )
    scans = static_compile_hits(source)
    if scans:
        return CompileVerdict(
            ok=False,
            retry="fix",
            stage="static",
            log_lines=scans,
            retry_errors=scans,
        )
    misses = inspect_gold(source, cases)
    if not misses:
        return CompileVerdict(
            ok=True, retry=None, stage="ok", log_lines=[], retry_errors=[]
        )
    log_lines = run_cases(source, cases)
    if all(miss.kind == "price" for miss in misses):
        return CompileVerdict(
            ok=False,
            retry="regen",
            stage="gold",
            log_lines=log_lines,
            retry_errors=[occupancy_retry_suffix(misses)],
        )
    retry_errors = [
        f"{miss.note}: {miss.message}" if miss.message else miss.note
        for miss in misses
        if miss.kind == "exception"
    ]
    return CompileVerdict(
        ok=False,
        retry="fix",
        stage="gold",
        log_lines=log_lines,
        retry_errors=retry_errors or [miss.note for miss in misses],
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
    role: str = "price_function_compile",
    retry_suffix: str = "",
) -> CompileQuoteDraft:
    """One 235B call. Returns the model's Python; the caller AST-checks it."""
    user = compile_user_prompt(
        site_name=site_name,
        lodgings=lodgings,
        guest_types=guest_types,
        rows=rows,
        visitor_info=visitor_info,
    )
    if retry_suffix.strip():
        user = user + "\n" + retry_suffix.strip() + "\n"
    return _complete_python(
        system=SYSTEM_PROMPT,
        user=user,
        role=role,
        client=client,
        usage=usage,
        model=model,
    )


def compile_quote_fix(
    source: str,
    errors: list[str],
    *,
    client: OpenAI | None = None,
    usage: LlmUsage | None = None,
    model: str | None = None,
) -> CompileQuoteDraft:
    """Fix-turn: the failed function plus error text, no gold prices."""
    listed = "\n".join(f"- {line}" for line in errors if line)
    user = f"Fix this function. Errors were:\n{listed}\n\n{source.strip()}\n"
    return _complete_python(
        system=FIX_SYSTEM_PROMPT,
        user=user,
        role="price_function_compile_fix",
        client=client,
        usage=usage,
        model=model,
    )


def run_gold_tests(source: str, cases: list[GoldCase]) -> list[str]:
    """Return human-readable failures; empty means all cases matched."""
    return run_cases(source, cases)


def gold_cases_for_site(*, url: str) -> list[GoldCase] | None:
    return gold_for_url(url)


def digest_source(source: str) -> str:
    return source_sha256(source)
