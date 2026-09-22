"""Pick 2–3 planner fits and write a cited Hebrew recommendation."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, NamedTuple

from langchain_core.messages import (
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)

from source.agent.constraints import latest_constraints_json
from source.agent.messages import latest_user_text, message_text
from source.agent.prompts import EMPTY_REPLY_FALLBACK
from source.agent.recommender.fallback import (
    kimi_super_fallback,
    primary_recommend_call,
    recommend_with_fallback,
)
from source.agent.recommender.json_text import (
    drop_invalid_json_escapes,
    json_object_prefix,
    parse_stream_json,
)
from source.agent.recommender.models import prepare_model, recommender_model
from source.agent.recommender.stay_dates import (
    StayWindow,
    booking_lines,
    stay_date_label,
    windows_from_dates,
)
from source.agent.recommender.stream import chunk_text, chunk_thinking, iter_chat_chunks
from source.agent.recommender.timing import RecommendClock, record_recommend
from source.agent.recommender.why_not import query_is_hebrew, render_why_not
from source.agent.timing import stage
from source.agent.turn_status import RANKING, report_turn_status
from source.scraper.amenity_enrichment.llm import (
    _parse_json_payload,
    langchain_chat_usage,
)

logger = logging.getLogger(__name__)

RECOMMENDER_SYSTEM_PROMPT = """
You pick campsite stays for Trippy. Output JSON only.

The user JSON has query (original ask), extract (structured constraints),
and fits (available stays that already match dates, party size, and price
when those were given). Recommend only from fits. Never invent a campsite,
type, date, price, amenity, or rule. When a fit includes price_explanation,
cite that breakdown; do not recompute the sum.

Pick the top 2 or 3 stays (fits is best-first). If only one fit
exists, pick that one. Never more than 3. Prefer different campsites
over two types at the same site. When you pick more than one, set
intro to a short spoken note: there is more than one option, what
they are, and how they differ (tent vs staff room, north vs south,
cheaper vs closer). Phrase it however sounds natural; do not use a
fixed template. Each why is only about that stay; do not repeat the
intro there. intro is null when you pick 1. A fit's dates list is
every night that unit is free; the reply lists those nights, so do
not recap them in why.
Keep planner order as a hint (fits is best-first) but you may skip a
worse later row. Copy campsite_id, accommodation_type, start, end, and
booking_url exactly from the fit you pick. Do not invent or rewrite a
booking_url.

Write why in this order. Lead with the matching facts — not a restatement
of the query. Then, only about those same things, add listing-vs-review
notes and quality caveats the user did not ask for (dirty vs clean,
working vs broken). Do not mention an amenity or complaint the user did
not ask about and that is not about the ask — no shade note on a
hot-shower query, and never "no info on showers" when they did not ask.

Do not quote or recap the query ("the request was for X", "הבקשה הייתה").
Do not recap dates, party size, or price; the stay line already has them.
Do not write form-language: "the option offers", "האפשרות מציעה",
"הליסטינג", "fits the request at the specified dates". Sound like a
person who looked this up, not a translated checklist.

Paraphrase English claims into ordinary words in the user's language.
Never paste claim text. Never coin a word or calque English
(מרווחים not מרחביים; מדורות/מנגל not שמדליות). If you are unsure of a
word, omit it.

Cite listing vs reviews (use date / days_ago when you name a review):
- Listing and reviews agree the thing exists: say it once. "There are
  hot showers." Do not also write "and guests say there are hot
  showers" / "יש אוהלים וגם אורחים מספרים שיש אוהלים". Reviews add
  quality, condition, or a contradiction — not a second copy of the
  same yes.
- Listing yes, reviews no: name the contradiction. "The site says hot
  showers exist but reviews from 3 months ago say the showers only
  have cold water."
- Concrete amenity the user asked for (disabled parking, hot showers,
  fridge) is missing from the listing but a review mentions it:
  "Although the site doesn't specify, a review from 2 months ago
  does."
- Vibe / atmosphere (quiet, desert feel): reviews alone are enough.
  "Reviews say it is quiet." Do not add "the site doesn't specify."
- Related quality after the match: "People also say they are very
  clean." Or a caveat: "Some recent reviews say they are not working.
  Also, many reviews say the showers are dirty."

Evidence, in this order of trust:
- why: how each request was met. stated_amenity is the unit listing,
  site_amenity is the campsite listing, locus room means the guest wanted
  it inside the unit. A why entry with claim is a guest review only —
  never present it as listed.
- review_claims: guest claims a judge kept as about the request,
  positive and negative (is_positive), with date and days_ago. Use them
  for match, contradiction, review-only facts, and related caveats —
  not a dump of every claim. One complaint does not disqualify a site
  that lists the amenity. Weigh recent reviews more. claim text is
  English (text_en) — paraphrase it in the user's language; never paste
  it.
- rules: official listing rows retrieved for the request. MOST are
  unrelated nearest neighbors. Cite a rule only when it is actually about
  the ask, including polarity-false forbids (dogs_allowed false for pet
  friendly). A tent/cabin/room/hut subject is lodging, not a location or
  a vibe. electric_stove / kettle is cooking, not campsite electricity.
  A caravan-bay hookup does not serve a guest without a caravan.
  subject is an internal key (tent_pitch, dogs_allowed). Use
  evidence_span when you name the listing, never the key.
- claim_judge: planner verdict (satisfies, satisfy_by, reason). Do not
  quote it; it is a hint, not the user-facing why.

If extract.date_notice or date_truncated is set, say that only the first
4 date ranges were used. If fits is empty, recommendations is [] and
empty is a short follow-up (dates, area, budget, amenities). empty is
null when you recommend.

Language: the user JSON field reply_language is already decided. Obey it.
Packed JSON is English (field names, snake_case subjects, review claims).
Copy campsite and accommodation_type names from the fit as stored, even
when that name is Hebrew and reply_language is english.

If reply_language is hebrew, why, intro, and empty are Hebrew only — no
Latin, CJK, or mixed-script tokens. For reviews write אורחים מספרים,
never Guests / guests / ゲuests / ospites. Do not write pitch,
tent_pitch, outlets, bungalow, camping, accommodation, Stay, stations,
dank, or glue Latin inside a Hebrew word (not בungalו, איןoutlets,
יש.pitch). None of the packed English belongs in why, intro, or empty.

If reply_language is english, why, intro, and empty are English
sentences only. Do not write the prose in Hebrew because the campsite
name or the listing is Hebrew. Write "guests report" for reviews.

Output JSON only:
{"recommendations": [{"campsite_id": int, "accommodation_type": str,
 "start": str, "end": str, "booking_url": str, "why": str}],
 "intro": str | null, "empty": str | null}
""".strip()

_recommend_text_sink: ContextVar[Callable[[str], None] | None] = ContextVar(
    "trippy_recommend_text", default=None
)


class _StayKey(NamedTuple):
    campsite_id: int
    accommodation_type: str
    start: str
    end: str


@dataclass(frozen=True)
class Recommendation:
    campsite_id: int
    campsite: str
    accommodation_type: str
    start: str
    end: str
    price_per_night: Any
    why: str
    booking_url: str = ""
    price_explanation: str = ""
    dates: tuple[StayWindow, ...] = ()


@dataclass(frozen=True)
class RecommendResult:
    recommendations: tuple[Recommendation, ...]
    empty: str | None
    text: str
    intro: str | None = None
    ttft_chunk_ms: float | None = None
    ttft_spoken_ms: float | None = None
    elapsed_ms: float | None = None
    reasoning_tokens: int | None = None
    thinking_stream: bool = False


@contextmanager
def listen_recommend_text(on_text: Callable[[str], None]) -> Iterator[None]:
    """Paint the spoken reply as recommend tokens arrive (Streamlit)."""
    token = _recommend_text_sink.set(on_text)
    try:
        yield
    finally:
        _recommend_text_sink.reset(token)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def stay_key(row: dict[str, Any]) -> _StayKey | None:
    cid = _as_int(row.get("campsite_id"))
    typ = _as_text(row.get("accommodation_type"))
    start = _as_text(row.get("start"))
    end = _as_text(row.get("end"))
    if cid is None or not typ or not start or not end:
        return None
    return _StayKey(cid, typ, start, end)


def _compact_mapping(
    row: dict[str, Any], keys: tuple[str, ...]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        if row.get(key) is not None:
            out[key] = row[key]
    return out


_WHY_KEYS = (
    "query",
    "stated_amenity",
    "site_amenity",
    "claim",
    "locus",
    "reason",
    "detail",
    "is_positive",
    "date",
    "days_ago",
)
_CLAIM_KEYS = ("query", "claim", "date", "days_ago", "is_positive")
_RULE_KEYS = ("subject", "polarity", "evidence_span", "qualifier")
_JUDGE_KEYS = (
    "query",
    "satisfies",
    "satisfy_by",
    "relevant_claims",
    "reason",
)
_FIT_KEYS = (
    "campsite_id",
    "campsite",
    "accommodation_type",
    "start",
    "end",
    "room_count",
    "max_occupancy",
    "occupancy_unknown",
    "price_per_night",
    "price_explanation",
    "booking_url",
    "dates",
)


def _compact_rules(rows: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("error"):
            continue
        compact = _compact_mapping(row, _RULE_KEYS)
        if compact:
            out.append(compact)
    return out


def _rules_for_fit(fit: dict[str, Any]) -> list[dict[str, Any]]:
    retrieved = fit.get("retrieved") or []
    packed: list[dict[str, Any]] = []
    if isinstance(retrieved, list) and retrieved:
        for rec in retrieved:
            if not isinstance(rec, dict):
                continue
            rules = _compact_rules(list(rec.get("rules") or []))
            if not rules:
                continue
            packed.append({"query": rec.get("query"), "rules": rules})
        if packed:
            return packed
    campsite_rules = fit.get("campsite_rules") or {}
    if isinstance(campsite_rules, dict):
        for query, rows in campsite_rules.items():
            rules = _compact_rules(list(rows or []))
            if rules:
                packed.append({"query": query, "rules": rules})
    return packed


def compact_fit(fit: dict[str, Any]) -> dict[str, Any]:
    """Identity + evidence for one survivor; drop score and retrieve dumps."""
    out = _compact_mapping(fit, _FIT_KEYS)
    why = [
        _compact_mapping(entry, _WHY_KEYS)
        for entry in fit.get("why") or []
        if isinstance(entry, dict)
    ]
    if why:
        out["why"] = why
    claims = [
        _compact_mapping(entry, _CLAIM_KEYS)
        for entry in fit.get("review_claims") or []
        if isinstance(entry, dict)
    ]
    if claims:
        out["review_claims"] = claims
    rules = _rules_for_fit(fit)
    if rules:
        out["rules"] = rules
    verdicts = [
        _compact_mapping(entry, _JUDGE_KEYS)
        for entry in fit.get("claim_judge") or []
        if isinstance(entry, dict)
    ]
    if verdicts:
        out["claim_judge"] = verdicts
    return out


def pack_recommender_input(query: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Explicit pack: original query, extractor JSON, compact fits."""
    extract = payload.get("constraints")
    if not isinstance(extract, dict):
        extract = {}
    pack: dict[str, Any] = {
        "query": query,
        "reply_language": "hebrew" if query_is_hebrew(query) else "english",
        "extract": extract,
        "fits": [
            compact_fit(fit)
            for fit in payload.get("fits") or []
            if isinstance(fit, dict)
        ],
    }
    for key in ("skipped", "date_notice", "error"):
        if payload.get(key) is not None:
            pack[key] = payload[key]
    return pack


def latest_planner_payload(messages: list[BaseMessage]) -> dict[str, Any]:
    for msg in reversed(messages):
        if not isinstance(msg, ChatMessage):
            continue
        raw = message_text(msg.content).strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "fits" in data:
            return data
    extract = latest_constraints_json(messages)
    return {"fits": [], "constraints": extract}


def parse_recommender_payload(raw: str) -> dict[str, Any]:
    text = raw or ""
    try:
        parsed = _parse_json_payload(text)
    except (json.JSONDecodeError, ValueError):
        try:
            parsed = _parse_json_payload(drop_invalid_json_escapes(text))
        except (json.JSONDecodeError, ValueError):
            logger.warning("recommender unparseable: %s", text[:200])
            return {"recommendations": [], "empty": None, "intro": None}
    recs = parsed.get("recommendations")
    if not isinstance(recs, list):
        recs = []
    empty = parsed.get("empty")
    empty_text = _as_text(empty) if empty is not None else ""
    return {
        "recommendations": [row for row in recs if isinstance(row, dict)],
        "empty": empty_text or None,
        "intro": _as_text(parsed.get("intro")) or None,
    }


def _stay_windows(fit: dict[str, Any]) -> tuple[StayWindow, ...]:
    key = stay_key(fit)
    fallback = None
    if key is not None:
        fallback = StayWindow(
            key.start, key.end, _as_text(fit.get("booking_url"))
        )
    return windows_from_dates(fit.get("dates"), fallback=fallback)


def validate_recommendations(
    parsed: dict[str, Any], fits: list[dict[str, Any]]
) -> list[Recommendation]:
    """Keep at most three recs whose stay identity exists in fits."""
    by_key: dict[_StayKey, dict[str, Any]] = {}
    for fit in fits:
        if not isinstance(fit, dict):
            continue
        key = stay_key(fit)
        if key is None or key in by_key:
            continue
        by_key[key] = fit
    kept: list[Recommendation] = []
    seen: set[_StayKey] = set()
    for row in parsed.get("recommendations") or []:
        if not isinstance(row, dict):
            continue
        key = stay_key(row)
        if key is None or key not in by_key or key in seen:
            continue
        seen.add(key)
        fit = by_key[key]
        why = _as_text(row.get("why"))
        kept.append(
            Recommendation(
                campsite_id=key.campsite_id,
                campsite=_as_text(fit.get("campsite")),
                accommodation_type=key.accommodation_type,
                start=key.start,
                end=key.end,
                price_per_night=fit.get("price_per_night"),
                why=why,
                booking_url=_as_text(fit.get("booking_url")),
                price_explanation=_as_text(fit.get("price_explanation")),
                dates=_stay_windows(fit),
            )
        )
        if len(kept) >= 3:
            break
    return kept


def _pad_to_top(
    recs: list[Recommendation], fits: list[dict[str, Any]]
) -> list[Recommendation]:
    """One model pick still shows the next best fits, up to three."""
    if len(recs) != 1:
        return recs
    rows = [
        {
            "campsite_id": rec.campsite_id,
            "accommodation_type": rec.accommodation_type,
            "start": rec.start,
            "end": rec.end,
            "why": rec.why,
        }
        for rec in recs
    ]
    for fit in fits:
        key = stay_key(fit)
        if key is None:
            continue
        rows.append(
            {
                "campsite_id": key.campsite_id,
                "accommodation_type": key.accommodation_type,
                "start": key.start,
                "end": key.end,
                "why": "",
            }
        )
    return validate_recommendations({"recommendations": rows}, fits)


def _rec_windows(rec: Recommendation) -> tuple[StayWindow, ...]:
    if rec.dates:
        return rec.dates
    return (StayWindow(rec.start, rec.end, rec.booking_url),)


def _price_label(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number.is_integer():
        return f"₪{int(number)}"
    return f"₪{number:g}"


def render_recommendations(
    recs: list[Recommendation],
    *,
    empty: str | None = None,
    date_notice: str | None = None,
    intro: str | None = None,
    query: str = "",
    why_not: list[dict[str, Any]] | None = None,
) -> str:
    hebrew = query_is_hebrew(query)
    funnel = render_why_not(why_not, hebrew=hebrew)
    if not recs:
        text = _as_text(empty) or EMPTY_REPLY_FALLBACK
        notice = _as_text(date_notice)
        if notice and notice not in text:
            text = f"{notice}\n\n{text}".strip()
        if funnel:
            return f"{text}\n\n{funnel}".strip()
        return text
    labels = [stay_date_label(_rec_windows(rec)) for rec in recs]
    shared = len(set(labels)) == 1 and bool(labels[0])
    lines = [labels[0]] if shared else []
    notice = _as_text(date_notice)
    if notice:
        lines.append(notice)
    intro_text = _as_text(intro) if len(recs) >= 2 else ""
    if intro_text:
        lines.append("")
        lines.append(intro_text)
    lines.append("")
    for i, rec in enumerate(recs, start=1):
        title = rec.campsite or str(rec.campsite_id)
        if rec.accommodation_type:
            title = f"{title} — {rec.accommodation_type}"
        price = _price_label(rec.price_per_night)
        if price:
            title = f"{title} ({price})"
        lines.append(f"{i}. {title}")
        if not shared and labels[i - 1]:
            lines.append(f"   {labels[i - 1]}")
        if rec.why:
            lines.append(f"   {rec.why}")
        lines.extend(
            booking_lines(rec.dates, rec.booking_url, hebrew=hebrew)
        )
        if i < len(recs):
            lines.append("")
    if funnel:
        lines.append("")
        lines.append(funnel)
    return "\n".join(lines).strip()


def recommendation_row(rec: Recommendation) -> dict[str, Any]:
    return {
        "campsite_id": rec.campsite_id,
        "campsite": rec.campsite,
        "accommodation_type": rec.accommodation_type,
        "start": rec.start,
        "end": rec.end,
        "price_per_night": rec.price_per_night,
        "why": rec.why,
        "booking_url": rec.booking_url,
    }


def _draft_spoken_text(
    raw: str,
    fits: list[dict[str, Any]],
    *,
    date_notice: str | None = None,
    query: str = "",
    why_not: list[dict[str, Any]] | None = None,
) -> str | None:
    """Spoken reply from a possibly incomplete JSON prefix. None if nothing to show."""
    blob = json_object_prefix(raw)
    if not blob:
        return None
    parsed = parse_stream_json(blob)
    if parsed is None:
        return None
    rec_rows = parsed.get("recommendations")
    if not isinstance(rec_rows, list):
        rec_rows = []
    empty_val = parsed.get("empty")
    empty_text = empty_val.strip() if isinstance(empty_val, str) else ""
    recs = _pad_to_top(
        validate_recommendations(
            {"recommendations": [row for row in rec_rows if isinstance(row, dict)]},
            fits,
        ),
        fits,
    )
    intro_val = parsed.get("intro")
    intro_text = intro_val.strip() if isinstance(intro_val, str) else None
    if recs:
        return render_recommendations(
            recs,
            empty=None,
            date_notice=date_notice,
            intro=intro_text,
            query=query,
            why_not=why_not,
        )
    if empty_text:
        return render_recommendations(
            [],
            empty=empty_text,
            date_notice=date_notice,
            query=query,
            why_not=why_not,
        )
    return None


def recommend_from_payload(
    query: str,
    payload: dict[str, Any],
    *,
    chat: Any | None = None,
) -> RecommendResult:
    report_turn_status(RANKING)
    pack = pack_recommender_input(query, payload)
    user_msg = HumanMessage(
        content=json.dumps(pack, ensure_ascii=False, default=str)
    )
    fits = [row for row in (payload.get("fits") or []) if isinstance(row, dict)]
    date_notice = pack.get("date_notice") or (pack.get("extract") or {}).get(
        "date_notice"
    )
    notice = date_notice if isinstance(date_notice, str) else None
    why_not = payload.get("why_not")
    funnel = why_not if isinstance(why_not, list) else None

    def _run(call, *, started: float, fallback_from: str | None = None):
        on_text = _recommend_text_sink.get()
        last_draft = ""
        parts: list[str] = []
        usage_from: Any = None
        last_chunk: Any = None
        clock = RecommendClock(started)
        prepared = prepare_model(
            call.model, RECOMMENDER_SYSTEM_PROMPT, call.chat
        )
        system_msg = SystemMessage(content=prepared.system)
        for chunk in iter_chat_chunks(
            prepared.chat,
            [system_msg, user_msg],
            first_token_sec=call.first_token_sec,
        ):
            last_chunk = chunk
            if langchain_chat_usage(chunk) is not None:
                usage_from = chunk
            if chunk_thinking(chunk):
                clock.note_thinking()
            delta = chunk_text(chunk)
            if not delta:
                clock.note_empty()
                continue
            clock.note_text()
            parts.append(delta)
            draft = _draft_spoken_text(
                "".join(parts),
                fits,
                date_notice=notice,
                query=query,
                why_not=funnel,
            )
            if draft is None or draft == last_draft:
                continue
            last_draft = draft
            clock.note_spoken()
            if on_text is not None:
                on_text(draft)
        if usage_from is None:
            usage_from = last_chunk
        timings = record_recommend(
            clock,
            model=call.model,
            fallback_from=fallback_from,
            extra=prepared.extra_body,
            usage_from=usage_from,
        )
        raw = "".join(parts)
        parsed = parse_recommender_payload(raw)
        recs = _pad_to_top(validate_recommendations(parsed, fits), fits)
        empty = parsed.get("empty") if not recs else None
        intro = parsed.get("intro") if len(recs) >= 2 else None
        text = render_recommendations(
            recs,
            empty=empty,
            date_notice=notice,
            intro=intro,
            query=query,
            why_not=funnel,
        )
        if on_text is not None and text != last_draft:
            on_text(text)
        return RecommendResult(
            recommendations=tuple(recs),
            empty=empty,
            text=text,
            intro=intro,
            ttft_chunk_ms=timings.chunk_ms,
            ttft_spoken_ms=timings.spoken_ms,
            elapsed_ms=timings.elapsed_ms,
            reasoning_tokens=timings.reasoning_tokens,
            thinking_stream=timings.thinking_stream,
        )

    primary = primary_recommend_call(recommender_model(), chat)
    with stage("recommend"):
        return recommend_with_fallback(
            primary, kimi_super_fallback(primary), _run
        )


def recommend_from_messages(messages: list[BaseMessage]) -> RecommendResult:
    query = latest_user_text(messages)
    payload = latest_planner_payload(messages)
    if "constraints" not in payload:
        payload = dict(payload)
        payload["constraints"] = latest_constraints_json(messages)
    return recommend_from_payload(query, payload)
