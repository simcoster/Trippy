"""Pick 1–2 planner fits and write a cited Hebrew recommendation."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
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
from langchain_core.utils.json import parse_partial_json

from source.agent.constraints import latest_constraints_json
from source.agent.messages import latest_user_text, message_text
from source.agent.prompts import EMPTY_REPLY_FALLBACK
from source.agent.timing import stage
from source.scraper.amenity_enrichment.llm import (
    KIMI_K3_MODEL,
    NEMOTRON_SUPER_MODEL,
    QWEN_INSTRUCT_MODEL,
    _parse_json_payload,
    collected_llm_usage,
    langchain_chat_usage,
    make_agent_chat_model,
)

logger = logging.getLogger(__name__)

RECOMMENDER_SYSTEM_PROMPT = """
You pick campsite stays for Trippy. Output JSON only.

The user JSON has query (original ask), extract (structured constraints),
and fits (available stays that already match dates, party size, and price
when those were given). Recommend only from fits. Never invent a campsite,
type, date, price, amenity, or rule.

Pick 1 stay, or 2 when they are genuinely different useful options
(prefer two campsites over two types at the same site). Never more than 2.
When you pick 2, set intro to a short spoken note: there is more than
one option, what they are, and how they differ (tent vs staff room,
north vs south, cheaper vs closer). Phrase it however sounds natural;
do not use a fixed template. Each why is only about that stay; do not
repeat the intro there. intro is null when you pick 1.
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

Language: pick one from query and stay in it. Packed JSON is English
(field names, snake_case subjects, review claims). None of that English
belongs in why, intro, or empty.

If query is mostly Hebrew, why, intro, and empty are Hebrew only — no
Latin, CJK, or mixed-script tokens. For reviews write אורחים מספרים,
never Guests / guests / ゲuests / ospites. Do not write pitch,
tent_pitch, outlets, bungalow, camping, accommodation, Stay, stations,
dank, or glue Latin inside a Hebrew word (not בungalו, איןoutlets,
יש.pitch). If query is mostly English, why, intro, and empty are
English only — no Hebrew prose; write "guests report" for reviews.
Copy campsite and accommodation_type names from the fit as stored.

Output JSON only:
{"recommendations": [{"campsite_id": int, "accommodation_type": str,
 "start": str, "end": str, "booking_url": str, "why": str}],
 "intro": str | null, "empty": str | null}
""".strip()

RECOMMENDER_NO_THINK_SUFFIX = "/no_think"

_override_recommender = None
_override_recommender_key = None
_recommend_text_sink: ContextVar[Callable[[str], None] | None] = ContextVar(
    "trippy_recommend_text", default=None
)
_recommend_timing: ContextVar[dict[str, float | None] | None] = ContextVar(
    "trippy_recommend_timing", default=None
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


@dataclass(frozen=True)
class RecommendResult:
    recommendations: tuple[Recommendation, ...]
    empty: str | None
    text: str
    intro: str | None = None
    ttft_chunk_ms: float | None = None
    ttft_spoken_ms: float | None = None
    elapsed_ms: float | None = None


def recommender_model() -> str:
    """Kimi-K3 unless `TRIPPY_RECOMMENDER_MODEL` is super / 235B / a full id."""
    raw = (os.environ.get("TRIPPY_RECOMMENDER_MODEL") or "").strip()
    key = raw.casefold()
    if not raw or key in {"kimi", "kimi-k3", "k3"}:
        return KIMI_K3_MODEL
    if key in {"super", "nemotron", "nemotron-super"}:
        return NEMOTRON_SUPER_MODEL
    if key in {"235b", "big", "qwen"}:
        return QWEN_INSTRUCT_MODEL
    return raw


def _recommender_system() -> str:
    prompt = RECOMMENDER_SYSTEM_PROMPT
    if "nemotron" in recommender_model().casefold():
        return f"{prompt}\n\n{RECOMMENDER_NO_THINK_SUFFIX}"
    return prompt


def _recommender_extra_body() -> dict[str, Any] | None:
    model = recommender_model().casefold()
    if "nemotron" in model:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    if "instruct-2507" in model or model in {"235b", "big", "qwen"}:
        return None
    extra: dict[str, Any] = {
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "thinking": {"type": "disabled"},
    }
    if "kimi" in model:
        extra["reasoning_effort"] = "none"
    return extra


def _recommender_chat():
    """Default Kimi recommender; `TRIPPY_RECOMMENDER_MODEL` rebuilds for probes."""
    global _override_recommender, _override_recommender_key
    model = recommender_model()
    extra = _recommender_extra_body()
    key = (model, json.dumps(extra, sort_keys=True) if extra else "")
    if _override_recommender is None or _override_recommender_key != key:
        _override_recommender = make_agent_chat_model(
            temperature=0, model=model, extra_body=extra
        )
        _override_recommender.stream_usage = True
        _override_recommender_key = key
    return _override_recommender


_warmup_lock = threading.Lock()
_warmup_started = False


def warmup_recommender(*, chat: Any | None = None, blocking: bool = False) -> None:
    """One-token `hi` so the first real recommend is not a cold replica."""
    global _warmup_started
    with _warmup_lock:
        if _warmup_started:
            return
        _warmup_started = True

    def _ping() -> None:
        try:
            client = chat or _recommender_chat()
            if hasattr(client, "bind"):
                client = client.bind(max_tokens=1)
            client.invoke([HumanMessage(content="hi")])
        except Exception:
            logger.warning("recommender warmup failed", exc_info=True)

    if blocking:
        _ping()
        return
    threading.Thread(
        target=_ping, daemon=True, name="recommender-warmup"
    ).start()


def last_recommend_timing() -> dict[str, float | None] | None:
    """TTFT inside the recommend call: first LLM chunk, first spoken paint."""
    return _recommend_timing.get()


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
    "booking_url",
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
            parsed = _parse_json_payload(_drop_invalid_json_escapes(text))
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


def validate_recommendations(
    parsed: dict[str, Any], fits: list[dict[str, Any]]
) -> list[Recommendation]:
    """Keep at most two recs whose stay identity exists in fits."""
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
            )
        )
        if len(kept) >= 2:
            break
    return kept


def _day_month(iso: str) -> str:
    parts = _as_text(iso).split("-")
    if len(parts) < 3:
        return _as_text(iso)
    try:
        return f"{int(parts[2])}.{int(parts[1])}"
    except ValueError:
        return _as_text(iso)


def _date_lead(recs: list[Recommendation]) -> str:
    ranges: list[str] = []
    seen: set[str] = set()
    for rec in recs:
        label = _day_month(rec.start)
        if rec.end and rec.end != rec.start:
            end_label = _day_month(rec.end)
            if end_label != label:
                label = f"{label}–{end_label}"
        if label not in seen:
            seen.add(label)
            ranges.append(label)
    return ", ".join(ranges)


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
) -> str:
    if not recs:
        text = _as_text(empty) or EMPTY_REPLY_FALLBACK
        notice = _as_text(date_notice)
        if notice and notice not in text:
            return f"{notice}\n\n{text}".strip()
        return text
    lines = [_date_lead(recs)]
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
        if rec.why:
            lines.append(f"   {rec.why}")
        if rec.booking_url:
            lines.append(f"   {rec.booking_url}")
        if i < len(recs):
            lines.append("")
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


def _chunk_text(chunk: Any) -> str:
    return message_text(getattr(chunk, "content", None))


def _iter_chat_chunks(chat: Any, messages: list[Any]) -> Iterator[Any]:
    if not hasattr(chat, "stream"):
        yield chat.invoke(messages)
        return
    try:
        stream = chat.stream(messages, stream_usage=True)
    except TypeError:
        stream = chat.stream(messages)
    yield from stream


def _json_object_prefix(raw: str) -> str:
    start = (raw or "").find("{")
    if start < 0:
        return ""
    return raw[start:]


def _drop_invalid_json_escapes(text: str) -> str:
    """Keep only JSON string escapes; turn `\\pitch` into `pitch`."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "\\":
            out.append(text[i])
            i += 1
            continue
        nxt = text[i + 1] if i + 1 < n else ""
        if nxt in '"\\/bfnrt':
            out.append(text[i : i + 2])
            i += 2
            continue
        hex_digits = "0123456789abcdefABCDEF"
        if (
            nxt == "u"
            and i + 5 < n
            and all(ch in hex_digits for ch in text[i + 2 : i + 6])
        ):
            out.append(text[i : i + 6])
            i += 6
            continue
        i += 1
    return "".join(out)


def _parse_stream_json(blob: str) -> dict[str, Any] | None:
    """parse_partial_json re-raises on a finished-but-illegal `\\escape`."""
    for candidate in (blob, _drop_invalid_json_escapes(blob)):
        try:
            parsed = parse_partial_json(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _draft_spoken_text(
    raw: str,
    fits: list[dict[str, Any]],
    *,
    date_notice: str | None = None,
) -> str | None:
    """Spoken reply from a possibly incomplete JSON prefix. None if nothing to show."""
    blob = _json_object_prefix(raw)
    if not blob:
        return None
    parsed = _parse_stream_json(blob)
    if parsed is None:
        return None
    rec_rows = parsed.get("recommendations")
    if not isinstance(rec_rows, list):
        rec_rows = []
    empty_val = parsed.get("empty")
    empty_text = empty_val.strip() if isinstance(empty_val, str) else ""
    recs = validate_recommendations(
        {"recommendations": [row for row in rec_rows if isinstance(row, dict)]},
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
        )
    if empty_text:
        return render_recommendations(
            [], empty=empty_text, date_notice=date_notice
        )
    return None


def recommend_from_payload(
    query: str,
    payload: dict[str, Any],
    *,
    chat: Any | None = None,
) -> RecommendResult:
    pack = pack_recommender_input(query, payload)
    system_msg = SystemMessage(content=_recommender_system())
    user_msg = HumanMessage(
        content=json.dumps(pack, ensure_ascii=False, default=str)
    )
    fits = [row for row in (payload.get("fits") or []) if isinstance(row, dict)]
    date_notice = pack.get("date_notice") or (pack.get("extract") or {}).get(
        "date_notice"
    )
    notice = date_notice if isinstance(date_notice, str) else None
    on_text = _recommend_text_sink.get()
    last_draft = ""
    parts: list[str] = []
    usage_from: Any = None
    last_chunk: Any = None
    started = time.perf_counter()
    chunk_at: float | None = None
    spoken_at: float | None = None
    with stage("recommend"):
        for chunk in _iter_chat_chunks(
            chat or _recommender_chat(), [system_msg, user_msg]
        ):
            last_chunk = chunk
            if langchain_chat_usage(chunk) is not None:
                usage_from = chunk
            delta = _chunk_text(chunk)
            if not delta:
                continue
            if chunk_at is None:
                chunk_at = time.perf_counter()
            parts.append(delta)
            draft = _draft_spoken_text("".join(parts), fits, date_notice=notice)
            if draft is None or draft == last_draft:
                continue
            last_draft = draft
            if spoken_at is None:
                spoken_at = time.perf_counter()
            if on_text is not None:
                on_text(draft)
        if usage_from is None:
            usage_from = last_chunk
    elapsed_ms = (time.perf_counter() - started) * 1000
    chunk_ms = (chunk_at - started) * 1000 if chunk_at is not None else None
    spoken_ms = (spoken_at - started) * 1000 if spoken_at is not None else None
    _recommend_timing.set(
        {
            "chunk_ms": chunk_ms,
            "spoken_ms": spoken_ms,
            "total_ms": elapsed_ms,
        }
    )
    sink = collected_llm_usage()
    raw_usage = langchain_chat_usage(usage_from) if usage_from is not None else None
    if sink is not None and raw_usage is not None:
        sink.add_chat(raw_usage, role="recommend", model=recommender_model())
    raw = "".join(parts)
    parsed = parse_recommender_payload(raw)
    recs = validate_recommendations(parsed, fits)
    empty = parsed.get("empty") if not recs else None
    intro = parsed.get("intro") if len(recs) >= 2 else None
    text = render_recommendations(
        recs, empty=empty, date_notice=notice, intro=intro
    )
    if on_text is not None and text != last_draft:
        on_text(text)
    return RecommendResult(
        recommendations=tuple(recs),
        empty=empty,
        text=text,
        intro=intro,
        ttft_chunk_ms=chunk_ms,
        ttft_spoken_ms=spoken_ms,
        elapsed_ms=elapsed_ms,
    )


def recommend_from_messages(messages: list[BaseMessage]) -> RecommendResult:
    query = latest_user_text(messages)
    payload = latest_planner_payload(messages)
    if "constraints" not in payload:
        payload = dict(payload)
        payload["constraints"] = latest_constraints_json(messages)
    return recommend_from_payload(query, payload)
