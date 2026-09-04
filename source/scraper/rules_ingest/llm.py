"""Hebrew section text → structured rule/amenity statements."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from openai import OpenAI

from source.scraper.amenity_enrichment.llm import (
    _FENCE_RE,
    QWEN_INSTRUCT_MODEL,
    LlmUsage,
    _parse_json_payload,
    make_nebius_openai_client,
)
from source.scraper.rules_ingest.schemas import RuleExtract

SYSTEM_PROMPT = """You are a precise JSON extraction engine.
Extract site-level rules and amenities from a Hebrew section of a campsite page.

You are given the section title and its text. Emit one statement per distinct fact.

Rules:
- Output valid JSON only, without markdown wrappers.
- subject: a lower snake_case English label with ONE canonical shape:
    <topic>[_<scope>]_<predicate>      for a rule
    <thing>[_in_<place>]               for an amenity: a bare noun, no predicate.
                                       Whether it is provided is `polarity`.
  topic / thing names the matter in as few words as possible, and every
  statement about the same matter in this section REUSES it verbatim:
  late_check_out throughout, never late_stay or late_departure on the next line.
- predicate (rules only) is the LAST part of the name and is one of exactly:
    allowed       may a guest do it                       -> polarity
    required      must a guest do it, or pay it           -> polarity
    time          the hour something happens or ends      -> qualifier hour_of_day
    fee_ils       what it costs                           -> qualifier ils
    fee_percent   the cost as a share of another price    -> qualifier percent
    min_age  max_age  min_nights  max_nights  min_occupancy  max_occupancy  count
  Never coin another predicate. Map every synonym onto these:
    available / possible / permitted / an option  -> allowed
    applies / charged / payable / subject to fee   -> required
    until / by / no later than / deadline          -> time
  A range is two statements whose scope says which end: <topic>_start_time and
  <topic>_end_time. "until 17:00" is <topic>_end_time. A numeric range is
  likewise a min and a max of ONE topic, never a single statement:
  "30-80 לנים" is <topic>_min_occupancy 30 AND <topic>_max_occupancy 80;
  "מעל 80" is only <topic>_min_occupancy 80; "מגיל 5 ועד 14" is <topic>_min_age 5
  AND <topic>_max_age 14.
- scope is the condition under which the rule holds. Keep it short, put it
  BETWEEN topic and predicate, and phrase it identically every time it recurs:
  on_saturday_evening, in_accommodation_units, for_dogs. The unconditional rule
  and a conditional variant are TWO subjects, both emitted when both are stated:
    late_check_out_allowed                       (in general)
    late_check_out_on_saturday_evening_allowed   (the Saturday variant)
- An amenity that is part of another amenity names the part, then the container:
  gas_in_field_kitchen, hot_water_in_showers, lighting_in_tents. The container
  is its own subject. One list item can yield both, with different polarities:
    "מטבח שדה (1) בשלב הזה בלי גז"
      -> field_kitchen         / amenity / true  / 1    / count
      -> gas_in_field_kitchen  / amenity / false / null / none
- ALWAYS phrase the subject POSITIVELY. Negation goes in `polarity`, never in the
  name. Never emit not_/no_/cant_/cannot_/without_/_forbidden/_banned subjects.
  - "הכניסה לכלבים אסורה"      -> dogs_allowed / boolean_rule / false
  - "יש להצטייד במגבות"        -> towels / amenity / false
  - "כלבים חייבים מחסום"       -> muzzle_for_dogs_required / boolean_rule / true
- Ignore hedges about time ("for now", "at this stage", "currently"): polarity
  states what holds today.
- category: one of three.
  "amenity"      something the site provides or does not provide
                 (shower, refrigerator, electric_hookup, towels).
  "boolean_rule" something a guest may, must or must not do, answered by
                 polarity. Every rule whose predicate is allowed or required:
                 dogs_allowed, late_check_out_fee_required.
  "numeric_rule" a limit or a time, answered by a number. Every rule whose
                 predicate is time, fee_ils, fee_percent, min_age, max_age,
                 min_nights, max_nights, min_occupancy, max_occupancy or count:
                 check_out_time, weekend_min_nights, adult_min_age,
                 late_check_out_fee_percent.
  A boolean_rule always carries a polarity; a numeric_rule always carries a
  qualifier. One sentence can yield several categories:
  "ניתן להדליק מנגל בציוד עצמי" -> barbecue_allowed (boolean_rule, true)
                                 + barbecue_equipment (amenity, false)
- polarity: true when the thing is allowed or provided, false when it is forbidden
  or explicitly not provided, null when the statement is purely a quantity.
- Every statement must carry a polarity OR a qualifier. One with neither says
  nothing and is discarded. When a fact has no number, name the thing being
  asserted and answer it with polarity:
  "מרכז שירות למבקר / שעות פתיחה: על פי צורך"
    -> visitor_service_center               / amenity / true  / null / none
    -> visitor_service_center_regular_hours_allowed / boolean_rule / false / null / none
  (opening "as needed" means it does NOT keep regular hours)
- qualifier: the number the statement carries, or null. Write a time of day as a
  decimal hour: "20:30" -> 20.5, "12:00" -> 12.
  Never emit the same subject twice in one section — the second one is discarded.
- qualifier_unit: one of
  none, count, hour_of_day, nights, days, years, ils, meters, percent.
  Use none when qualifier is null.
- Counts are worth keeping: "ברזיות מים לשתייה (6)" is
  subject "drinking_water_fountain", category amenity, polarity true,
  qualifier 6, unit count.
- evidence_span: the Hebrew sentence or list item you read it from, verbatim.
- confidence: 0..1.
- This section describes the CAMPSITE AS A WHOLE. Ignore anything specific to one
  room or unit type, and ignore prices, phone numbers, addresses and marketing.
- Emit nothing rather than guessing. An empty list is a valid answer.

Examples:
- "הכניסה לחניון הלילה החל מהשעה 15:00 עד השעה 20:30 לכל המאוחר"
  -> check_in_start_time / numeric_rule / null / 15   / hour_of_day
  -> check_in_end_time   / numeric_rule / null / 20.5 / hour_of_day
- "יש לפנות את האוהלים עד השעה 12:00 ביום העזיבה"
  -> check_out_time / numeric_rule / null / 12 / hour_of_day
- "לנים המבקשים להישאר באתר לאחר השעה 12:00 ועד לסיום שעות הפעילות בשעה 17:00
   נדרשים לתשלום של 50% מדמי כניסת יום לאתר"
  -> late_check_out_allowed      / boolean_rule / true / null / none
  -> late_check_out_end_time     / numeric_rule / null / 17   / hour_of_day
  -> late_check_out_fee_percent  / numeric_rule / null / 50   / percent
- "יציאה מאוחרת במוצאי שבת וחג לאחר שעת סגירת האתר בתוספת תשלום"
  -> late_check_out_on_saturday_evening_allowed      / boolean_rule / true / null / none
  -> late_check_out_on_saturday_evening_fee_required / boolean_rule / true / null / none
- "ביחידות האירוח תתאפשר יציאה מאוחרת בתוספת תשלום ועל בסיס מקום פנוי"
  -> late_check_out_in_accommodation_units_allowed      / boolean_rule / true / null / none
  -> late_check_out_in_accommodation_units_fee_required / boolean_rule / true / null / none
- "חלוקת מזרנים בין השעות 15:00-20:00"
  -> mattress_pickup_start_time / numeric_rule / null / 15 / hour_of_day
  -> mattress_pickup_end_time   / numeric_rule / null / 20 / hour_of_day
- "מותנה במינימום 2 לילות" (weekend rate)
  -> weekend_min_nights / numeric_rule / null / 2 / nights
- "גיל 14 ומעלה" (adult rate)
  -> adult_min_age / numeric_rule / null / 14 / years
- "תיאום לינה לקבוצות משפחות וחברים (30-80 לנים המשלמים יחד)"
  -> family_and_friends_group_min_occupancy / numeric_rule / null / 30 / count
  -> family_and_friends_group_max_occupancy / numeric_rule / null / 80 / count
- "תיאום לינה לקבוצות (מעל 80 לנים)"
  -> group_min_occupancy / numeric_rule / null / 80 / count
- "ניתן להדליק מצלה (מנגל) בציוד עצמי"
  -> barbecue_allowed   / boolean_rule    / true  / null / none
  -> barbecue_equipment / amenity / false / null / none
- "החניון אינו מותאם לציבור שומרי השבת"
  -> shabbat_observance_suitable_allowed / boolean_rule / false / null / none

Schema:
{
  "statements": [
    {
      "subject": str,
      "category": "amenity" | "boolean_rule" | "numeric_rule",
      "polarity": bool | null,
      "qualifier": number | null,
      "qualifier_unit": "none"|"count"|"hour_of_day"|"nights"|"days"|"years"|"ils"|"meters"|"percent",
      "evidence_span": str,
      "confidence": number
    }
  ]
}
"""


def _coerce_payload(raw: str) -> dict[str, Any]:
    """Accept `{"statements": [...]}` or the bare list the model sometimes sends.

    A section with nothing to extract reliably comes back as `[]` rather than
    `{"statements": []}`, and the shared parser rejects a non-dict outright.
    """
    try:
        return _parse_json_payload(raw)
    except ValueError:
        pass
    text = _FENCE_RE.sub("", (raw or "").strip()).strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        raise ValueError(f"expected an object or a list, got: {text[:80]!r}")
    listed = json.loads(text[start : end + 1])
    if not isinstance(listed, list):
        raise ValueError(f"expected a list, got {type(listed).__name__}")
    return {"statements": listed}


class RuleExtractorLLMClient:
    """Nebius chat model for Hebrew section text → rule/amenity statements."""

    MODEL = QWEN_INSTRUCT_MODEL
    TEMPERATURE = 0
    # A dense amenity list runs long: `מה בחניון?` at Akhziv covers two
    # sub-campsites, ~30 amenities each carrying a Hebrew evidence span, and
    # Hebrew tokenises at roughly one token per one to two characters. At 2500
    # the reply was cut mid-object, failed to parse, and the whole section was
    # dropped as if it had produced nothing.
    MAX_TOKENS = 8000

    def __init__(
        self,
        client: OpenAI | None = None,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._client = client
        self.model = model or self.MODEL
        self.system_prompt = system_prompt or SYSTEM_PROMPT

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = make_nebius_openai_client()
        return self._client

    def extract(
        self,
        text: str,
        *,
        section_title: str,
        usage: LlmUsage | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> RuleExtract:
        """Extract one section's statements.

        The reply is streamed so a caller can show life during the 30-90s a
        dense section takes; `progress` is called with the running chunk count
        as deltas arrive. The JSON is still parsed only once it is complete.
        Token counts arrive on a final, choice-less chunk, and only because
        `stream_options` asks for them.
        """
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": f"Section: {section_title}\n\nText:\n{text}",
                },
            ],
            temperature=self.TEMPERATURE,
            max_tokens=self.MAX_TOKENS,
            stream=True,
            stream_options={"include_usage": True},
        )
        parts: list[str] = []
        finish_reason: str | None = None
        final_usage: Any | None = None
        chunks = 0
        for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                final_usage = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if content:
                parts.append(content)
            if getattr(choice, "finish_reason", None) is not None:
                finish_reason = choice.finish_reason
            chunks += 1
            if progress is not None:
                progress(chunks)
        if usage is not None:
            usage.add_chat(final_usage, role="rules_extract", model=self.model)
        if finish_reason == "length":
            # Say what actually went wrong. Truncated JSON surfaces as a parse
            # error otherwise, which sends you looking in the wrong place.
            raise ValueError(
                f"reply truncated at max_tokens={self.MAX_TOKENS} for section "
                f"{section_title!r}; the section needs splitting or a bigger cap"
            )
        data = _coerce_payload("".join(parts))
        return RuleExtract.model_validate(data)
