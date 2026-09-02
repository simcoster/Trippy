"""Hebrew section text → structured rule/amenity statements."""

from __future__ import annotations

import json
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
- subject: a lower snake_case English label naming the predicate.
- ALWAYS phrase the subject POSITIVELY. Negation goes in `polarity`, never in the
  name. Never emit not_/no_/cant_/cannot_/without_/_forbidden/_banned subjects.
  - "הכניסה לכלבים אסורה"      -> subject "dogs_allowed", polarity false
  - "יש להצטייד במגבות"        -> subject "towels_included", polarity false
  - "כלבים חייבים מחסום"       -> subject "dogs_must_wear_a_muzzle", polarity true
- category: "amenity" for something the site provides or does not provide
  (shower, refrigerator, electric_hookup, towels); "rule" for something a guest
  may, must or must not do, or a limit on a stay (dogs_allowed, check_out_time,
  min_weekend_nights, adult_min_age). One sentence can yield both:
  "ניתן להדליק מנגל בציוד עצמי" -> barbecue_allowed (rule, true)
                                 + barbecue_equipment_included (amenity, false)
- polarity: true when the thing is allowed or provided, false when it is forbidden
  or explicitly not provided, null when the statement is purely a quantity.
- Every statement must carry a polarity OR a qualifier. One with neither says
  nothing and is discarded. When a fact has no number, name the thing being
  asserted and answer it with polarity:
  "מרכז שירות למבקר / שעות פתיחה: על פי צורך"
    -> visitor_service_center        / amenity / true  / null / none
    -> service_center_regular_hours  / rule    / false / null / none
  (opening "as needed" means it does NOT keep regular hours)
- Direction belongs in the subject name, not in a separate field:
  min_weekend_nights, max_occupancy, check_in_time, check_out_time,
  latest_arrival_time, last_dogs_entry_time, pool_min_age, adult_min_age.
- qualifier: the number the statement carries, or null. Write a time of day as a
  decimal hour: "20:30" -> 20.5, "12:00" -> 12.
- A RANGE is two statements with two subjects, never one. Only one number can be
  stored per subject, so put the end of the range in the name:
  "חלוקת מזרנים בין השעות 15:00-20:00"
    -> mattress_pickup_start_time / rule / null / 15 / hour_of_day
    -> mattress_pickup_end_time   / rule / null / 20 / hour_of_day
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
  -> check_in_time / rule / null / 15 / hour_of_day
  -> latest_arrival_time / rule / null / 20.5 / hour_of_day
- "יש לפנות את האוהלים עד השעה 12:00 ביום העזיבה"
  -> check_out_time / rule / null / 12 / hour_of_day
- "מותנה במינימום 2 לילות" (weekend rate)
  -> min_weekend_nights / rule / null / 2 / nights
- "גיל 14 ומעלה" (adult rate)
  -> adult_min_age / rule / null / 14 / years
- "ניתן להדליק מצלה (מנגל) בציוד עצמי"
  -> barbecue_allowed / rule / true / null / none
  -> barbecue_equipment_included / amenity / false / null / none
- "החניון אינו מותאם לציבור שומרי השבת"
  -> suitable_for_shabbat_observers / rule / false / null / none

Schema:
{
  "statements": [
    {
      "subject": str,
      "category": "amenity" | "rule",
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
    MAX_TOKENS = 2500

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
    ) -> RuleExtract:
        response = self.client.chat.completions.create(
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
        )
        if usage is not None:
            usage.add_chat(response.usage)
        data = _coerce_payload(response.choices[0].message.content or "")
        return RuleExtract.model_validate(data)
