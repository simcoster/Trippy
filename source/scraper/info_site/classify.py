"""Classify rate-card labels into accommodation / guest / period / kind."""

from __future__ import annotations

import re
from amenity_enrichment.llm import (
    LlmUsage,
    QWEN_INSTRUCT_30B_MODEL,
    _parse_json_payload,
    make_nebius_openai_client,
)
from openai import OpenAI

from .parse import normalize_label
from .schemas import ClassificationPayload, ClassifiedPriceRow, RawPriceRow

_FEE_PREFIX_RE = re.compile(r"^תוספת\b")
_WS_RE = re.compile(r"\s+")


SYSTEM_PROMPT = """You classify Hebrew campsite rate-card row labels into JSON.

Output valid JSON only, no markdown:
{
  "accommodation_type": string,
  "guest_type": "adult" | "child" | "any",
  "rate_period": "weekday" | "weekend_holiday" | "any",
  "kind": "lodging" | "fee"
}

Rules:
- accommodation_type is the canonical catalog name availability will match.
  Use INPA-style Hebrew labels, not the full sentence.
  Examples:
    "לינת שטח באוהלים פרטיים - מבוגר" → "לינת שטח באוהלים פרטיים"
    "לינה בבונגלו עם מזגן אמצע שבוע" → "בונגלו עם מזגן"
    "לינה בחדרי צוות אמצע שבוע" → "חדר צוות"
    "לינה בחדר צוות עץ סופי שבוע וחגים" → "חדר צוות עץ"
    "עמדת חניה לקרוואן (עד 2 לנים)" → "עמדת חניה לקרוואן"
    "לינה בחושות אמצע שבוע" → "חושה"
    "לינה בחושה כפולה עם מזגן סופי שבוע וחגים" → "חושה כפולה עם מזגן"
- guest_type: "adult" if מבוגר, "child" if ילד, else "any" (per-unit lodging).
- rate_period: "weekday" if אמצע שבוע, "weekend_holiday" if סופי שבוע / סוף שבוע / חגים, else "any".
- kind: "fee" if the row is a surcharge (תוספת, late checkout, extra person), else "lodging".
"""


def is_fee_label(label: str) -> bool:
    """Rate-card rows that start with תוספת are fees (late checkout, extras)."""
    return bool(_FEE_PREFIX_RE.match(normalize_label(label)))


def _guest_type_from_label(label: str) -> str:
    if re.search(r"\bמבוגר\b", label):
        return "adult"
    if re.search(r"\bילד\b", label):
        return "child"
    return "any"


def _rate_period_from_label(label: str) -> str:
    if "אמצע שבוע" in label:
        return "weekday"
    if "סופי שבוע" in label or "סוף שבוע" in label or "ס. שבוע" in label:
        return "weekend_holiday"
    return "any"


def _fee_accommodation_type(label: str) -> str:
    text = normalize_label(label)
    text = _FEE_PREFIX_RE.sub("", text).strip()
    text = re.sub(r"^יציאה מאוחרת\s*", "", text).strip()
    text = re.sub(
        r"\s*(אמצע שבוע|סופי שבוע וחגים|סופי שבוע|סוף שבוע|ס\.\s*שבוע)\s*$",
        "",
        text,
    ).strip()
    return _WS_RE.sub(" ", text) if text else normalize_label(label)


def classify_fee_row(raw: RawPriceRow | dict) -> ClassifiedPriceRow:
    """Heuristic classify for תוספת rows (no LLM)."""
    row = raw if isinstance(raw, RawPriceRow) else RawPriceRow.model_validate(raw)
    label = normalize_label(row.raw_label)
    return ClassifiedPriceRow(
        raw_label=label,
        price=row.price,
        notes=row.notes,
        accommodation_type=_fee_accommodation_type(label),
        guest_type=_guest_type_from_label(label),
        rate_period=_rate_period_from_label(label),
        kind="fee",
    )


class RateCardClassifier:
    """Qwen 30B classifier for lodging rate-card labels."""

    MODEL = QWEN_INSTRUCT_30B_MODEL
    TEMPERATURE = 0

    def __init__(
        self,
        client: OpenAI | None = None,
        *,
        model: str | None = None,
    ) -> None:
        self.client = client or make_nebius_openai_client()
        self.model = model or self.MODEL

    def classify_label(
        self,
        raw_label: str,
        *,
        usage: LlmUsage | None = None,
    ) -> ClassificationPayload:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Label: {raw_label}"},
            ],
            temperature=self.TEMPERATURE,
        )
        if usage is not None:
            usage.add_chat(response.usage)
        content = response.choices[0].message.content or ""
        data = _parse_json_payload(content)
        return ClassificationPayload.model_validate(data)


def classify_row(
    raw: RawPriceRow | dict,
    *,
    classifier: RateCardClassifier | None = None,
    cache: dict[str, ClassificationPayload] | None = None,
    usage: LlmUsage | None = None,
) -> ClassifiedPriceRow:
    row = raw if isinstance(raw, RawPriceRow) else RawPriceRow.model_validate(raw)
    label = normalize_label(row.raw_label)
    if is_fee_label(label):
        return classify_fee_row(row)

    if classifier is None:
        raise RuntimeError("RateCardClassifier is required for lodging rows")

    store = cache if cache is not None else {}
    payload = store.get(label)
    if payload is None:
        payload = classifier.classify_label(label, usage=usage)
        store[label] = payload
    return ClassifiedPriceRow(
        raw_label=label,
        price=row.price,
        notes=row.notes,
        accommodation_type=payload.accommodation_type.strip(),
        guest_type=payload.guest_type,
        rate_period=payload.rate_period,
        kind=payload.kind,
    )


def classify_rows(
    raws: list[RawPriceRow | dict],
    *,
    classifier: RateCardClassifier | None = None,
    usage: LlmUsage | None = None,
) -> list[ClassifiedPriceRow]:
    cache: dict[str, ClassificationPayload] = {}
    return [
        classify_row(raw, classifier=classifier, cache=cache, usage=usage)
        for raw in raws
    ]


def lodging_rows_to_persist(
    rows: list[ClassifiedPriceRow],
) -> list[ClassifiedPriceRow]:
    """v1 persist filter: lodging only. Fees are parsed but not stored."""
    return [row for row in rows if row.kind == "lodging"]
