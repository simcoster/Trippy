"""Pydantic payloads for the rule extractor."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, field_validator

from db.models import QualifierUnit, SubjectCategory

# The names the extractor writes; anything else falls back to NONE.
UNIT_BY_NAME: dict[str, QualifierUnit] = {
    "none": QualifierUnit.NONE,
    "count": QualifierUnit.COUNT,
    "hour_of_day": QualifierUnit.HOUR_OF_DAY,
    "nights": QualifierUnit.NIGHTS,
    "days": QualifierUnit.DAYS,
    "years": QualifierUnit.YEARS,
    "ils": QualifierUnit.ILS,
    "meters": QualifierUnit.METERS,
    "percent": QualifierUnit.PERCENT,
}


class RuleStatement(BaseModel):
    """One extracted fact about a campsite."""

    subject: str
    # The extractor reads the sentence, so it knows whether it is stating a
    # provision, a permission or a number far better than a classifier shown
    # one word. Amenities, boolean rules and numeric rules are searched apart,
    # so a permission is never a merge candidate for a deadline on the same
    # topic. None when the model omitted it: search every category rather than
    # assert a category that may be wrong.
    category: int | None = None
    polarity: bool | None = None
    qualifier: Decimal | None = None
    qualifier_unit: int = int(QualifierUnit.NONE)
    evidence_span: str | None = None
    confidence: float | None = None

    @field_validator("subject")
    @classmethod
    def _require_subject(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("subject must not be empty")
        return text

    @field_validator("category", mode="before")
    @classmethod
    def _coerce_category(cls, value: object) -> int | None:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            text = value.strip().casefold()
            if text.startswith("amenit"):
                return int(SubjectCategory.AMENITY)
            if text.startswith("bool"):
                return int(SubjectCategory.BOOLEAN_RULE)
            if text.startswith("num"):
                return int(SubjectCategory.NUMERIC_RULE)
            if not text.isdigit():
                # A bare "rule" no longer says which kind; None searches everything.
                return None
        try:
            number = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return number if number in {int(c) for c in SubjectCategory} else None

    @field_validator("qualifier", mode="before")
    @classmethod
    def _coerce_qualifier(cls, value: object) -> object:
        if value is None or value == "":
            return None
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            return None

    @field_validator("qualifier_unit", mode="before")
    @classmethod
    def _coerce_unit(cls, value: object) -> int:
        if isinstance(value, str):
            return int(UNIT_BY_NAME.get(value.strip().casefold(), QualifierUnit.NONE))
        try:
            number = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return int(QualifierUnit.NONE)
        units = {int(u) for u in QualifierUnit}
        return number if number in units else int(QualifierUnit.NONE)


class RuleExtract(BaseModel):
    """The extractor's whole reply for one section."""

    statements: list[RuleStatement] = []
