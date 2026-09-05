"""Pydantic payloads for the subject resolver's two small-LLM calls."""

from __future__ import annotations

from pydantic import BaseModel, field_validator

from db.models import SubjectCategory


class AdjudicationPayload(BaseModel):
    """Which of the nearest candidates, if any, is the same subject."""

    match: str | None = None
    # How sure the judge is of THIS answer, 0..1. None when the model omitted it
    # (never seen in 87 probed calls); the caller decides what None means.
    confidence: float | None = None

    @field_validator("match", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, value: object) -> float | None:
        if value is None or value == "":
            return None
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return min(1.0, max(0.0, number))


class ClassificationPayload(BaseModel):
    """A brand-new subject: what kind it is, and what to call it."""

    category: int
    canonical_name: str

    @field_validator("category", mode="before")
    @classmethod
    def _coerce_category(cls, value: object) -> int:
        if isinstance(value, str):
            text = value.strip().casefold()
            if text.startswith("amenit"):
                return int(SubjectCategory.AMENITY)
            if text.startswith(("bool", "rule")):
                return int(SubjectCategory.BOOLEAN_RULE)
            if text.startswith("num"):
                return int(SubjectCategory.NUMERIC_RULE)
        try:
            number = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return int(SubjectCategory.AMENITY)
        # Anything unrecognised is the safer, more common case.
        known = {int(c) for c in SubjectCategory}
        return number if number in known else int(SubjectCategory.AMENITY)

    @field_validator("canonical_name")
    @classmethod
    def _require_name(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("canonical_name must not be empty")
        return text
