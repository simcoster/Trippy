"""Pydantic payloads for the subject resolver's two small-LLM calls."""

from __future__ import annotations

from pydantic import BaseModel, field_validator

from db.models import SubjectCategory


class AdjudicationPayload(BaseModel):
    """Which of the nearest candidates, if any, is the same subject."""

    match: str | None = None

    @field_validator("match", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class ClassificationPayload(BaseModel):
    """A brand-new subject: what kind it is, and what to call it."""

    category: int
    canonical_name: str

    @field_validator("category", mode="before")
    @classmethod
    def _coerce_category(cls, value: object) -> int:
        if isinstance(value, str):
            text = value.strip().casefold()
            if text.startswith("rule"):
                return int(SubjectCategory.RULE)
            if text.startswith("amenit"):
                return int(SubjectCategory.AMENITY)
        try:
            number = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return int(SubjectCategory.AMENITY)
        # Anything unrecognised is the safer, more common case.
        return number if number in (1, 2) else int(SubjectCategory.AMENITY)

    @field_validator("canonical_name")
    @classmethod
    def _require_name(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("canonical_name must not be empty")
        return text
