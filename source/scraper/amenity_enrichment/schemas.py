"""Pydantic schemas and field validation for unit-detail extraction."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, field_validator

AccommodationCategory = Literal[
    "room",
    "cabin",
    "tent",
    "trailer_parking",
    "tent_pitch",
    "bungalow",
    "dorm",
    "other",
]

ALLOWED_CATEGORIES = frozenset(
    {
        "room",
        "cabin",
        "tent",
        "trailer_parking",
        "tent_pitch",
        "bungalow",
        "dorm",
        "other",
    }
)


class AccommodationExtract(BaseModel):
    """The unit's own structured fields, read from its booking tooltip.

    Amenities and rules are not here: the same text goes through the rules
    extractor (`rules_ingest.units`), which reads them as statements carrying
    the sentence each was read from and writes them to `campsite_rules`. That
    includes check-in/check-out times and minimum-night policies, which used to
    be columns and are facts about the unit like any other.
    """

    accommodation_category: AccommodationCategory = "other"
    double_bed: int = 0
    single_bed: int = 0
    room_count: int = 1
    max_people: int | None = None

    @field_validator("accommodation_category", mode="before")
    @classmethod
    def _normalize_category(cls, v: Any) -> str:
        raw = str(v or "").strip().lower()
        return raw if raw in ALLOWED_CATEGORIES else "other"

    @field_validator("double_bed", "single_bed", mode="before")
    @classmethod
    def _int_or_zero(cls, v: Any) -> int:
        if v is None or v == "":
            return 0
        return int(v)

    @field_validator("room_count", mode="before")
    @classmethod
    def _room_count_default(cls, v: Any) -> int:
        if v is None or v == "":
            return 1
        n = int(v)
        if n < 1:
            raise ValueError("room_count must be >= 1")
        return n

    @field_validator("max_people", mode="before")
    @classmethod
    def _optional_int(cls, v: Any) -> int | None:
        if v is None or v == "":
            return None
        return int(v)

    def as_details_dict(self) -> dict[str, Any]:
        return {
            "accommodation_category": self.accommodation_category,
            "double_bed": self.double_bed,
            "single_bed": self.single_bed,
            "room_count": self.room_count,
            "max_people": self.max_people,
        }
