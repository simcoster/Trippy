"""Pydantic schemas and field validation for amenity extraction."""

from __future__ import annotations

import re
from datetime import time
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

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

_TIME_RE = re.compile(r"^(\d{1,2})(?::(\d{2}))?(?::(\d{2}))?$")


def parse_time_of_day(value: Any) -> time | None:
    if value is None or value == "":
        return None
    if isinstance(value, time):
        return value
    if isinstance(value, (int, float)):
        hour = int(value)
        if 0 <= hour <= 23:
            return time(hour=hour, minute=0)
        raise ValueError(f"invalid hour: {value}")
    text = str(value).strip().lower().replace(".", ":")
    match = _TIME_RE.match(text)
    if not match:
        raise ValueError(f"invalid time: {value!r}")
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    second = int(match.group(3) or 0)
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        raise ValueError(f"invalid time: {value!r}")
    return time(hour=hour, minute=minute, second=second)


class PolicyRules(BaseModel):
    min_nights: int | None = None
    min_weekend_nights: int | None = None
    min_holiday_nights: int | None = None
    pets_allowed: bool | None = None

    @field_validator(
        "min_nights",
        "min_weekend_nights",
        "min_holiday_nights",
    )
    @classmethod
    def _non_negative(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("nights must be >= 0")
        return v

    def as_db_dict(self) -> dict[str, Any] | None:
        data = self.model_dump(exclude_none=True)
        return data or None


class AccommodationExtract(BaseModel):
    accommodation_category: AccommodationCategory = "other"
    double_bed: int = 0
    single_bed: int = 0
    room_count: int = 1
    max_people: int | None = None
    check_in_time: time | None = None
    check_out_time: time | None = None
    policy_rules: PolicyRules | None = None
    amenities: list[str] = Field(default_factory=list)
    not_included: list[str] = Field(default_factory=list)

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

    @field_validator("check_in_time", "check_out_time", mode="before")
    @classmethod
    def _coerce_time(cls, v: Any) -> time | None:
        return parse_time_of_day(v)

    @field_validator("amenities", "not_included", mode="before")
    @classmethod
    def _string_list(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("expected a list")
        return [str(a).strip() for a in v if str(a).strip()]

    @field_validator("policy_rules", mode="before")
    @classmethod
    def _empty_policy_to_none(cls, v: Any) -> Any:
        if v is None or v == {} or v == "":
            return None
        return v

    @model_validator(mode="after")
    def _ensure_category_amenity(self) -> AccommodationExtract:
        category = self.accommodation_category
        amenities = [a for a in self.amenities if a != category]
        amenities.insert(0, category)
        self.amenities = amenities
        return self

    def as_details_dict(self) -> dict[str, Any]:
        return {
            "accommodation_category": self.accommodation_category,
            "double_bed": self.double_bed,
            "single_bed": self.single_bed,
            "room_count": self.room_count,
            "max_people": self.max_people,
            "check_in_time": self.check_in_time,
            "check_out_time": self.check_out_time,
            "policy_rules": (
                self.policy_rules.as_db_dict() if self.policy_rules else None
            ),
            "amenities": list(self.amenities),
            "not_included": list(self.not_included),
        }
