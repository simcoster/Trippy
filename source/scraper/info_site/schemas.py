"""Pydantic models for info-site rate-card rows."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

GuestType = Literal["adult", "child", "any"]
RatePeriod = Literal["weekday", "weekend_holiday", "any"]
RowKind = Literal["lodging", "fee"]
RateClass = Literal["regular"]


class RawPriceRow(BaseModel):
    raw_label: str
    price: float
    notes: str | None = None


class ClassifiedPriceRow(RawPriceRow):
    accommodation_type: str
    guest_type: GuestType
    rate_period: RatePeriod
    kind: RowKind
    rate_class: RateClass = "regular"


class ClassificationPayload(BaseModel):
    accommodation_type: str = Field(min_length=1)
    guest_type: GuestType
    rate_period: RatePeriod
    kind: RowKind
