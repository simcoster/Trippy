"""Quote one night from published list_prices rows."""

from __future__ import annotations

from typing import Protocol

from .schemas import GuestType, RatePeriod


class RateRow(Protocol):
    guest_type: GuestType
    rate_period: RatePeriod
    price: float


def _period_rank(row_period: RatePeriod, wanted: RatePeriod) -> int | None:
    if row_period == wanted:
        return 0
    if row_period == "any":
        return 1
    return None


def _pick_rate(
    rates: list[RateRow],
    *,
    guest_type: GuestType,
    rate_period: RatePeriod,
) -> RateRow | None:
    ranked: list[tuple[int, RateRow]] = []
    for row in rates:
        if row.guest_type != guest_type:
            continue
        rank = _period_rank(row.rate_period, rate_period)
        if rank is None:
            continue
        ranked.append((rank, row))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    return ranked[0][1]


def quote_night(
    rates: list[RateRow],
    *,
    adults: int,
    children: int = 0,
    rate_period: RatePeriod,
) -> float:
    """Published lodging total for one night.

    ``guest_type="any"`` is per unit (bungalow, room). Adult/child rows are
    per person. Prefer an exact weekday/weekend row over ``rate_period="any"``.
    """
    if adults < 0 or children < 0:
        raise ValueError("adults and children must be >= 0")
    if adults == 0 and children == 0:
        return 0.0

    unit = _pick_rate(rates, guest_type="any", rate_period=rate_period)
    if unit is not None:
        return float(unit.price)

    total = 0.0
    if adults:
        adult = _pick_rate(rates, guest_type="adult", rate_period=rate_period)
        if adult is None:
            raise ValueError(f"no adult rate for {rate_period}")
        total += float(adult.price) * adults
    if children:
        child = _pick_rate(rates, guest_type="child", rate_period=rate_period)
        if child is None:
            raise ValueError(f"no child rate for {rate_period}")
        total += float(child.price) * children
    return total
