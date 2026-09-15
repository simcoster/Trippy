"""Shared quote inputs. Callers pass this record; generated functions see kwargs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class QuoteParams:
    lodging: str
    adults_num: int = 1
    child_ages: tuple[int, ...] = ()
    is_matmon_sub: bool = False
    is_soldier: bool = False
    is_active_reserve: bool = False
    is_senior: bool = False
    is_student: bool = False
    is_disabled_idf: bool = False
    is_group: bool = False
    is_weekend_or_holiday: bool = False
    planned_entry_time: str | None = None
    planned_exit_time: str | None = None

    def as_call_kwargs(self) -> dict[str, Any]:
        return {
            "lodging": self.lodging,
            "adults_num": int(self.adults_num),
            "child_ages": tuple(int(age) for age in self.child_ages),
            "is_matmon_sub": bool(self.is_matmon_sub),
            "is_soldier": bool(self.is_soldier),
            "is_active_reserve": bool(self.is_active_reserve),
            "is_senior": bool(self.is_senior),
            "is_student": bool(self.is_student),
            "is_disabled_idf": bool(self.is_disabled_idf),
            "is_group": bool(self.is_group),
            "is_weekend_or_holiday": bool(self.is_weekend_or_holiday),
            "planned_entry_time": self.planned_entry_time,
            "planned_exit_time": self.planned_exit_time,
        }

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["child_ages"] = list(self.child_ages)
        return payload

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> QuoteParams:
        ages_raw = data.get("child_ages") or ()
        ages = tuple(int(age) for age in ages_raw)
        return cls(
            lodging=str(data.get("lodging") or ""),
            adults_num=int(data.get("adults_num") or 0),
            child_ages=ages,
            is_matmon_sub=bool(data.get("is_matmon_sub") or False),
            is_soldier=bool(data.get("is_soldier") or False),
            is_active_reserve=bool(data.get("is_active_reserve") or False),
            is_senior=bool(data.get("is_senior") or False),
            is_student=bool(data.get("is_student") or False),
            is_disabled_idf=bool(data.get("is_disabled_idf") or False),
            is_group=bool(data.get("is_group") or False),
            is_weekend_or_holiday=bool(data.get("is_weekend_or_holiday") or False),
            planned_entry_time=_opt_str(data.get("planned_entry_time")),
            planned_exit_time=_opt_str(data.get("planned_exit_time")),
        )


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class QuoteResult:
    price: float
    explanation: str

    @classmethod
    def from_raw(cls, raw: Any) -> QuoteResult:
        if isinstance(raw, QuoteResult):
            return raw
        if isinstance(raw, dict):
            price = raw.get("price")
            explanation = raw.get("explanation")
            if price is None:
                raise ValueError("quote result dict needs price")
            return cls(price=float(price), explanation=str(explanation or ""))
        if isinstance(raw, (tuple, list)) and len(raw) >= 2:
            return cls(price=float(raw[0]), explanation=str(raw[1] or ""))
        raise ValueError("quote() must return (price, explanation)")
