"""Gold quote cases, keyed by a stable fragment of the campsite URL."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import unquote, urlparse

from ..params import QuoteParams
from .cases import CATALOG


@dataclass(frozen=True)
class GoldCase:
    params: QuoteParams
    expected_price: float
    note: str = ""
    explanation: str = ""


def _url_haystack(url: str) -> str:
    parsed = urlparse(url or "")
    path = unquote(parsed.path or "")
    return f"{path} {url}".casefold()


def load_gold_catalog() -> list[dict]:
    return list(CATALOG)


def gold_for_url(url: str) -> list[GoldCase] | None:
    """Five (or so) cases whose ``match`` fragment sits in *url*, or None."""
    haystack = _url_haystack(url)
    for row in load_gold_catalog():
        match = str(row.get("match") or "").strip()
        if not match or match.casefold() not in haystack:
            continue
        cases: list[GoldCase] = []
        for raw in row.get("cases") or []:
            if not isinstance(raw, dict):
                continue
            params = QuoteParams.from_mapping(raw.get("params") or {})
            cases.append(
                GoldCase(
                    params=params,
                    expected_price=float(raw["expected_price"]),
                    note=str(raw.get("note") or ""),
                    explanation=str(raw.get("explanation") or ""),
                )
            )
        return cases
    return None


def prices_close(got: float, expected: float, *, places: int = 2) -> bool:
    scale = 10**places
    return round(got * scale) == round(expected * scale)
