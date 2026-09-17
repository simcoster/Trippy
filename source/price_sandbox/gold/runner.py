"""Load per-campsite gold JSON and run quote() against those prices."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from source.price_sandbox.execute import eval_quote_inprocess
from source.price_sandbox.params import QuoteParams

SITES_DIR = Path(__file__).resolve().parent / "sites"

# parks.org.il north → south, same order as source/scraper/campsites.json
SITE_FILES = (
    "horashat-tal.json",
    "achziv.json",
    "yehiam.json",
    "nachal-amud.json",
    "yehudiya.json",
    "mishmar-hacarmel.json",
    "kochav-hayarden.json",
    "maayan-harod.json",
    "gan-hashlosha.json",
    "yarkon.json",
    "castel.json",
    "ashkelon.json",
    "habesor.json",
    "masada.json",
    "tel-arad.json",
    "mamshit.json",
    "beerot.json",
    "yotvata.json",
)


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


def site_paths() -> list[Path]:
    """Gold files in catalog order, then any extra JSON sitting in the folder."""
    known = [SITES_DIR / name for name in SITE_FILES]
    missing = [path for path in known if not path.is_file()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(f"gold JSON missing: {names}")
    extras = sorted(
        path
        for path in SITES_DIR.glob("*.json")
        if path.name not in SITE_FILES
    )
    return known + extras


def load_site(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cases_from_doc(doc: dict) -> list[GoldCase]:
    cases: list[GoldCase] = []
    for raw in doc.get("cases") or []:
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


def load_gold_catalog() -> list[dict]:
    return [load_site(path) for path in site_paths()]


def gold_for_url(url: str) -> list[GoldCase] | None:
    """Cases whose ``match`` fragment sits in *url*, or None."""
    haystack = _url_haystack(url)
    for path in site_paths():
        doc = load_site(path)
        match = str(doc.get("match") or "").strip()
        if not match or match.casefold() not in haystack:
            continue
        return cases_from_doc(doc)
    return None


def prices_close(got: float, expected: float, *, places: int = 2) -> bool:
    scale = 10**places
    return round(got * scale) == round(expected * scale)


def run_cases(source: str, cases: list[GoldCase]) -> list[str]:
    """Return human-readable failures; empty means all cases matched."""
    failures: list[str] = []
    for index, case in enumerate(cases, start=1):
        label = case.note or f"case {index}"
        try:
            result = eval_quote_inprocess(source, case.params)
        except Exception as exc:
            failures.append(f"{label}: {exc}")
            continue
        if not prices_close(result.price, case.expected_price):
            detail = case.explanation or ""
            if detail:
                detail = f" ({detail})"
            failures.append(
                f"{label}: got {result.price} expected {case.expected_price}{detail}"
            )
    return failures
