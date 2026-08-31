"""LLM amenity extraction: policy_rules and room_count from Hebrew tooltips."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

# populate_availability imports amenity_enrichment from source/scraper cwd.
_SCRAPER_DIR = Path(__file__).resolve().parents[2] / "scraper"
sys.path.insert(0, str(_SCRAPER_DIR))

from amenity_enrichment import ExtractorLLMClient  # noqa: E402

pytestmark = pytest.mark.llm

HUSHA_ACCESSIBLE_TOOLTIP = (
    "בכל חושה: 4 מיטות, מזרנים, תאורה, מיני מקרר, מאוורר. "
    "עד 4 לנים בחושה. יש להצטייד במצעים, מגבות, כריות ושמיכות. "
    "דרך נגישה לחושה, דלת כניסה רחבה ושולחן מונגש צמוד לחושה. "
    "מותנה במינימום 2 לילות בסופי שבוע ובחגים."
)

DOUBLE_HUSHA_TOOLTIP = (
    "שתי חושות מחוברות עם דלת מקשרת שבכל חדר: 4 מיטות, 4 מזרנים, "
    "מיני מקרר ומאוורר. עד 8 לנים בחושה. יש להצטייד במצעים, מגבות, "
    "שמיכות וכריות. מותנה במינימום 2 לילות בסופי שבוע ובחגים."
)


@pytest.fixture(scope="module")
def extractor():
    if not (os.environ.get("NEBIUS_API_KEY") or os.environ.get("NEBULA_API_KEY")):
        pytest.skip("NEBIUS_API_KEY (or NEBULA_API_KEY) required")
    return ExtractorLLMClient()


def test_extract_min_weekend_and_holiday_nights_policy(extractor):
    details = extractor.extract(
        HUSHA_ACCESSIBLE_TOOLTIP,
        type_name="חושה נגישה",
    )
    assert details.get("policy_rules") == {
        "min_weekend_nights": 2,
        "min_holiday_nights": 2,
    }
    assert details.get("room_count") == 1


def test_extract_connected_hushas_room_count(extractor):
    details = extractor.extract(
        DOUBLE_HUSHA_TOOLTIP,
        type_name="חושה כפולה מספר 7-8",
    )
    assert details.get("room_count") == 2


def _amenity_blob(amenities: list) -> str:
    return " ".join(str(a).lower().replace("_", " ") for a in amenities)


def test_extract_named_place_ramon_crater(extractor):
    """מכתש רמון → keep the place and add crater + desert types."""
    details = extractor.extract(
        "חניון אוהלים ליד מכתש רמון. נוף מדברי, שקט בלילה.",
        type_name="חניון אוהלים",
    )
    blob = _amenity_blob(details.get("amenities") or [])
    print("ramon amenities:", details.get("amenities"))
    assert "ramon" in blob or "רמון" in blob
    assert "crater" in blob or "makhtesh" in blob or "מכתש" in blob
    assert "desert" in blob


def test_extract_named_place_kineret(extractor):
    """כנרת → keep the lake name and add lake / body of water."""
    details = extractor.extract(
        "אתר קמפינג על שפת הכנרת. גישה למים, מדשאה.",
        type_name="מתחם אוהלים",
    )
    blob = _amenity_blob(details.get("amenities") or [])
    print("kineret amenities:", details.get("amenities"))
    assert "kineret" in blob or "kinneret" in blob or "galilee" in blob or "כנרת" in blob
    assert "lake" in blob or "body of water" in blob


def test_extract_named_place_eilat_beach(extractor):
    """חוף אילת — keep the place and beach, without inventing nearby seas."""
    details = extractor.extract(
        "לינה ליד חוף אילת.",
        type_name="אוהל",
    )
    blob = _amenity_blob(details.get("amenities") or [])
    print("eilat amenities:", details.get("amenities"))
    assert "eilat" in blob or "אילת" in blob
    assert "beach" in blob or "חוף" in blob
