"""Gold scoring for planner eval cases (no LLM)."""

from __future__ import annotations

from source.eval.score import score_case


def test_must_include_and_exclude_sites():
    planner = {
        "fits": [
            {"campsite_id": 37, "accommodation_type": "אוהל"},
            {"campsite_id": 38, "accommodation_type": "אוהל"},
        ]
    }
    extract = {"date": {"start": "2026-09-17", "end": "2026-09-18"}}
    out = score_case(
        {
            "date": {"start": "2026-09-17", "end": "2026-09-18"},
            "must_include_sites": [37, 38],
            "must_exclude_sites": [14],
        },
        extract,
        planner,
    )
    assert out["ok"] is True


def test_missing_site_fails():
    out = score_case(
        {"must_include_sites": [37, 38]},
        {},
        {"fits": [{"campsite_id": 37}]},
    )
    assert out["ok"] is False
    assert "missing site 38" in out["failures"]


def test_skipped_no_date():
    out = score_case(
        {"current_planner": "skipped_no_date"},
        {"date": None},
        {"fits": [], "skipped": "no_date"},
    )
    assert out["ok"] is True


def test_why_fridge_and_quoted_price():
    planner = {
        "fits": [
            {
                "campsite_id": 37,
                "accommodation_type": "לינת שטח באוהלים פרטיים",
                "price_per_night": 228.0,
                "why": [{"query": "fridge", "site_amenity": "refrigerator"}],
            }
        ]
    }
    out = score_case(
        {
            "must_include_sites": [37],
            "why_fridge": "site_amenity refrigerator",
            "quoted_price_per_night": {"37": 228},
        },
        {},
        planner,
    )
    assert out["ok"] is True


def test_exclude_hut_type():
    out = score_case(
        {"must_exclude_types": ["חושה"]},
        {},
        {
            "fits": [
                {
                    "campsite_id": 38,
                    "accommodation_type": "חושה עם מזגן שירותים ומקלחת",
                }
            ]
        },
    )
    assert out["ok"] is False
    assert any("unexpected type" in f for f in out["failures"])
