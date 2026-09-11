"""Planner fits get a BE_Results booking URL; the rec prints the fit's URL."""

from __future__ import annotations

from types import SimpleNamespace

from source.agent.booking import (
    RESULTS_PATH,
    attach_booking_urls,
    booking_results_url,
)
from source.agent.recommender import (
    compact_fit,
    recommend_from_payload,
    render_recommendations,
    validate_recommendations,
)


def test_booking_results_url_matches_availability_search_shape():
    url = booking_results_url(
        "8_1",
        "2026-09-24",
        "2026-09-25",
        adults=2,
    )
    assert url == (
        f"{RESULTS_PATH}?lang=heb&hotel=8_1"
        "&in=2026-09-24&out=2026-09-25&rooms=1&ad1=2&ch1=0&inf1=0"
    )


def test_booking_results_url_none_without_hotel():
    assert booking_results_url(None, "2026-09-24", "2026-09-25") is None
    assert booking_results_url("8_1", None, "2026-09-25") is None


def test_attach_booking_urls_uses_injected_hotel_map():
    fits = [
        {
            "campsite_id": 37,
            "start": "2026-09-24",
            "end": "2026-09-25",
        }
    ]
    attach_booking_urls(
        fits,
        {
            "numeric_constraints": [
                {"field": "party_size", "operator": "=", "value": 2}
            ]
        },
        hotel_ids={37: "8_1"},
    )
    assert fits[0]["booking_hotel_id"] == "8_1"
    assert fits[0]["booking_url"] == booking_results_url(
        "8_1", "2026-09-24", "2026-09-25", adults=2
    )


def test_attach_booking_urls_skips_when_hotel_unknown():
    fits = [{"campsite_id": 99, "start": "2026-09-24", "end": "2026-09-25"}]
    attach_booking_urls(fits, {}, hotel_ids={})
    assert "booking_url" not in fits[0]


def test_validate_uses_fit_booking_url_not_model_string():
    fits = [
        {
            "campsite_id": 37,
            "campsite": "אכזיב צפון",
            "accommodation_type": "אוהל",
            "start": "2026-09-10",
            "end": "2026-09-11",
            "price_per_night": 150,
            "booking_url": "https://secure-hotels.net/INPA/BE_Results.aspx?hotel=8_1",
        }
    ]
    recs = validate_recommendations(
        {
            "recommendations": [
                {
                    "campsite_id": 37,
                    "accommodation_type": "אוהל",
                    "start": "2026-09-10",
                    "end": "2026-09-11",
                    "booking_url": "https://evil.example/nope",
                    "why": "גישה לחוף",
                }
            ]
        },
        fits,
    )
    assert recs[0].booking_url == fits[0]["booking_url"]
    text = render_recommendations(recs)
    assert fits[0]["booking_url"] in text
    assert "evil.example" not in text


def test_compact_fit_keeps_booking_url():
    compact = compact_fit(
        {
            "campsite_id": 37,
            "campsite": "אכזיב צפון",
            "accommodation_type": "אוהל",
            "start": "2026-09-10",
            "end": "2026-09-11",
            "booking_url": "https://secure-hotels.net/INPA/BE_Results.aspx?hotel=8_1",
            "score": 0.1,
        }
    )
    assert compact["booking_url"].endswith("hotel=8_1")
    assert "score" not in compact


def test_recommend_lookup_ignores_emitted_url():
    class _Chat:
        def invoke(self, messages):
            return SimpleNamespace(
                content=(
                    '{"recommendations": [{"campsite_id": 37,'
                    '"accommodation_type": "אוהל",'
                    '"start": "2026-09-10", "end": "2026-09-11",'
                    '"booking_url": "https://invented.example/",'
                    '"why": "גישה לחוף"}], "empty": null}'
                ),
                usage_metadata={"input_tokens": 1, "output_tokens": 1},
            )

    url = booking_results_url("8_1", "2026-09-10", "2026-09-11", adults=2)
    result = recommend_from_payload(
        "ליד הים",
        {
            "constraints": {
                "numeric_constraints": [
                    {"field": "party_size", "operator": "=", "value": 2}
                ]
            },
            "fits": [
                {
                    "campsite_id": 37,
                    "campsite": "אכזיב צפון",
                    "accommodation_type": "אוהל",
                    "start": "2026-09-10",
                    "end": "2026-09-11",
                    "price_per_night": 150,
                    "booking_url": url,
                }
            ],
        },
        chat=_Chat(),
    )
    assert result.recommendations[0].booking_url == url
    assert url in result.text
    assert "invented.example" not in result.text
