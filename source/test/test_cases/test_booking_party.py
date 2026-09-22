"""A 2+2 party is two adults and two children on the booking link."""

from source.agent.booking import RESULTS_PATH, attach_booking_urls, booking_results_url


def test_two_adults_and_two_children_are_not_four_adults():
    fits = [
        {
            "campsite_id": 37,
            "start": "2026-09-24",
            "end": "2026-09-25",
            "dates": [
                {"start": "2026-09-24", "end": "2026-09-25"},
                {"start": "2026-09-25", "end": "2026-09-26"},
            ],
        }
    ]
    attach_booking_urls(
        fits,
        {
            "child_num": 2,
            "numeric_constraints": [
                {"field": "party_size", "operator": ">=", "value": 4}
            ],
        },
        hotel_ids={37: "8_1"},
    )
    url = booking_results_url(
        "8_1",
        "2026-09-24",
        "2026-09-25",
        adults=2,
        children=2,
    )
    assert fits[0]["booking_url"] == url
    assert "ad1=2" in url
    assert "ch1=2" in url
    assert "ad1=4" not in url
    assert fits[0]["dates"][1]["booking_url"].startswith(RESULTS_PATH)
    assert "ch1=2" in fits[0]["dates"][1]["booking_url"]
