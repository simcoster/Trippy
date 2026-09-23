"""Every fitting night is shown, and the reply ends with a filter funnel."""

from types import SimpleNamespace

from source.agent.recommender.recommend import (
    Recommendation,
    pack_recommender_input,
    recommend_from_payload,
    render_recommendations,
    validate_recommendations,
)
from source.agent.recommender.stay_dates import StayWindow


def _rec(**overrides) -> Recommendation:
    row = {
        "campsite_id": 37,
        "campsite": "אכזיב",
        "accommodation_type": "אוהל",
        "start": "2026-05-24",
        "end": "2026-05-25",
        "price_per_night": 180,
        "why": "יש בריכה",
        "booking_url": "",
        "price_explanation": "",
        "dates": (),
    }
    row.update(overrides)
    return Recommendation(**row)


def _one_nights() -> tuple[StayWindow, ...]:
    return tuple(
        StayWindow(f"2026-05-{day:02d}", f"2026-05-{day + 1:02d}", f"https://book/{day}")
        for day in range(24, 30)
    )


def test_pack_sets_reply_language_from_the_query():
    english = pack_recommender_input("a pool for the kids", {"fits": []})
    hebrew = pack_recommender_input("בריכה לילדים", {"fits": []})
    assert english["reply_language"] == "english"
    assert hebrew["reply_language"] == "hebrew"


def test_contiguous_one_nights_render_as_one_span():
    text = render_recommendations(
        [_rec(dates=_one_nights(), booking_url="https://book/site")],
        query="a pool",
    )
    assert text.startswith("24–29.5")
    assert "one night" not in text
    assert "https://book/site" in text
    assert "https://book/24" not in text
    assert "Adjust the date on the booking page." not in text
    assert text.count("https://book/") == 1


def test_hebrew_span_has_one_booking_link():
    text = render_recommendations(
        [_rec(dates=_one_nights(), booking_url="https://book/site")],
        query="בריכה",
    )
    assert text.startswith("24–29.5")
    assert "שנו את התאריך בעמוד ההזמנה." not in text
    assert text.count("https://book/") == 1


def test_gapped_nights_are_separate_ranges():
    dates = (
        StayWindow("2026-09-22", "2026-09-23", "https://book/22"),
        StayWindow("2026-09-23", "2026-09-24", "https://book/23"),
        StayWindow("2026-09-24", "2026-09-25", "https://book/24"),
        StayWindow("2026-09-26", "2026-09-27", "https://book/26"),
    )
    text = render_recommendations(
        [_rec(dates=dates, booking_url="https://book/site")],
        query="pool",
    )
    assert text.startswith("22–24.9, 26.9")
    assert "22.9–23.9" not in text
    assert text.count("https://") == 1


def test_two_months_keep_a_range_each():
    dates = tuple(
        StayWindow(f"2026-09-{day:02d}", f"2026-09-{day + 1:02d}", "")
        for day in range(21, 29)
    ) + tuple(
        StayWindow(
            f"2026-10-{day:02d}",
            f"2026-10-{day + 1:02d}" if day < 31 else "2026-11-01",
            "",
        )
        for day in range(11, 16)
    )
    text = render_recommendations([_rec(dates=dates)], query="pool")
    assert text.startswith("21–28.9, 11.10–15.10")


def test_why_not_names_the_other_sites():
    steps = [
        {
            "stage": "missing",
            "count": 3,
            "query": "pools",
            "sites": ["site_1", "site_2", "site_3"],
        }
    ]
    text = render_recommendations([_rec()], query="a pool", why_not=steps)
    sentence = (
        "3 campsites don't have an indication of pools "
    )
    assert sentence in text
    assert "Why not" not in text
    assert text.index("1. אכזיב") < text.index(sentence)


def test_hebrew_why_not():
    steps = [
        {
            "stage": "missing",
            "count": 3,
            "query": "בריכות לילדים",
            "sites": ["אתר א", "אתר ב", "אתר ג"],
        }
    ]
    text = render_recommendations([], empty="אין מקום", query="בריכה", why_not=steps)
    sentence = (
        "3 אתרים בלי אינדיקציה על בריכות לילדים "
    )
    assert "אין מקום" in text
    assert sentence in text
    assert "Why not" not in text
    assert text.index("אין מקום") < text.index("3 אתרים")


def test_price_why_not_quotes_the_requested_limit():
    steps = [{"stage": "price", "count": 3, "sites": ["a", "b", "c"]}]
    text = render_recommendations(
        [_rec()],
        query="a pool",
        why_not=steps,
        constraints={
            "numeric_constraints": [
                {"field": "price_per_night", "operator": "<=", "value": 300}
            ]
        },
    )
    assert "3 campsite slots are outside the price range (up to 300 NIS)." in text


def test_why_not_is_rendered_and_kept_out_of_the_model_pack():
    class _Chat:
        def invoke(self, messages):
            body = messages[1].content
            assert "why_not" not in body
            assert "rejected" not in body
            return SimpleNamespace(
                content=(
                    '{"recommendations": [{"campsite_id": 37,'
                    '"accommodation_type": "אוהל",'
                    '"start": "2026-05-24", "end": "2026-05-25",'
                    '"why": "יש בריכה"}], "intro": null, "empty": null}'
                ),
                usage_metadata={"input_tokens": 4, "output_tokens": 4},
            )

    result = recommend_from_payload(
        "a pool",
        {
            "constraints": {},
            "fits": [
                {
                    "campsite_id": 37,
                    "campsite": "אכזיב",
                    "accommodation_type": "אוהל",
                    "start": "2026-05-24",
                    "end": "2026-05-25",
                    "price_per_night": 180,
                    "dates": [
                        {"start": "2026-05-24", "end": "2026-05-25"},
                        {"start": "2026-05-25", "end": "2026-05-26"},
                    ],
                }
            ],
            "why_not": [
                {
                    "stage": "missing",
                    "count": 2,
                    "query": "pool",
                    "sites": ["דרום", "צפון"],
                }
            ],
            "rejected": [{"campsite_id": 9}],
        },
        chat=_Chat(),
    )
    assert result.text.startswith("24–25.5")
    assert (
        "2 campsites [דרום and צפון] don't have an indication of pool."
        in result.text
    )
    assert len(result.recommendations) == 1
    assert len(result.recommendations[0].dates) == 2


def test_one_model_pick_is_filled_out_to_three():
    class _Chat:
        def invoke(self, messages):
            return SimpleNamespace(
                content=(
                    '{"recommendations": [{"campsite_id": 37,'
                    '"accommodation_type": "אוהל",'
                    '"start": "2026-05-24", "end": "2026-05-25",'
                    '"why": "יש בריכה"}], "intro": null, "empty": null}'
                ),
                usage_metadata={"input_tokens": 4, "output_tokens": 4},
            )

    def _fit(campsite_id: int, name: str) -> dict:
        return {
            "campsite_id": campsite_id,
            "campsite": name,
            "accommodation_type": "אוהל",
            "start": "2026-05-24",
            "end": "2026-05-25",
            "price_per_night": 100 + campsite_id,
        }

    result = recommend_from_payload(
        "a pool",
        {
            "constraints": {},
            "fits": [
                _fit(37, "אכזיב"),
                _fit(38, "דרום"),
                _fit(39, "צפון"),
                _fit(40, "מזרח"),
            ],
        },
        chat=_Chat(),
    )
    assert [rec.campsite_id for rec in result.recommendations] == [37, 38, 39]
    assert result.recommendations[0].why == "יש בריכה"
    assert result.recommendations[1].why == ""
    assert "2. דרום" in result.text
    assert "3. צפון" in result.text
    assert "4. מזרח" not in result.text


def test_validate_keeps_dates_from_the_fit():
    fits = [
        {
            "campsite_id": 37,
            "campsite": "אכזיב",
            "accommodation_type": "אוהל",
            "start": "2026-05-24",
            "end": "2026-05-25",
            "dates": [
                {"start": "2026-05-24", "end": "2026-05-25"},
                {"start": "2026-05-29", "end": "2026-05-30"},
            ],
        }
    ]
    recs = validate_recommendations(
        {
            "recommendations": [
                {
                    "campsite_id": 37,
                    "accommodation_type": "אוהל",
                    "start": "2026-05-24",
                    "end": "2026-05-25",
                    "why": "בריכה",
                }
            ]
        },
        fits,
    )
    assert [window.start for window in recs[0].dates] == [
        "2026-05-24",
        "2026-05-29",
    ]
