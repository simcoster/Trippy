"""Every fitting night is shown, and the reply ends with a filter funnel."""

from types import SimpleNamespace

from source.agent.recommender.recommend import (
    Recommendation,
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


def test_contiguous_one_nights_render_as_one_span():
    text = render_recommendations([_rec(dates=_one_nights())], query="a pool")
    assert text.startswith("24.5–29.5, one night each")
    assert "24.5 https://book/24" in text
    assert "29.5 https://book/29" in text


def test_hebrew_span_says_each_night():
    text = render_recommendations([_rec(dates=_one_nights())], query="בריכה")
    assert text.startswith("24.5–29.5, כל לילה בנפרד")


def test_gapped_nights_stay_listed():
    dates = (
        StayWindow("2026-05-24", "2026-05-25", ""),
        StayWindow("2026-05-27", "2026-05-28", ""),
    )
    text = render_recommendations([_rec(dates=dates)], query="pool")
    assert text.startswith("24.5–25.5, 27.5–28.5")


def test_why_not_follows_the_options():
    steps = [
        {"stage": "availability", "count": 11},
        {"stage": "price", "count": 8},
        {"stage": "missing", "count": 6, "query": "pool"},
    ]
    text = render_recommendations([_rec()], query="a pool", why_not=steps)
    assert "Why not" in text
    assert "11 campsites have availability" in text
    assert "→ 8 are in the price range" in text
    assert "→ 6 don't have pool like you requested" in text
    assert text.index("1. אכזיב") < text.index("Why not")


def test_hebrew_why_not():
    steps = [
        {"stage": "availability", "count": 11},
        {"stage": "price", "count": 8},
        {"stage": "missing", "count": 6, "query": "בריכה"},
    ]
    text = render_recommendations([], empty="אין מקום", query="בריכה", why_not=steps)
    assert "אין מקום" in text
    assert "למה לא" in text
    assert "11 אתרי קמפינג עם זמינות" in text
    assert "→ 8 בטווח המחיר" in text
    assert "→ 6 בלי בריכה כמו שביקשת" in text
    assert text.index("אין מקום") < text.index("למה לא")


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
                {"stage": "availability", "count": 4},
                {"stage": "missing", "count": 2, "query": "pool"},
            ],
            "rejected": [{"campsite_id": 9}],
        },
        chat=_Chat(),
    )
    assert result.text.startswith("24.5–25.5, one night each")
    assert "Why not" in result.text
    assert "→ 2 don't have pool like you requested" in result.text
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
