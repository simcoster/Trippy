"""Two-stay intro sits above the numbered list; a single stay omits it."""

from types import SimpleNamespace

from source.agent.recommender import (
    Recommendation,
    parse_recommender_payload,
    recommend_from_payload,
    render_recommendations,
)


def _rec(**overrides) -> Recommendation:
    row = {
        "campsite_id": 37,
        "campsite": "אכזיב צפון",
        "accommodation_type": "אוהל",
        "start": "2026-09-10",
        "end": "2026-09-11",
        "price_per_night": 150,
        "why": "יש אוהלים",
        "booking_url": "",
    }
    row.update(overrides)
    return Recommendation(**row)


def _fit(**overrides) -> dict:
    row = {
        "campsite_id": 37,
        "campsite": "אכזיב צפון",
        "accommodation_type": "אוהל",
        "start": "2026-09-10",
        "end": "2026-09-11",
        "price_per_night": 150,
    }
    row.update(overrides)
    return row


def test_parse_recommender_payload_reads_intro():
    parsed = parse_recommender_payload(
        '{"recommendations": [], "intro": "יש כאן שתי אפשרויות",'
        ' "empty": null}'
    )
    assert parsed["intro"] == "יש כאן שתי אפשרויות"


def test_render_two_stays_includes_intro():
    text = render_recommendations(
        [
            _rec(),
            _rec(
                campsite_id=38,
                campsite="אכזיב דרום",
                accommodation_type="חדר צוות",
                why="חדר עץ",
            ),
        ],
        intro="יש כאן שתי אפשרויות: הראשונה לינת שטח באוהלים, השנייה חדר צוות עץ.",
    )
    assert text.startswith("10.9–11.9")
    assert "יש כאן שתי אפשרויות" in text
    assert text.index("יש כאן שתי אפשרויות") < text.index("1. אכזיב צפון")
    assert "2. אכזיב דרום" in text


def test_render_one_stay_drops_intro():
    text = render_recommendations(
        [_rec()],
        intro="יש כאן שתי אפשרויות",
    )
    assert "יש כאן שתי אפשרויות" not in text
    assert "1. אכזיב צפון" in text


def test_recommend_from_payload_renders_intro_for_two_picks():
    class _Chat:
        def invoke(self, messages):
            return SimpleNamespace(
                content=(
                    '{"recommendations": ['
                    '{"campsite_id": 37, "accommodation_type": "אוהל",'
                    '"start": "2026-09-10", "end": "2026-09-11",'
                    '"why": "יש אוהלים"},'
                    '{"campsite_id": 38, "accommodation_type": "חדר צוות",'
                    '"start": "2026-09-10", "end": "2026-09-11",'
                    '"why": "חדר עץ"}],'
                    '"intro": "יש כאן שתי אפשרויות: הראשונה אוהלים, השנייה חדר צוות.",'
                    '"empty": null}'
                ),
                usage_metadata={"input_tokens": 8, "output_tokens": 6},
            )

    result = recommend_from_payload(
        "אוהל או חדר",
        {
            "constraints": {},
            "fits": [
                _fit(),
                _fit(
                    campsite_id=38,
                    campsite="אכזיב דרום",
                    accommodation_type="חדר צוות",
                ),
            ],
        },
        chat=_Chat(),
    )
    assert len(result.recommendations) == 2
    assert result.intro is not None
    assert "יש כאן שתי אפשרויות" in result.intro
    assert "יש כאן שתי אפשרויות" in result.text
    assert result.text.index("יש כאן שתי אפשרויות") < result.text.index(
        "1. אכזיב צפון"
    )
