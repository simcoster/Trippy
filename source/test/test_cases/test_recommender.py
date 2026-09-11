"""Recommender packs planner fits, validates picks, and renders Hebrew."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage, ChatMessage, HumanMessage

from source.agent.prompts import EMPTY_REPLY_FALLBACK
from source.agent.recommender import (
    compact_fit,
    latest_planner_payload,
    pack_recommender_input,
    parse_recommender_payload,
    recommend_from_payload,
    render_recommendations,
    stay_key,
    validate_recommendations,
)
from source.eval.run import write_report


def _fit(**overrides):
    row = {
        "campsite_id": 37,
        "campsite": "אכזיב צפון",
        "accommodation_type": "אוהל",
        "start": "2026-09-10",
        "end": "2026-09-11",
        "price_per_night": 150,
        "score": 0.42,
        "why": [
            {
                "query": "near the sea",
                "site_amenity": "beach_access",
                "distance": 0.1,
            }
        ],
        "review_claims": [
            {
                "query": "near the sea",
                "claim": "Access to the beach is easy.",
                "is_positive": True,
                "date": "2026-08-01",
                "days_ago": 40,
            },
            {
                "query": "near the sea",
                "claim": "The shower block was dirty.",
                "is_positive": False,
            },
        ],
        "retrieved": [
            {
                "query": "near the sea",
                "claims": [{"claim": "Access to the beach is easy.", "is_positive": True}],
                "rules": [
                    {
                        "subject": "beach_access",
                        "polarity": True,
                        "evidence_span": "גישה לחוף",
                    },
                    {
                        "subject": "tent_pitch",
                        "polarity": True,
                        "evidence_span": "לינת שטח",
                    },
                ],
            }
        ],
        "claim_judge": [
            {
                "query": "near the sea",
                "satisfies": True,
                "satisfy_by": "both",
                "relevant_claims": ["Access to the beach is easy."],
                "reason": "beach access listed",
            }
        ],
        "campsite_rules": {
            "near the sea": [
                {
                    "subject": "ignored_when_retrieved",
                    "polarity": True,
                    "evidence_span": "should not appear",
                }
            ]
        },
    }
    row.update(overrides)
    return row


def test_pack_drops_score_and_rejected():
    payload = {
        "constraints": {"date": {"start": "2026-09-10", "end": "2026-09-11"}},
        "fits": [_fit()],
        "rejected": [{"campsite_id": 14}],
        "rejected_count": 3,
        "skipped": None,
    }
    pack = pack_recommender_input("ליד הים", payload)
    assert pack["query"] == "ליד הים"
    assert pack["extract"]["date"]["start"] == "2026-09-10"
    assert "rejected" not in pack
    assert "rejected_count" not in pack
    assert "score" not in pack["fits"][0]
    assert "retrieved" not in pack["fits"][0]


def test_compact_fit_keeps_relevant_claims_and_retrieved_rules():
    compact = compact_fit(_fit())
    assert [c["claim"] for c in compact["review_claims"]] == [
        "Access to the beach is easy.",
        "The shower block was dirty.",
    ]
    assert compact["rules"][0]["query"] == "near the sea"
    subjects = [r["subject"] for r in compact["rules"][0]["rules"]]
    assert subjects == ["beach_access", "tent_pitch"]
    assert "ignored_when_retrieved" not in subjects
    assert compact["claim_judge"][0]["satisfy_by"] == "both"
    assert "distance" not in compact["why"][0]


def test_rules_fall_back_to_campsite_rules():
    compact = compact_fit(_fit(retrieved=[]))
    assert compact["rules"][0]["rules"][0]["subject"] == "ignored_when_retrieved"


def test_validate_drops_invented_and_caps_at_two():
    fits = [
        _fit(),
        _fit(
            campsite_id=38,
            campsite="אכזיב דרום",
            accommodation_type="בקתה",
        ),
        _fit(
            campsite_id=1,
            campsite="חורשת טל",
            accommodation_type="אוהל",
        ),
    ]
    parsed = {
        "recommendations": [
            {
                "campsite_id": 99,
                "accommodation_type": "אוהל",
                "start": "2026-09-10",
                "end": "2026-09-11",
                "why": "invented",
            },
            {
                "campsite_id": 37,
                "accommodation_type": "אוהל",
                "start": "2026-09-10",
                "end": "2026-09-11",
                "why": "ים",
            },
            {
                "campsite_id": 38,
                "accommodation_type": "בקתה",
                "start": "2026-09-10",
                "end": "2026-09-11",
                "why": "שקט",
            },
            {
                "campsite_id": 1,
                "accommodation_type": "אוהל",
                "start": "2026-09-10",
                "end": "2026-09-11",
                "why": "third",
            },
        ]
    }
    recs = validate_recommendations(parsed, fits)
    assert [r.campsite_id for r in recs] == [37, 38]
    assert recs[0].campsite == "אכזיב צפון"
    assert recs[0].why == "ים"


def test_validate_drops_type_mismatch():
    parsed = {
        "recommendations": [
            {
                "campsite_id": 37,
                "accommodation_type": "בקתה",
                "start": "2026-09-10",
                "end": "2026-09-11",
                "why": "wrong type",
            }
        ]
    }
    assert validate_recommendations(parsed, [_fit()]) == []


def test_render_leads_with_day_month_and_prices():
    recs = validate_recommendations(
        {
            "recommendations": [
                {
                    "campsite_id": 37,
                    "accommodation_type": "אוהל",
                    "start": "2026-09-10",
                    "end": "2026-09-11",
                    "why": "קרוב לים",
                }
            ]
        },
        [_fit()],
    )
    text = render_recommendations(recs)
    assert text.startswith("10.9–11.9")
    assert "אכזיב צפון — אוהל (₪150)" in text
    assert "קרוב לים" in text


def test_render_empty_uses_model_empty_then_fallback():
    assert "אין מקום" in render_recommendations([], empty="אין מקום")
    assert render_recommendations([]) == EMPTY_REPLY_FALLBACK


def test_parse_recommender_payload_reads_fenced_json():
    raw = """```json
{"recommendations": [], "empty": "אין תאריך"}
```"""
    parsed = parse_recommender_payload(raw)
    assert parsed["recommendations"] == []
    assert parsed["empty"] == "אין תאריך"


def test_recommend_from_payload_with_fake_chat():
    class _Chat:
        def invoke(self, messages):
            assert "ליד הים" in messages[1].content
            assert "score" not in messages[1].content
            return SimpleNamespace(
                content=(
                    '{"recommendations": [{"campsite_id": 37,'
                    '"accommodation_type": "אוהל",'
                    '"start": "2026-09-10", "end": "2026-09-11",'
                    '"why": "גישה לחוף"}], "empty": null}'
                ),
                usage_metadata={"input_tokens": 10, "output_tokens": 4},
            )

    result = recommend_from_payload(
        "ליד הים",
        {
            "constraints": {},
            "fits": [_fit()],
        },
        chat=_Chat(),
    )
    assert len(result.recommendations) == 1
    assert result.recommendations[0].campsite_id == 37
    assert "גישה לחוף" in result.text
    assert result.empty is None


def test_stay_key_rejects_incomplete_row():
    assert stay_key({"campsite_id": 1, "accommodation_type": "אוהל"}) is None


def test_latest_planner_payload_prefers_chat_fits():
    messages = [
        HumanMessage(content="ליד הים"),
        AIMessage(
            content='{"date": {"start": "2026-09-10", "end": "2026-09-11"}}'
        ),
        ChatMessage(
            content='{"fits": [{"campsite_id": 37}], "constraints": {"x": 1}}',
            role="assistant",
        ),
    ]
    payload = latest_planner_payload(messages)
    assert payload["fits"][0]["campsite_id"] == 37
    assert payload["constraints"]["x"] == 1


def test_report_includes_recommendation_block(tmp_path: Path):
    path = tmp_path / "report.md"
    rows = [
        {
            "id": "E04",
            "difficulty": "easy",
            "query": "ליד הים",
            "seconds": 1.0,
            "score": {
                "ok": True,
                "failures": [],
                "extract_date": {"start": "2026-09-10", "end": "2026-09-11"},
                "fit_sites": [37],
            },
            "extract": {},
            "planner": {"fits": [], "rejected": []},
            "recommend": {
                "recommendations": [
                    {
                        "campsite_id": 37,
                        "campsite": "אכזיב צפון",
                        "accommodation_type": "אוהל",
                        "start": "2026-09-10",
                        "end": "2026-09-11",
                        "price_per_night": 150,
                        "why": "גישה לחוף",
                    }
                ],
                "empty": None,
                "text": "10.9–11.9\n\n1. אכזיב צפון — אוהל (₪150)\n   גישה לחוף",
            },
            "usage": {
                "input_tokens": 1200,
                "output_tokens": 100,
                "by_role": [
                    {
                        "role": "extract",
                        "input_tokens": 200,
                        "output_tokens": 20,
                        "calls": 1,
                    },
                    {
                        "role": "claim_judge",
                        "input_tokens": 800,
                        "output_tokens": 60,
                        "calls": 10,
                    },
                    {
                        "role": "recommend",
                        "input_tokens": 200,
                        "output_tokens": 20,
                        "calls": 1,
                    },
                ],
            },
        }
    ]
    write_report(path, {"id": "planner_v1", "as_of": "2026-09-08"}, rows, 1.0)
    text = path.read_text(encoding="utf-8")
    assert "37 אכזיב צפון אוהל 2026-09-10→2026-09-11 150" in text
    assert "why: גישה לחוף" in text
    assert "| E04 | 1200 | 100 | 200/20 | 800/60×10 | 200/20 |" in text
    assert "| id | in | out | extract | judge | recommend |" in text
    assert "## Recommendations" in text
    assert "| E04 | ליד הים | 1 | אכזיב צפון — אוהל (₪150) | גישה לחוף |" in text
    assert "## Cost" in text
    assert "| id | extract | embed | judge | recommend | total |" in text
    assert "| E04 | $0.0001 |  | $0.0002 | $0.0001 | $0.0003 |" in text
