"""Party size is scored even when the planner skips for no date."""

from source.eval.score import score_case


def test_skipped_no_date_still_checks_party_size():
    out = score_case(
        {"current_planner": "skipped_no_date", "party_size": 2},
        {
            "numeric_constraints": [
                {"field": "party_size", "operator": ">=", "value": 2}
            ]
        },
        {"fits": [], "skipped": "no_date"},
    )
    assert out["ok"] is True


def test_skipped_no_date_fails_when_party_missing():
    out = score_case(
        {"current_planner": "skipped_no_date", "party_size": 2},
        {},
        {"fits": [], "skipped": "no_date"},
    )
    assert out["ok"] is False
    assert "party_size None != 2" in out["failures"]
