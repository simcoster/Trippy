"""Compact judge output: claim indices, short reason, map back to memory."""

from source.agent.claim_judge import (
    CLAIM_JUDGE_COMPACT_SUFFIX,
    _relevant_claim_texts,
    judge_compact,
)


def test_judge_compact_default_on(monkeypatch):
    monkeypatch.delenv("TRIPPY_JUDGE_COMPACT", raising=False)
    assert judge_compact() is True


def test_judge_compact_env_on(monkeypatch):
    monkeypatch.setenv("TRIPPY_JUDGE_COMPACT", "1")
    assert judge_compact() is True


def test_compact_suffix_asks_for_indices_and_short_reason():
    assert '"relevant": [int]' in CLAIM_JUDGE_COMPACT_SUFFIX
    assert "4-5" in CLAIM_JUDGE_COMPACT_SUFFIX


def test_relevant_claim_texts_maps_indices():
    rows = [
        {"claim": "Pets are not allowed.", "is_positive": False},
        {"claim": "Campfires are allowed.", "is_positive": True},
        {"claim": "The desert atmosphere is perfect.", "is_positive": True},
    ]
    texts = _relevant_claim_texts(
        {"relevant": [0, 2], "satisfies": True},
        rows,
        compact=True,
    )
    assert texts == [
        "Pets are not allowed.",
        "The desert atmosphere is perfect.",
    ]


def test_relevant_claim_texts_accepts_string_indices():
    rows = [{"claim": "only", "is_positive": True}]
    texts = _relevant_claim_texts(
        {"relevant": ["0"]},
        rows,
        compact=True,
    )
    assert texts == ["only"]


def test_relevant_claim_texts_skips_bad_indices():
    rows = [{"claim": "only", "is_positive": True}]
    texts = _relevant_claim_texts(
        {"relevant": [-1, 0, 9, True]},
        rows,
        compact=True,
    )
    assert texts == ["only"]


def test_relevant_claim_texts_falls_back_to_quoted_strings():
    rows = [{"claim": "Pets are not allowed.", "is_positive": False}]
    texts = _relevant_claim_texts(
        {"relevant_claims": ["Pets are not allowed."]},
        rows,
        compact=True,
    )
    assert texts == ["Pets are not allowed."]


def test_default_mode_keeps_quoted_relevant_claims():
    rows = [{"claim": "ignored", "is_positive": True}]
    texts = _relevant_claim_texts(
        {"relevant_claims": ["The desert atmosphere is perfect."]},
        rows,
        compact=False,
    )
    assert texts == ["The desert atmosphere is perfect."]
