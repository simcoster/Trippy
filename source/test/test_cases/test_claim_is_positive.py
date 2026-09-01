"""Splitter polarity strings → claims.is_positive."""

from __future__ import annotations

from source.scraper.populate_reviews_and_claims import (
    filter_claims,
    is_positive_from_polarity,
)


def test_is_positive_from_polarity():
    assert is_positive_from_polarity("positive") is True
    assert is_positive_from_polarity("NEGATIVE") is False
    assert is_positive_from_polarity("neutral") is None
    assert is_positive_from_polarity(None) is None
    assert is_positive_from_polarity(True) is True
    assert is_positive_from_polarity(False) is False


def test_filter_claims_sets_is_positive():
    kept = filter_claims(
        [
            {
                "text_en": "There is shade.",
                "polarity": "positive",
                "evidence_span": "יש צל",
                "confidence": 0.9,
            },
            {
                "text_en": "Pets are not allowed.",
                "polarity": "negative",
                "evidence_span": "אסורה כניסה",
                "confidence": 0.9,
            },
            {
                "text_en": "A reservation is required.",
                "polarity": "neutral",
                "evidence_span": "נדרש לבצע הזמנה",
                "confidence": 0.9,
            },
        ]
    )
    assert [row["is_positive"] for row in kept] == [True, False, None]
    assert [row["polarity"] for row in kept] == [
        "positive",
        "negative",
        "neutral",
    ]
