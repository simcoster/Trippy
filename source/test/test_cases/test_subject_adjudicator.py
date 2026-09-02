"""The adjudicator may only return a name it was actually offered."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from source.scraper.subjects.llm import SubjectAdjudicatorLLMClient
from source.scraper.subjects.schemas import ClassificationPayload


def make_client(content: str) -> SubjectAdjudicatorLLMClient:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=None,
    )
    openai = MagicMock()
    openai.chat.completions.create.return_value = response
    return SubjectAdjudicatorLLMClient(openai)


def test_pick_match_returns_an_offered_candidate():
    client = make_client('{"match": "air_conditioning"}')
    assert (
        client.pick_match("air_conoditioning", ["air_conditioning", "heating"])
        == "air_conditioning"
    )


def test_pick_match_rejects_a_name_that_was_not_offered():
    """A confident hallucination is still not a match."""
    client = make_client('{"match": "air_conditioner_unit"}')
    assert client.pick_match("air_conoditioning", ["air_conditioning"]) is None


@pytest.mark.parametrize("content", ['{"match": null}', '{"match": ""}', "{}"])
def test_pick_match_accepts_an_explicit_no_match(content):
    assert make_client(content).pick_match("thing", ["shower"]) is None


def test_pick_match_with_no_candidates_makes_no_call():
    client = make_client('{"match": "shower"}')
    assert client.pick_match("thing", []) is None
    client.client.chat.completions.create.assert_not_called()


def test_pick_match_tolerates_a_fenced_reply():
    client = make_client('```json\n{"match": "shower"}\n```')
    assert client.pick_match("showers", ["shower"]) == "shower"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("rule", 2), ("rules", 2), ("amenity", 1), ("amenities", 1), (2, 2), (1, 1)],
)
def test_classify_coerces_the_category(raw, expected):
    payload = ClassificationPayload.model_validate(
        {"category": raw, "canonical_name": "dogs_allowed"}
    )
    assert payload.category == expected


@pytest.mark.parametrize("raw", ["nonsense", 7, None, -1])
def test_unknown_category_falls_back_to_amenity(raw):
    payload = ClassificationPayload.model_validate(
        {"category": raw, "canonical_name": "shower"}
    )
    assert payload.category == 1


def test_classify_rejects_an_empty_name():
    with pytest.raises(ValueError):
        ClassificationPayload.model_validate({"category": 1, "canonical_name": "  "})
