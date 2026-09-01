"""Live 30B visit gate against visit_gate.json (relevant true/false)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from source.scraper.amenity_enrichment.llm import make_nebius_openai_client
from source.scraper.populate_reviews_and_claims import (
    judge_personal_visit,
    normalize_review_dict,
)

load_dotenv()

pytestmark = pytest.mark.llm

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "reviews" / "visit_gate.json"
)
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["reviews"]


@pytest.fixture(scope="module")
def nebius_client():
    if not os.environ.get("NEBIUS_API_KEY"):
        pytest.skip("NEBIUS_API_KEY required")
    return make_nebius_openai_client()


@pytest.mark.parametrize(
    "raw",
    CASES,
    ids=[str(item.get("id") or i) for i, item in enumerate(CASES)],
)
def test_visit_gate_matches_relevant(nebius_client, raw, record_property):
    place = str(raw["name"])
    expected = bool(raw["relevant"])
    review = normalize_review_dict(raw)
    personal, note = judge_personal_visit(nebius_client, review, place=place)
    record_property("skip_note", note or "")
    print(f"{raw.get('id')} relevant={expected} personal={personal} note={note}")
    assert personal is expected, (
        f"id={raw.get('id')} expected relevant={expected} "
        f"got personal={personal} note={note}"
    )
