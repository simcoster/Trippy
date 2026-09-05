"""`הונגשו: חניה, שירותים, מקלחות` states that these are accessible.

The extractor used to drop the property and emit the bare nouns, which then
alias-hit the counted amenities and were refused at the upsert (experiments.md
2026-09-04 §11). The prompt now carries the rule with a worked example; the
no-token test pins the prompt, the `llm` test pins the behaviour on the two
Akhziv clauses.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from db.models import SubjectCategory
from source.scraper.rules_ingest.llm import SYSTEM_PROMPT, RuleExtractorLLMClient

load_dotenv()

CLAUSES = [
    "הונגשו בחניון הלילה: חניה, שירותים, מקלחות, פינת פיקניק ושביל לאזור הקמת האוהלים",
    "הונגשו בחניון הלילה: חניה, שירותים, מקלחות, פינת פיקניק, שתי חושות, שבילים",
]


def test_the_prompt_names_the_accessibility_rule_with_an_example():
    assert "הונגשו X, Y, Z" in SYSTEM_PROMPT
    assert "-> accessible_toilets      / amenity / true / null / none" in SYSTEM_PROMPT


@pytest.fixture(scope="module")
def extractor() -> RuleExtractorLLMClient:
    if not os.environ.get("NEBIUS_API_KEY"):
        pytest.skip("NEBIUS_API_KEY required")
    return RuleExtractorLLMClient()


@pytest.mark.llm
@pytest.mark.parametrize("clause", CLAUSES)
def test_everything_made_accessible_is_named_accessible(extractor, clause):
    statements = extractor.extract(clause, section_title="נגישות").statements
    names = [s.subject for s in statements]
    assert names, "nothing extracted"
    bare = [n for n in names if "accessib" not in n]
    assert not bare, f"property dropped from: {bare}"
    assert all(s.category == int(SubjectCategory.AMENITY) for s in statements), names
    assert all(s.polarity is True for s in statements), names
    # The three things every clause lists.
    for thing in ("parking", "toilets", "showers"):
        assert any(thing in n for n in names), (thing, names)


def test_the_prompt_says_what_a_hut_is():
    """`שתי חושות` was read as a fountain and as senses; the glossary names the hut."""
    assert "חושה (plural חושות) is a hut" in SYSTEM_PROMPT


@pytest.mark.llm
def test_two_huts_are_huts_not_fountains(extractor):
    statements = extractor.extract(CLAUSES[1], section_title="נגישות").statements
    names = [s.subject for s in statements]
    assert not [n for n in names if "fountain" in n or "sensory" in n], names
    assert any("hut" in n for n in names), names
