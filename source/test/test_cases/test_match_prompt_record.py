"""A flagged match keeps the prompt that produced it.

The run report used to print the answer -- `UNCERTAIN 0.60: 'x' -> 'y'` -- and
nothing else, so a wrong pick could not be attributed: the prompt may not have
told the model what it needed, or the model may have ignored what it was told.
Khan Be'erot is the case that forced this. The panel lists `חדרי צוות חדרים 3-4`
and the rate card says `(חדרים 3 ו- 4)`, the room numbers matching verbatim, and
the model picked `חדר מספר 1-2` anyway. Reading the exact prompt is the only way
to tell whether the numbers were even legible to it.

No database and no network: the record is built in `pick_name`, so a stubbed
client exercises the real code path.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from source.scraper.info_site.match_listing import InfoWebsiteNameMatcher
from source.scraper.info_site.scrape import match_verdict

CANDIDATES = ["חדר צוות מאובזר כפול חדר מספר 1-2", "חדרי צוות חדרים 3-4"]


def matcher_returning(reply: str) -> InfoWebsiteNameMatcher:
    """A real matcher over a stubbed client, so `pick_name` records for real."""
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = reply
    response.usage = None
    client.chat.completions.create.return_value = response
    return InfoWebsiteNameMatcher(client=client)


def test_the_call_keeps_the_prompt_verbatim():
    matcher = matcher_returning('{"name": "חדרי צוות חדרים 3-4", "confidence": 0.6}')

    matcher.pick_name("חדר צוות קטן אמצע שבוע (חדרים 3 ו- 4)", CANDIDATES)

    assert len(matcher.calls) == 1
    call = matcher.calls[0]
    # Brackets are stripped on the way in: on the 30B the answer turned on
    # their order, and they are measured inert on the 235B (experiments.md §20).
    assert "Name: חדר צוות קטן אמצע שבוע חדרים 3 ו- 4" in call.user
    assert "(" not in call.user.splitlines()[0]
    # Numbered exactly as the model saw them, in the order they were given.
    assert f"1. {CANDIDATES[0]}" in call.user
    assert f"2. {CANDIDATES[1]}" in call.user
    assert "Output valid JSON only" in call.system
    assert call.reply == '{"name": "חדרי צוות חדרים 3-4", "confidence": 0.6}'
    assert (call.picked, call.confidence) == ("חדרי צוות חדרים 3-4", 0.6)


def test_a_low_confidence_answer_reads_as_uncertain():
    matcher = matcher_returning('{"name": "%s", "confidence": 0.6}' % CANDIDATES[0])
    matcher.pick_name("חדר צוות גדול", CANDIDATES)

    assert match_verdict(matcher.calls[0]) == "uncertain"


def test_a_refusal_reads_as_forced_not_uncertain():
    """A null is the model ignoring the prompt, not reporting a poor match."""
    matcher = matcher_returning('{"name": null, "confidence": null}')
    matcher.pick_name("מתחם pitch (עד 4 לנים)", CANDIDATES)

    call = matcher.calls[0]
    assert call.picked is None
    assert match_verdict(call) == "forced"


def test_a_name_that_is_not_a_candidate_is_a_refusal_too():
    """An invented name is recorded as sent, and still counts as no answer."""
    matcher = matcher_returning('{"name": "בונגלו עם מזגן", "confidence": 0.95}')
    matcher.pick_name("חדר צוות", CANDIDATES)

    call = matcher.calls[0]
    assert "בונגלו עם מזגן" in call.reply
    assert call.picked is None
    assert match_verdict(call) == "forced"


def test_a_confident_match_is_not_flagged():
    matcher = matcher_returning('{"name": "%s", "confidence": 0.95}' % CANDIDATES[1])
    matcher.pick_name("חדר צוות קטן (חדרים 3 ו- 4)", CANDIDATES)

    assert match_verdict(matcher.calls[0]) == ""
