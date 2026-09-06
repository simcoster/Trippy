"""Live scraper OpenAI clients are off unless the test is marked `llm`."""

from __future__ import annotations

import pytest

from source.scraper.amenity_enrichment.llm import (
    LiveLlmDisabled,
    make_nebius_openai_client,
)


def test_requesting_a_nebius_client_raises_without_the_llm_mark():
    with pytest.raises(LiveLlmDisabled):
        make_nebius_openai_client()
