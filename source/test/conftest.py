"""Disable live Nebius clients unless the test is marked `llm`."""

from __future__ import annotations

from source.scraper.amenity_enrichment.llm import set_live_llm_allowed


def pytest_configure(config) -> None:
    set_live_llm_allowed(False)


def pytest_runtest_setup(item) -> None:
    set_live_llm_allowed(item.get_closest_marker("llm") is not None)


def pytest_runtest_teardown(item) -> None:
    set_live_llm_allowed(False)
