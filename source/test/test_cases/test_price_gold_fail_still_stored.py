"""Gold-fail compiles that passed AST still show as stored in the report."""

from datetime import datetime
from types import SimpleNamespace

from source.scraper.amenity_enrichment.llm import LlmUsage
from source.scraper.info_site.price_report import PriceFunctionRun, render_run_report


def test_gold_failed_run_notes_that_it_was_stored():
    usage = LlmUsage()
    usage.add_chat(
        SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        role="price_function_compile",
        model="org/Qwen3-235B",
    )
    text = render_run_report(
        [
            PriceFunctionRun(
                site_id=17,
                site_name="בארות",
                outcome="gold_failed",
                store_status="updated",
                n_gold=21,
                digest="abcdef123456",
                failures=["חדרי צוות weekday: got 480 expected 430"],
            )
        ],
        usage,
        started_at=datetime(2026, 9, 17, 13, 0, 0),
        seconds=10,
    )
    assert "gold failed (stored updated)" in text
    assert "failed: 1" in text
    assert "`17.py`" in text
