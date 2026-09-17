"""scrape-prices Markdown report and versioned fail dumps."""

from datetime import datetime
from types import SimpleNamespace

from source.scraper.amenity_enrichment.llm import LlmUsage
from source.scraper.info_site.price_report import (
    PriceFunctionRun,
    render_run_report,
    write_run_report,
)
from source.scraper.info_site.scrape import next_fail_stem


def test_next_fail_stem_skips_latest_and_other_ids(tmp_path):
    assert next_fail_stem(tmp_path, 13) == "13_v1"
    (tmp_path / "13.py").write_text("latest")
    assert next_fail_stem(tmp_path, 13) == "13_v1"
    (tmp_path / "13_v1.py").write_text("first fail")
    assert next_fail_stem(tmp_path, 13) == "13_v2"
    (tmp_path / "11_v1.py").write_text("other site")
    assert next_fail_stem(tmp_path, 1) == "1_v1"


def test_report_lists_failing_tests_and_versioned_dumps(tmp_path):
    usage = LlmUsage()
    usage.add_chat(
        SimpleNamespace(prompt_tokens=100, completion_tokens=20),
        role="price_function_compile",
        model="org/Qwen3-235B",
    )
    usage.add_chat(
        SimpleNamespace(prompt_tokens=50, completion_tokens=10),
        role="price_function_compile_fix",
        model="org/Qwen3-235B",
    )
    runs = [
        PriceFunctionRun(
            site_id=13,
            site_name="הבשור",
            url="https://example/besor",
            outcome="ast_failed",
            n_gold=20,
            retry="fix",
            failures=["line 147: string membership ':'"],
            fail_stems=["13_v1", "13_v2"],
        ),
        PriceFunctionRun(
            site_id=2,
            site_name="חורשת טל",
            outcome="stored",
            n_gold=17,
            store_status="updated",
            digest="abcdef1234567890",
        ),
        PriceFunctionRun(
            site_id=9,
            site_name="skipped park",
            outcome="skipped",
            skip_reason="no gold tests",
        ),
    ]
    started = datetime(2026, 9, 17, 6, 20, 0)
    text = render_run_report(runs, usage, started_at=started, seconds=90)
    assert "failed: 1" in text
    assert "stored: 1 (1 updated)" in text
    assert "skipped: 1" in text
    assert "הבשור" in text
    assert "line 147: string membership ':'" in text
    assert "`13_v1.py`" in text
    assert "`13_v2.py`" in text
    assert "`13_v1.prompt.txt`" in text
    assert "retry: fix" in text
    assert "price_function_compile" in text
    assert "price_function_compile_fix" in text
    assert "1 min 30 s" in text
    assert "skipped park" in text
    assert "`9.py`" not in text
    path = write_run_report(
        runs, usage, started_at=started, seconds=90, directory=tmp_path
    )
    assert path.name == "2026-09-17_062000.md"
    assert path.read_text(encoding="utf-8") == text
