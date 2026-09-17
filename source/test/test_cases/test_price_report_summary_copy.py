"""Actions Summary copy of scrape-prices report.md."""

import os
from datetime import datetime
from types import SimpleNamespace

from source.scraper.amenity_enrichment.llm import LlmUsage
from source.scraper.info_site.price_report import REPORT_COPY_ENV, write_run_report


def test_write_run_report_copies_to_prices_report_path(tmp_path, monkeypatch):
    copy = tmp_path / "prices.md"
    monkeypatch.setenv(REPORT_COPY_ENV, str(copy))
    usage = LlmUsage()
    usage.add_chat(
        SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        role="price_function_compile",
        model="org/Qwen3-235B",
    )
    path = write_run_report(
        [],
        usage,
        started_at=datetime(2026, 9, 17, 12, 0, 0),
        seconds=1,
        directory=tmp_path / "run",
        name="report.md",
    )
    assert path.read_text(encoding="utf-8") == copy.read_text(encoding="utf-8")
    assert os.environ[REPORT_COPY_ENV] == str(copy)
