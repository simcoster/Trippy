"""Each scrape-prices run gets its own reports/scrape_prices/<timestamp>/ folder."""

from datetime import datetime
from types import SimpleNamespace

from source.scraper.amenity_enrichment.llm import LlmUsage
from source.scraper.info_site.price_report import run_folder, write_run_report


def test_run_folder_holds_dumps_and_report_md(tmp_path):
    started = datetime(2026, 9, 17, 10, 40, 0)
    folder = run_folder(started, directory=tmp_path)
    assert folder == tmp_path / "2026-09-17_104000"
    usage = LlmUsage()
    usage.add_chat(
        SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        role="price_function_compile",
        model="org/Qwen3-235B",
    )
    path = write_run_report(
        [], usage, started_at=started, seconds=1, directory=folder, name="report.md"
    )
    assert path == folder / "report.md"
    assert path.read_text(encoding="utf-8").startswith("# scrape-prices")
