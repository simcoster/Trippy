"""Failing persist tests: fees and newsflashes are parsed but not stored yet."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRAPER_DIR = Path(__file__).resolve().parents[2] / "scraper"
sys.path.insert(0, str(_SCRAPER_DIR))

from info_site.classify import classify_row, lodging_rows_to_persist  # noqa: E402
from info_site.newsflashes import (  # noqa: E402
    parse_flashbacks_json,
    persist_flashbacks_to_notices,
)
from info_site.parse import parse_rate_table  # noqa: E402

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "info_site"
_TABLE_HTML = (_FIXTURES / "horashat_tal_table.html").read_text(encoding="utf-8")
_FLASHBACKS = json.loads((_FIXTURES / "flashbacks.json").read_text(encoding="utf-8"))


@pytest.mark.xfail(strict=True, reason="fee rows not stored yet")
def test_late_checkout_fee_is_persisted():
    raw = next(row for row in parse_rate_table(_TABLE_HTML) if row["price"] == 265.0)
    classified = classify_row(raw)
    persisted = lodging_rows_to_persist([classified])
    assert any(row.kind == "fee" and row.price == 265.0 for row in persisted)


@pytest.mark.xfail(strict=True, reason="newsflash scrape not wired yet")
def test_flashback_is_upserted_to_notices():
    items = parse_flashbacks_json(_FLASHBACKS)
    assert items
    persisted = persist_flashbacks_to_notices(
        None,
        site_id=1,
        items=items,
        page_url="https://www.parks.org.il/camping/example/",
    )
    assert persisted
