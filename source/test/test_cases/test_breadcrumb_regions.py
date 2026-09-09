"""#breadcrumbs region slugs for scrape-info claims."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from source.scraper.info_site.breadcrumbs import (
    BREADCRUMB_NOTES,
    snapshot_breadcrumb_claims,
)
from source.scraper.info_site.parse import parse_breadcrumb_regions

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "info_site"


def _html(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


def test_hurshat_tal_is_north_and_upper_galilee():
    rows = parse_breadcrumb_regions(_html("hurshat_tal_breadcrumbs.html"))
    assert [r["slug"] for r in rows] == ["area:north", "region:upper-galilee"]
    by_slug = {r["slug"]: r["label"] for r in rows}
    assert by_slug["area:north"] == "צפון"
    assert by_slug["region:upper-galilee"] == "גליל עליון"


def test_masada_is_south_and_dead_sea():
    rows = parse_breadcrumb_regions(_html("masada_breadcrumbs.html"))
    assert [r["slug"] for r in rows] == ["area:south", "region:dead-sea"]
    assert rows[1]["label"] == "ארץ ים המלח"


def test_ashkelon_is_center_and_coastal_plain():
    rows = parse_breadcrumb_regions(_html("ashkelon_breadcrumbs.html"))
    assert [r["slug"] for r in rows] == ["area:center", "region:coastal-plain"]


def test_missing_breadcrumbs_are_empty():
    assert parse_breadcrumb_regions("<html><body></body></html>") == []


def test_snapshot_embeds_slugs_and_tags_notes():
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    regions = parse_breadcrumb_regions(_html("hurshat_tal_breadcrumbs.html"))
    saved = snapshot_breadcrumb_claims(
        conn, campsite_id=5, regions=regions, embedder=embedder
    )
    assert saved == 2
    embedder.embed.assert_called_once_with(
        ["area:north", "region:upper-galilee"], usage=None
    )
    sqls = [str(call.args[0]) for call in cur.execute.call_args_list]
    assert any("DELETE FROM claims" in sql for sql in sqls)
    inserts = [
        call.args[1]
        for call in cur.execute.call_args_list
        if "INSERT INTO claims" in str(call.args[0])
    ]
    assert [row["claim"] for row in inserts] == [
        "area:north",
        "region:upper-galilee",
    ]
    assert all(row["notes"] == BREADCRUMB_NOTES for row in inserts)
    assert inserts[0]["evidence_span"] == "צפון"
    assert inserts[0]["campsite_id"] == 5


def test_snapshot_with_no_regions_still_deletes():
    embedder = MagicMock()
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    saved = snapshot_breadcrumb_claims(
        conn, campsite_id=5, regions=[], embedder=embedder
    )
    assert saved == 0
    embedder.embed.assert_not_called()
    assert "DELETE FROM claims" in str(cur.execute.call_args_list[0].args[0])


def test_claim_retrieve_left_joins_reviews():
    from source.agent.search import _CLAIMS_BY_SITE_SQL, _CLAIMS_GLOBAL_SQL

    assert "LEFT JOIN reviews" in _CLAIMS_BY_SITE_SQL
    assert "LEFT JOIN reviews" in _CLAIMS_GLOBAL_SQL
    assert "r.id IS NULL OR r.skip_reason IS NULL" in _CLAIMS_BY_SITE_SQL
    assert "r.id IS NULL OR r.skip_reason IS NULL" in _CLAIMS_GLOBAL_SQL
