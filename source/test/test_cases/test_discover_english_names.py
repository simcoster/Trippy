"""scrape-sites fills campsites.english_name from one 235B call."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from source.scraper.amenity_enrichment.llm import LlmUsage
from source.scraper.discover_sites import (
    _english_by_id,
    english_listing_names_from_html,
    fill_english_names,
)


def test_english_by_id_reads_string_keys():
    assert _english_by_id({"12": "Horshat Tal", "38": "Achziv South"}) == {
        12: "Horshat Tal",
        38: "Achziv South",
    }


def test_english_by_id_skips_nulls():
    assert _english_by_id({"1": "Horshat Tal Campsite", "5": None, "6": "null"}) == {
        1: "Horshat Tal Campsite",
    }


def test_english_listing_names_from_html_takes_article_h2_and_dedupes():
    html = """
    <html><body>
      <h2>Categories</h2>
      <div class="article_content"><h2>Horshat Tal Campsite</h2></div>
      <div class="article_content"><h2>Akhziv Campsite</h2></div>
      <div class="article_content"><h2>Horshat Tal Campsite</h2></div>
    </body></html>
    """
    assert english_listing_names_from_html(html) == [
        "Horshat Tal Campsite",
        "Akhziv Campsite",
    ]


def _chat_client(payload: dict) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(payload, ensure_ascii=False)
                )
            )
        ],
    )
    return client


def test_fill_english_names_writes_mapped_rows():
    select_cur = MagicMock()
    select_cur.fetchall.return_value = [(12, "חניון לילה גן לאומי חורשת טל")]
    update_cur = MagicMock()
    update_cur.rowcount = 1
    select_conn = MagicMock()
    select_conn.cursor.return_value.__enter__.return_value = select_cur
    select_conn.__enter__.return_value = select_conn
    select_conn.__exit__.return_value = False
    update_conn = MagicMock()
    update_conn.cursor.return_value.__enter__.return_value = update_cur
    update_conn.__enter__.return_value = update_conn
    update_conn.__exit__.return_value = False
    usage = LlmUsage()
    client = _chat_client({"12": "Horshat Tal"})
    with (
        patch(
            "source.scraper.discover_sites.connect",
            side_effect=[select_conn, update_conn],
        ),
        patch(
            "source.scraper.discover_sites.instruct_chat_model",
            return_value="Qwen/Qwen3-235B-A22B-Instruct-2507",
        ),
    ):
        written = fill_english_names(
            ["Horshat Tal Campsite"], client=client, usage=usage
        )
    assert written == 1
    assert usage.chat_calls == 1
    update_cur.execute.assert_called_once()
    params = update_cur.execute.call_args.args[1]
    assert params["id"] == 12
    assert params["english_name"] == "Horshat Tal"
