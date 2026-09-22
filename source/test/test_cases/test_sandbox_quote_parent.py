"""A subcamp quote uses the parent's function when the child is not loaded."""

from __future__ import annotations

from source.agent.search import _open_slots_sql, _sandbox_quotes_for_slots
from source.price_sandbox.client import QuoteReply, QuoteRequest
from source.price_sandbox.params import QuoteResult
from source.price_sandbox.server import load_functions, quote_batch


def _source(label: str) -> str:
    return (
        "def quote(lodging, adults_num, child_num=0, child_ages=(), "
        'guest_type="רגיל", is_weekend_or_holiday=False, '
        "planned_entry_time=None, planned_exit_time=None):\n"
        f"    return 1.0, {label!r}\n"
    )


def test_open_slots_sql_selects_parent_id():
    sql, _params = _open_slots_sql(
        date_range={"start": "2026-09-22", "end": "2026-09-23"},
        site_id=None,
        party_size=None,
        limit=10,
    )
    assert sql is not None
    assert "c.parent_id" in sql


def test_quote_uses_parent_when_subcamp_is_not_loaded(monkeypatch):
    parent = _source("parent")
    load_functions([{"site_id": 2, "source": parent, "sha256": "p"}])
    seen: list[str] = []

    def fake_run(source, _params, **_kwargs):
        seen.append(source)
        return QuoteResult(price=10.0, explanation="parent")

    monkeypatch.setattr("source.price_sandbox.server.run_quote", fake_run)
    payload = quote_batch(
        [
            {
                "id": "north",
                "site_id": 37,
                "parent_site_id": 2,
                "params": {"lodging": "אוהל", "adults_num": 2},
            }
        ]
    )
    assert payload["results"][0]["ok"] is True
    assert payload["results"][0]["price"] == 10.0
    assert seen == [parent]


def test_quote_keeps_subcamp_function_when_it_is_loaded(monkeypatch):
    child = _source("child")
    parent = _source("parent")
    load_functions(
        [
            {"site_id": 37, "source": child, "sha256": "c"},
            {"site_id": 2, "source": parent, "sha256": "p"},
        ]
    )
    seen: list[str] = []

    def fake_run(source, _params, **_kwargs):
        seen.append(source)
        return QuoteResult(price=7.0, explanation="child")

    monkeypatch.setattr("source.price_sandbox.server.run_quote", fake_run)
    payload = quote_batch(
        [
            {
                "id": "north",
                "site_id": 37,
                "parent_site_id": 2,
                "params": {"lodging": "אוהל", "adults_num": 2},
            }
        ]
    )
    assert payload["results"][0]["ok"] is True
    assert seen == [child]


def test_quote_unknown_when_parent_is_also_missing(monkeypatch):
    load_functions([])

    def fake_run(*_args, **_kwargs):
        raise AssertionError("run_quote")

    monkeypatch.setattr("source.price_sandbox.server.run_quote", fake_run)
    payload = quote_batch(
        [
            {
                "id": "north",
                "site_id": 37,
                "parent_site_id": 2,
                "params": {"lodging": "אוהל", "adults_num": 2},
            }
        ]
    )
    assert payload["results"][0]["ok"] is False
    assert payload["results"][0]["error"] == "unknown site"


def test_sandbox_quotes_send_parent_site_id(monkeypatch):
    monkeypatch.setattr(
        "source.agent.search.sandbox_url", lambda: "http://127.0.0.1:8503"
    )
    monkeypatch.setattr("source.agent.search.sandbox_reachable", lambda **_: True)
    seen: list[QuoteRequest] = []

    def fake_replies(requests: list[QuoteRequest], **_kwargs):
        seen.extend(requests)
        return [
            QuoteReply(
                request_id=requests[0].request_id,
                ok=True,
                price=152.0,
                explanation="parent card",
            )
        ]

    monkeypatch.setattr("source.agent.search.quote_replies", fake_replies)
    batch = _sandbox_quotes_for_slots(
        [
            {
                "campsite_id": 37,
                "parent_id": 2,
                "campsite": "אכזיב – חניון צפוני",
                "accommodation_type": "אוהל",
            }
        ],
        party_size=2,
        rate_period="weekday",
    )
    assert seen[0].site_id == 37
    assert seen[0].parent_site_id == 2
    assert batch.by_key
    assert next(iter(batch.by_key)).campsite_id == 37
