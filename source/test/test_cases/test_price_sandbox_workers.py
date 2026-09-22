"""Four short-lived quote children. The server dedupes worker calls."""

from __future__ import annotations

from source.price_sandbox.execute import QuoteCall
from source.price_sandbox.params import QuoteResult
from source.price_sandbox.server import load_functions, quote_batch

_QUOTE = (
    "def quote(lodging, adults_num, child_num=0, child_ages=(), "
    'guest_type="רגיל", is_weekend_or_holiday=False, '
    "planned_entry_time=None, planned_exit_time=None):\n"
    "    return float(adults_num), 'ok'\n"
)


def test_duplicate_quotes_share_one_worker_call(monkeypatch):
    seen: list[int] = []

    def fake_run(calls: list[QuoteCall], **_kwargs):
        seen.append(len(calls))
        return [
            QuoteResult(price=float(call.params.adults_num), explanation="ok")
            for call in calls
        ]

    monkeypatch.setattr("source.price_sandbox.server.run_quotes", fake_run)
    load_functions([{"site_id": 7, "source": _QUOTE, "sha256": "dup"}])
    try:
        payload = quote_batch(
            [
                {"id": "a", "site_id": 7, "params": {"lodging": "x", "adults_num": 2}},
                {"id": "b", "site_id": 7, "params": {"lodging": "x", "adults_num": 2}},
                {"id": "c", "site_id": 7, "params": {"lodging": "x", "adults_num": 3}},
            ]
        )
    finally:
        load_functions([])
    assert seen == [2]
    assert [row["id"] for row in payload["results"]] == ["a", "b", "c"]
    assert [row["price"] for row in payload["results"]] == [2.0, 2.0, 3.0]


def test_batch_runs_on_short_lived_children():
    load_functions([{"site_id": 7, "source": _QUOTE, "sha256": "batch"}])
    try:
        payload = quote_batch(
            [
                {
                    "id": str(index),
                    "site_id": 7,
                    "params": {"lodging": "x", "adults_num": index + 1},
                }
                for index in range(6)
            ]
        )
    finally:
        load_functions([])
    assert payload["ok"] is True
    assert [row["price"] for row in payload["results"]] == [
        float(index + 1) for index in range(6)
    ]
