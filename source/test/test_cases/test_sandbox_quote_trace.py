"""Price-sandbox quotes are recorded for Streamlit / LangSmith."""

from source.agent.search import (
    _sandbox_quotes_for_slots,
    price_quote_cache,
)
from source.price_sandbox.client import QuoteReply, QuoteRequest, quote_via_sandbox
from source.price_sandbox.params import QuoteParams
from source.price_sandbox.server import MAX_BATCH


def test_sandbox_report_skips_when_url_unset(monkeypatch):
    monkeypatch.setattr("source.agent.search.sandbox_url", lambda: None)
    batch = _sandbox_quotes_for_slots(
        [
            {
                "campsite_id": 2,
                "campsite": "חורשת טל",
                "accommodation_type": "אוהל",
            }
        ],
        party_size=2,
        rate_period="weekday",
    )
    assert batch.by_key == {}
    assert batch.report["skipped"] == "PRICE_SANDBOX_URL unset"
    assert batch.report["calls"][0]["campsite"] == "חורשת טל"
    assert batch.report["calls"][0]["error"] == "PRICE_SANDBOX_URL unset"


def test_sandbox_report_per_campsite(monkeypatch):
    monkeypatch.setattr(
        "source.agent.search.sandbox_url", lambda: "http://127.0.0.1:8503"
    )
    monkeypatch.setattr(
        "source.agent.search.sandbox_reachable", lambda **_: True
    )

    def fake_replies(requests: list[QuoteRequest], **_kwargs):
        assert len(requests) == 1
        assert requests[0].site_id == 2
        return [
            QuoteReply(
                request_id=requests[0].request_id,
                ok=True,
                price=152.0,
                explanation="2 adults × 76",
            )
        ]

    monkeypatch.setattr("source.agent.search.quote_replies", fake_replies)
    batch = _sandbox_quotes_for_slots(
        [
            {
                "campsite_id": 2,
                "campsite": "חורשת טל",
                "accommodation_type": "אוהל",
            }
        ],
        party_size=2,
        rate_period="weekday",
    )
    assert batch.report["skipped"] is None
    row = batch.report["calls"][0]
    assert row["ok"] is True
    assert row["campsite"] == "חורשת טל"
    assert row["lodging"] == "אוהל"
    assert row["price"] == 152.0
    assert row["explanation"] == "2 adults × 76"
    key = next(iter(batch.by_key))
    assert key.campsite_id == 2
    assert batch.by_key[key].price == 152.0


def test_sandbox_report_keeps_jail_error(monkeypatch):
    monkeypatch.setattr(
        "source.agent.search.sandbox_url", lambda: "http://127.0.0.1:8503"
    )
    monkeypatch.setattr(
        "source.agent.search.sandbox_reachable", lambda **_: True
    )
    monkeypatch.setattr(
        "source.agent.search.quote_replies",
        lambda requests, **_: [
            QuoteReply(
                request_id=requests[0].request_id,
                ok=False,
                error="unknown lodging",
            )
        ],
    )
    batch = _sandbox_quotes_for_slots(
        [
            {
                "campsite_id": 2,
                "campsite": "חורשת טל",
                "accommodation_type": "אוהל",
            }
        ],
        party_size=2,
        rate_period="weekday",
    )
    assert batch.by_key == {}
    assert batch.report["calls"][0]["error"] == "unknown lodging"


def test_quote_via_sandbox_chunks_over_max_batch(monkeypatch):
    seen: list[int] = []

    def fake_post(_url, payload, **_kwargs):
        n = len(payload["quotes"])
        seen.append(n)
        assert n <= MAX_BATCH
        return {
            "ok": True,
            "results": [
                {
                    "id": row["id"],
                    "ok": True,
                    "price": 1.0,
                    "explanation": "",
                }
                for row in payload["quotes"]
            ],
        }

    monkeypatch.setattr("source.price_sandbox.client._post", fake_post)
    requests = [
        QuoteRequest(
            request_id=str(i),
            site_id=1,
            params=QuoteParams(lodging="אוהל", adults_num=1),
        )
        for i in range(MAX_BATCH + 1)
    ]
    out = quote_via_sandbox(requests, base_url="http://127.0.0.1:8503")
    assert seen == [MAX_BATCH, 1]
    assert len(out) == MAX_BATCH + 1


def test_sandbox_quotes_reuse_site_type_weekend(monkeypatch):
    monkeypatch.setattr(
        "source.agent.search.sandbox_url", lambda: "http://127.0.0.1:8503"
    )
    monkeypatch.setattr(
        "source.agent.search.sandbox_reachable", lambda **_: True
    )
    calls = {"n": 0}

    def fake_replies(requests: list[QuoteRequest], **_kwargs):
        calls["n"] += 1
        return [
            QuoteReply(
                request_id=requests[0].request_id,
                ok=True,
                price=152.0,
                explanation="cached-me",
            )
        ]

    monkeypatch.setattr("source.agent.search.quote_replies", fake_replies)
    slot = {
        "campsite_id": 2,
        "campsite": "חורשת טל",
        "accommodation_type": "אוהל",
    }
    with price_quote_cache():
        first = _sandbox_quotes_for_slots(
            [slot], party_size=2, rate_period="weekday"
        )
        second = _sandbox_quotes_for_slots(
            [slot], party_size=2, rate_period="weekday"
        )
        weekend = _sandbox_quotes_for_slots(
            [slot], party_size=2, rate_period="weekend_holiday"
        )
    assert calls["n"] == 2
    assert first.by_key == second.by_key
    assert second.report["cached"] == 1
    assert second.report["calls"][0]["cached"] is True
    assert weekend.report["cached"] == 0


def test_sandbox_quotes_do_not_leak_across_requests(monkeypatch):
    monkeypatch.setattr(
        "source.agent.search.sandbox_url", lambda: "http://127.0.0.1:8503"
    )
    monkeypatch.setattr(
        "source.agent.search.sandbox_reachable", lambda **_: True
    )
    calls = {"n": 0}

    def fake_replies(requests: list[QuoteRequest], **_kwargs):
        calls["n"] += 1
        return [
            QuoteReply(
                request_id=requests[0].request_id,
                ok=True,
                price=152.0,
                explanation="cached-me",
            )
        ]

    monkeypatch.setattr("source.agent.search.quote_replies", fake_replies)
    slot = {
        "campsite_id": 2,
        "campsite": "חורשת טל",
        "accommodation_type": "אוהל",
    }
    with price_quote_cache():
        _sandbox_quotes_for_slots([slot], party_size=2, rate_period="weekday")
    with price_quote_cache():
        again = _sandbox_quotes_for_slots(
            [slot], party_size=2, rate_period="weekday"
        )
    assert calls["n"] == 2
    assert again.report["cached"] == 0
