"""All stay windows are one availability query and one sandbox quote."""

from source.agent.search import _open_slots_sql, _sandbox_quotes_for_slots
from source.price_sandbox.client import QuoteReply, QuoteRequest


def test_windows_sql_is_one_query():
    sql, params = _open_slots_sql(
        windows=[
            {"start": "2026-09-22", "end": "2026-09-23"},
            {"start": "2026-09-26", "end": "2026-09-27"},
        ],
        site_id=None,
        party_size=4,
        limit=80,
    )
    assert sql is not None
    assert sql.count("unnest") == 1
    assert "c.parent_id" in sql
    assert "COUNT(DISTINCT a.start_date) = w.night_count" in sql
    assert "rn <= %s" in sql
    assert params[2] == [1, 1]
    assert params[-1] == 80


def test_one_quote_covers_weekday_and_weekend(monkeypatch):
    monkeypatch.setattr(
        "source.agent.search.sandbox_url", lambda: "http://127.0.0.1:8503"
    )
    monkeypatch.setattr(
        "source.agent.search.sandbox_reachable", lambda **_: True
    )
    seen: list[list[QuoteRequest]] = []

    def fake_replies(requests: list[QuoteRequest], **_kwargs):
        seen.append(list(requests))
        return [
            QuoteReply(
                request_id=item.request_id,
                ok=True,
                price=100.0,
                explanation="ok",
            )
            for item in requests
        ]

    monkeypatch.setattr("source.agent.search.quote_replies", fake_replies)
    batch = _sandbox_quotes_for_slots(
        [
            {
                "campsite_id": 1,
                "campsite": "אכזיב",
                "accommodation_type": "אוהל",
                "start": "2026-09-22",
                "end": "2026-09-23",
            },
            {
                "campsite_id": 1,
                "campsite": "אכזיב",
                "accommodation_type": "אוהל",
                "start": "2026-09-26",
                "end": "2026-09-27",
            },
        ],
        party_size=2,
        rate_period="weekday",
    )
    assert len(seen) == 1
    weekends = [item.params.is_weekend_or_holiday for item in seen[0]]
    assert weekends == [False, True]
    assert len(batch.by_key) == 2
