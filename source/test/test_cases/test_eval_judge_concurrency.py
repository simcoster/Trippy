"""Eval can override instruct model and run claim judges in parallel."""

from source.agent.claim_judge import apply_claim_rule_judgements
from source.agent.timing import collect_stages, record_stage
from source.scraper.amenity_enrichment.llm import (
    QWEN_INSTRUCT_30B_MODEL,
    QWEN_INSTRUCT_MODEL,
    instruct_chat_model,
)


def test_instruct_chat_model_30b(monkeypatch):
    monkeypatch.setenv("TRIPPY_INSTRUCT_MODEL", "30B")
    assert instruct_chat_model() == QWEN_INSTRUCT_30B_MODEL


def test_instruct_chat_model_default_is_235b(monkeypatch):
    monkeypatch.delenv("TRIPPY_INSTRUCT_MODEL", raising=False)
    assert instruct_chat_model() == QWEN_INSTRUCT_MODEL


def test_record_stage_counts_parallel_calls():
    with collect_stages() as clock:
        record_stage("judge", 1.5, calls=4)
    snap = clock.snapshot()
    assert snap["judge"]["n"] == 4
    assert snap["judge"]["s"] == 1.5


def test_parallel_judge_covers_every_site(monkeypatch):
    monkeypatch.setenv("TRIPPY_JUDGE_CONCURRENCY", "4")
    seen: list[str] = []

    def _fake(
        *,
        query,
        campsite,
        claims,
        rules,
        usage=None,
        client=None,
        time_stage=True,
    ):
        seen.append(str(campsite))
        return {
            "relevant_claims": [],
            "satisfies": True,
            "satisfy_by": "rule",
            "reason": "ok",
        }

    monkeypatch.setattr("source.agent.claim_judge.judge_site_request", _fake)
    fits = [
        {
            "campsite_id": i,
            "campsite": f"site-{i}",
            "why": [{"query": "tent", "stated_amenity": "tent_pitch"}],
        }
        for i in range(1, 5)
    ]
    out = apply_claim_rule_judgements(
        {"fits": fits, "rejected": [], "rejected_count": 0},
        search_rules=lambda *a, **k: [],
    )
    assert sorted(seen) == ["site-1", "site-2", "site-3", "site-4"]
    assert len(out["fits"]) == 4
