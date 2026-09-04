"""LlmUsage prices each call by the model that made it and reports per role.

Before this, `cost_usd` charged every chat call at the 235B extractor's rate
and every embedding at one rate, so a run mixing the 30B classifier, the 235B
judge and the embedder was mispriced. Now each `add_chat` / `add_embed` names
its role and model, cost comes from the rate table per model, and a run can be
written to the scrape cost log as one JSON line.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from source.scraper.amenity_enrichment.llm import (
    EMBED_INPUT_USD_PER_MTOK,
    QWEN_INSTRUCT_30B_MODEL,
    QWEN_INSTRUCT_MODEL,
    LlmUsage,
    chat_usd_per_mtok,
    record_scrape_cost,
)


def chat(prompt: int, completion: int) -> SimpleNamespace:
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)


def test_cost_is_priced_per_model_not_at_one_flat_rate():
    usage = LlmUsage()
    usage.add_chat(chat(1_000_000, 0), role="judge", model=QWEN_INSTRUCT_MODEL)
    usage.add_chat(chat(1_000_000, 0), role="classify", model=QWEN_INSTRUCT_30B_MODEL)

    big_in, _ = chat_usd_per_mtok(QWEN_INSTRUCT_MODEL)
    small_in, _ = chat_usd_per_mtok(QWEN_INSTRUCT_30B_MODEL)
    assert big_in != small_in, "the test needs two models with different prices"
    assert usage.cost_usd == pytest.approx(big_in + small_in)
    by_role = {b.role: b for b in usage.by_role()}
    assert by_role["judge"].cost_usd == pytest.approx(big_in)
    assert by_role["classify"].cost_usd == pytest.approx(small_in)


def test_flat_counters_still_add_up_across_roles():
    usage = LlmUsage()
    usage.add_chat(chat(10, 4), role="a", model="m")
    usage.add_chat(chat(5, 1), role="b", model="m")
    usage.add_embed(SimpleNamespace(prompt_tokens=7), role="embed", model="e")

    assert (usage.chat_calls, usage.chat_prompt_tokens, usage.chat_completion_tokens) == (
        2,
        15,
        5,
    )
    assert (usage.embed_calls, usage.embed_prompt_tokens) == (1, 7)
    assert usage.input_tokens == 22
    assert usage.output_tokens == 5


def test_embeddings_are_priced_at_the_embedding_rate():
    usage = LlmUsage()
    usage.add_embed(
        SimpleNamespace(prompt_tokens=1_000_000), role="embed", model="Qwen/Qwen3-Embedding-8B"
    )
    assert usage.cost_usd == pytest.approx(EMBED_INPUT_USD_PER_MTOK)


def test_merge_keeps_roles_apart_and_adds_within_a_role():
    total = LlmUsage()
    total.add_chat(chat(100, 10), role="judge", model="m")
    section = LlmUsage()
    section.add_chat(chat(50, 5), role="judge", model="m")
    section.add_chat(chat(1, 1), role="rules_extract", model="m")

    total.merge(section)

    by_role = {b.role: b for b in total.by_role()}
    assert by_role["judge"].calls == 2
    assert by_role["judge"].prompt_tokens == 150
    assert by_role["rules_extract"].calls == 1
    assert total.chat_calls == 3


def test_summary_keeps_its_headline_and_adds_one_row_per_role():
    usage = LlmUsage()
    usage.add_chat(chat(100, 10), role="judge", model="Qwen/Qwen3-235B-A22B-Instruct-2507")
    usage.add_embed(SimpleNamespace(prompt_tokens=20), role="embed", model="Qwen/Qwen3-Embedding-8B")

    lines = usage.summary(prefix="").splitlines()

    assert lines[0].startswith("LLM usage: in=120 ")
    rows = [line.strip() for line in lines[1:]]
    assert any(r.startswith("judge") and "Qwen3-235B-A22B-Instruct-2507" in r for r in rows)
    assert any(r.startswith("embed") and "Qwen3-Embedding-8B" in r for r in rows)


def test_report_is_json_serialisable_and_names_the_scrape():
    usage = LlmUsage()
    usage.add_chat(chat(100, 10), role="judge", model="m")

    report = usage.report("scrape-rules")

    json.dumps(report)  # must not raise
    assert report["kind"] == "scrape-rules"
    assert report["calls"] == 1
    assert report["by_role"][0]["role"] == "judge"
    assert report["by_role"][0]["input_tokens"] == 100


def test_record_scrape_cost_appends_one_line_per_run(tmp_path):
    log = tmp_path / "costs.jsonl"
    usage = LlmUsage()
    usage.add_chat(chat(100, 10), role="judge", model="m")

    assert record_scrape_cost("scrape-rules", usage, path=log) == log
    assert record_scrape_cost("scrape-prices", usage, path=log) == log

    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [r["kind"] for r in rows] == ["scrape-rules", "scrape-prices"]


def test_record_scrape_cost_writes_nothing_for_a_run_with_no_calls(tmp_path):
    log = tmp_path / "costs.jsonl"
    assert record_scrape_cost("scrape-sites", LlmUsage(), path=log) is None
    assert not log.exists()


def test_untagged_counters_are_still_priced_rather_than_zero():
    """Counters set directly, as older code and hand-built fixtures do."""
    usage = LlmUsage(chat_prompt_tokens=1_000_000)
    usd_in, _ = chat_usd_per_mtok(None)
    assert usage.cost_usd == pytest.approx(usd_in)
