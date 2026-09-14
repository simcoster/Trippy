"""Claim/rule judge is a LangChain tool so LangSmith nests it like resolve_dates."""

from source.agent.claim_judge import (
    _judge_trace_name,
    _judge_trace_payload,
    claim_judge_tool,
)
from source.agent.tracing import bind_to_current_trace, emit_child_span


def test_claim_judge_tool_invoke_returns_verdict(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

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
        assert query == "pets"
        assert campsite == "אכזיב"
        assert claims[0]["claim"] == "x"
        return {
            "relevant_claims": ["x"],
            "satisfies": False,
            "satisfy_by": None,
            "reason": "nope",
        }

    monkeypatch.setattr("source.agent.claim_judge.judge_site_request", _fake)
    assert claim_judge_tool.name == "claim_judge"
    out = claim_judge_tool.invoke(
        {
            "query": "pets",
            "campsite": "אכזיב",
            "claims": [{"claim": "x", "is_positive": False}],
            "rules": [{"subject": "dogs_allowed", "polarity": False}],
        }
    )
    assert out["reason"] == "nope"
    assert out["relevant_claims"] == ["x"]


def test_judge_trace_marks_each_claim_relevant_or_not():
    inputs, outputs = _judge_trace_payload(
        query="pet friendly",
        campsite="אכזיב",
        claims=[
            {"claim": "Pets are not allowed at the site.", "is_positive": False},
            {"claim": "Staff is friendly.", "is_positive": True},
        ],
        rules=[
            {
                "subject": "dogs_allowed",
                "polarity": False,
                "evidence_span": "אין כניסה לכלבים",
            }
        ],
        verdict={
            "relevant_claims": ["Pets are not allowed at the site."],
            "satisfies": False,
            "satisfy_by": None,
            "reason": "only forbids pets",
        },
    )
    assert inputs["request"] == "pet friendly"
    assert inputs["claims"][0]["claim"] == "Pets are not allowed at the site."
    assert inputs["rules"][0]["subject"] == "dogs_allowed"
    assert outputs["claims"][0]["relevant"] is True
    assert outputs["claims"][1]["relevant"] is False
    assert outputs["rules"][0]["subject"] == "dogs_allowed"
    assert outputs["satisfies"] is False
    assert outputs["satisfy_by"] is None
    assert outputs["reason"] == "only forbids pets"


def test_judge_trace_name_includes_campsite_and_query():
    assert _judge_trace_name("אכזיב", "desert") == "claim_judge · אכזיב · desert"


def test_emit_child_span_is_noop_when_tracing_off(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    emit_child_span(
        name="claim_judge",
        inputs={"request": "x"},
        outputs={"satisfies": False},
        tags=["claim_judge"],
    )


def test_bind_to_current_trace_is_identity_when_tracing_off(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

    def _fn(value: int) -> int:
        return value + 1

    assert bind_to_current_trace(_fn) is _fn
