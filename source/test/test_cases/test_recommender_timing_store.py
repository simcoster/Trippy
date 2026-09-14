"""Recommend timing is readable after a copied context (LangGraph)."""

from contextvars import copy_context
from types import SimpleNamespace

from source.agent.recommender import last_recommend_timing, recommend_from_payload


class _Chat:
    def stream(self, messages, **kwargs):
        yield SimpleNamespace(
            content='{"recommendations": [], "empty": "מתי?", "intro": null}',
            usage_metadata={
                "input_tokens": 12,
                "output_tokens": 3,
                "output_token_details": {"reasoning": 7},
            },
        )


def test_last_recommend_timing_survives_copied_context(capsys):
    def _run() -> None:
        recommend_from_payload(
            "ליד הים",
            {"constraints": {}, "fits": []},
            chat=_Chat(),
        )

    copy_context().run(_run)
    timing = last_recommend_timing()
    assert timing is not None
    assert timing["reasoning_tokens"] == 7
    assert timing["thinking_stream"] is False
    assert timing["total_ms"] is not None
    out = capsys.readouterr().out
    assert "ttft_chunk=" in out
    assert "reasoning=7" in out
    assert "thinking_stream=False" in out
    assert "recommender thinking still on" in out
