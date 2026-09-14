"""Recommend usage reports hidden thinking tokens when the provider sends them."""

from types import SimpleNamespace

from source.agent.recommender import recommend_from_payload
from source.eval.run import _recommend_dump, format_recommend_ttft
from source.scraper.amenity_enrichment.llm import langchain_chat_usage


def test_langchain_usage_reads_reasoning_from_token_details():
    raw = langchain_chat_usage(
        SimpleNamespace(
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 4,
                "output_token_details": {"reasoning": 80},
            }
        )
    )
    assert raw.prompt_tokens == 10
    assert raw.completion_tokens == 4
    assert raw.reasoning_tokens == 80


def test_langchain_usage_reads_openai_reasoning_tokens():
    raw = langchain_chat_usage(
        SimpleNamespace(
            usage_metadata=None,
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 2,
                    "completion_tokens_details": {"reasoning_tokens": 50},
                }
            },
        )
    )
    assert raw.reasoning_tokens == 50


def test_format_recommend_ttft_includes_reasoning():
    line = format_recommend_ttft(
        {
            "ttft_chunk_ms": 2100,
            "ttft_spoken_ms": 8400,
            "reasoning_tokens": 80,
            "thinking_stream": True,
        }
    )
    assert line == (
        "ttft_chunk=2.1s ttft_spoken=8.4s reasoning=80 thinking_stream=1"
    )


def test_recommend_dump_and_result_flag_thinking():
    class _Chat:
        def stream(self, messages, **kwargs):
            yield SimpleNamespace(
                content="",
                additional_kwargs={"reasoning_content": "let me think"},
            )
            yield SimpleNamespace(
                content=(
                    '{"recommendations": [], "empty": "מתי?", "intro": null}'
                ),
                usage_metadata={
                    "input_tokens": 12,
                    "output_tokens": 3,
                    "output_token_details": {"reasoning": 40},
                },
            )

    result = recommend_from_payload(
        "ליד הים",
        {"constraints": {}, "fits": []},
        chat=_Chat(),
    )
    assert result.thinking_stream is True
    assert result.reasoning_tokens == 40
    dump = _recommend_dump(result)
    assert dump["thinking_stream"] is True
    assert dump["reasoning_tokens"] == 40
