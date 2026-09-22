"""Clocks, usage, and the log line for one recommend stream."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from source.scraper.amenity_enrichment.llm import (
    collected_llm_usage,
    langchain_chat_usage,
)

logger = logging.getLogger(__name__)

# Process snapshot, not a ContextVar: LangGraph copies context into the
# node, so a ContextVar set there never comes back to Streamlit.
_last_recommend_timing: dict[str, Any] | None = None
_last_recommend_timing_lock = threading.Lock()


@dataclass
class RecommendClock:
    """First chunk, first spoken paint, and empty prefixes on one stream."""

    started: float
    chunk_at: float | None = None
    spoken_at: float | None = None
    empty_prefix: int = 0
    thinking_stream: bool = False

    def note_thinking(self) -> None:
        self.thinking_stream = True

    def note_empty(self) -> None:
        if self.chunk_at is None:
            self.empty_prefix += 1

    def note_text(self) -> None:
        if self.chunk_at is None:
            self.chunk_at = time.perf_counter()

    def note_spoken(self) -> None:
        if self.spoken_at is None:
            self.spoken_at = time.perf_counter()


@dataclass(frozen=True)
class RecommendTimings:
    chunk_ms: float | None
    spoken_ms: float | None
    elapsed_ms: float
    reasoning_tokens: int
    thinking_stream: bool


def last_recommend_timing() -> dict[str, Any] | None:
    """TTFT inside the last recommend call: first LLM chunk, first spoken paint."""
    with _last_recommend_timing_lock:
        if _last_recommend_timing is None:
            return None
        return dict(_last_recommend_timing)


def _store_recommend_timing(row: dict[str, Any]) -> None:
    global _last_recommend_timing
    with _last_recommend_timing_lock:
        _last_recommend_timing = row


def record_recommend(
    clock: RecommendClock,
    *,
    model: str,
    fallback_from: str | None,
    extra: dict[str, Any] | None,
    usage_from: Any,
) -> RecommendTimings:
    """Bill the call, store TTFT, and print the recommend timing line."""
    elapsed_ms = (time.perf_counter() - clock.started) * 1000
    chunk_ms = (
        (clock.chunk_at - clock.started) * 1000
        if clock.chunk_at is not None
        else None
    )
    spoken_ms = (
        (clock.spoken_at - clock.started) * 1000
        if clock.spoken_at is not None
        else None
    )
    sink = collected_llm_usage()
    raw_usage = (
        langchain_chat_usage(usage_from) if usage_from is not None else None
    )
    reasoning = int(getattr(raw_usage, "reasoning_tokens", 0) or 0)
    if sink is not None and raw_usage is not None:
        sink.add_chat(raw_usage, role="recommend", model=model)
    _store_recommend_timing(
        {
            "model": model,
            "fallback_from": fallback_from,
            "chunk_ms": chunk_ms,
            "spoken_ms": spoken_ms,
            "total_ms": elapsed_ms,
            "reasoning_tokens": reasoning,
            "thinking_stream": clock.thinking_stream,
            "empty_prefix": clock.empty_prefix,
        }
    )
    chunk_s = None if chunk_ms is None else f"{chunk_ms / 1000:.1f}s"
    spoken_s = None if spoken_ms is None else f"{spoken_ms / 1000:.1f}s"
    fallback_s = f" fallback_from={fallback_from}" if fallback_from else ""
    line = (
        f"recommend model={model} extra_body={extra}{fallback_s} "
        f"ttft_chunk={chunk_s} ttft_spoken={spoken_s} "
        f"elapsed={elapsed_ms / 1000:.1f}s "
        f"in={getattr(raw_usage, 'prompt_tokens', None)} "
        f"out={getattr(raw_usage, 'completion_tokens', None)} "
        f"reasoning={reasoning} empty_prefix={clock.empty_prefix} "
        f"thinking_stream={clock.thinking_stream}"
    )
    print(line, flush=True)
    logger.info(line)
    if reasoning or clock.thinking_stream:
        warn = (
            f"recommender thinking still on reasoning={reasoning} "
            f"thinking_stream={clock.thinking_stream}"
        )
        print(warn, flush=True)
        logger.warning(warn)
    return RecommendTimings(
        chunk_ms=chunk_ms,
        spoken_ms=spoken_ms,
        elapsed_ms=elapsed_ms,
        reasoning_tokens=reasoning,
        thinking_stream=clock.thinking_stream,
    )
