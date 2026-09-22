"""Fall back to another recommend model when the first token is late."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import Any

from source.agent.recommender.stream import FirstTokenTimeout, RecommendCall
from source.scraper.amenity_enrichment.llm import NEMOTRON_SUPER_MODEL

logger = logging.getLogger(__name__)

DEFAULT_KIMI_FIRST_TOKEN_SEC = 10.0


def is_kimi(model: str) -> bool:
    return "kimi" in model.casefold()


def kimi_first_token_sec() -> float:
    """Seconds to wait for a Kimi token before Super. 0 disables. Default 10."""
    raw = (os.environ.get("TRIPPY_KIMI_TTFT_SEC") or "").strip()
    if not raw:
        return DEFAULT_KIMI_FIRST_TOKEN_SEC
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning(
            "invalid TRIPPY_KIMI_TTFT_SEC=%r; using %s",
            raw,
            DEFAULT_KIMI_FIRST_TOKEN_SEC,
        )
        return DEFAULT_KIMI_FIRST_TOKEN_SEC


def primary_recommend_call(model: str, chat: Any | None) -> RecommendCall:
    wait = kimi_first_token_sec() if chat is None and is_kimi(model) else 0.0
    return RecommendCall(model=model, chat=chat, first_token_sec=wait)


def kimi_super_fallback(primary: RecommendCall) -> RecommendCall | None:
    if primary.first_token_sec <= 0 or not is_kimi(primary.model):
        return None
    return RecommendCall(model=NEMOTRON_SUPER_MODEL)


def recommend_with_fallback(
    primary: RecommendCall,
    fallback: RecommendCall | None,
    run: Callable[..., Any],
) -> Any:
    """Run primary; on first-token timeout, run fallback if one was given."""
    started = time.perf_counter()
    try:
        return run(primary, started=started)
    except FirstTokenTimeout:
        if fallback is None:
            raise
        waited = time.perf_counter() - started
        line = (
            f"recommend fallback {primary.model} → {fallback.model} "
            f"after {waited:.1f}s no token"
        )
        print(line, flush=True)
        logger.info(line)
        return run(fallback, started=started, fallback_from=primary.model)
