"""Which recommender model, and how that model turns thinking off."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, NamedTuple

from source.scraper.amenity_enrichment.llm import (
    KIMI_K3_MODEL,
    NEMOTRON_SUPER_MODEL,
    QWEN_INSTRUCT_MODEL,
    make_agent_chat_model,
)

logger = logging.getLogger(__name__)

_NO_THINK = "/no_think"

_clients: dict[tuple[str, str], Any] = {}


class PreparedModel(NamedTuple):
    """System text, request body, and chat client for one recommend call."""

    system: str
    extra_body: dict[str, Any] | None
    chat: Any


def recommender_model() -> str:
    """Kimi-K3 unless `TRIPPY_RECOMMENDER_MODEL` is super / 235B / a full id."""
    raw = (os.environ.get("TRIPPY_RECOMMENDER_MODEL") or "").strip()
    key = raw.casefold()
    if not raw or key in {"kimi", "kimi-k3", "k3"}:
        return KIMI_K3_MODEL
    if key in {"super", "nemotron", "nemotron-super"}:
        return NEMOTRON_SUPER_MODEL
    if key in {"235b", "big", "qwen"}:
        return QWEN_INSTRUCT_MODEL
    return raw


def _system_for(model: str, prompt: str) -> str:
    if "nemotron" in model.casefold():
        return f"{prompt}\n\n{_NO_THINK}"
    return prompt


def _extra_body_for(model: str) -> dict[str, Any] | None:
    name = model.casefold()
    if "nemotron" in name:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    if "instruct-2507" in name or name in {"235b", "big", "qwen"}:
        return None
    extra: dict[str, Any] = {
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "thinking": {"type": "disabled"},
    }
    if "kimi" in name:
        extra["reasoning_effort"] = "none"
    return extra


def chat_for(model: str):
    """LangChain chat client for one recommender model id."""
    extra = _extra_body_for(model)
    key = (model, json.dumps(extra, sort_keys=True) if extra else "")
    client = _clients.get(key)
    if client is None:
        client = make_agent_chat_model(
            temperature=0, model=model, extra_body=extra
        )
        client.stream_usage = True
        _clients[key] = client
        line = f"recommender client model={model} extra_body={extra}"
        print(line, flush=True)
        logger.info(line)
    return client


def _recommender_chat():
    """Default recommender; `TRIPPY_RECOMMENDER_MODEL` rebuilds for probes."""
    return chat_for(recommender_model())


def prepare_model(
    model: str, prompt: str, chat: Any | None = None
) -> PreparedModel:
    """Thinking-off setup for `model`. An injected `chat` is used as-is."""
    return PreparedModel(
        system=_system_for(model, prompt),
        extra_body=_extra_body_for(model),
        chat=chat if chat is not None else chat_for(model),
    )
