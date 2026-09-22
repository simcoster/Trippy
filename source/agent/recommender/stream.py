"""Recommend LLM stream: one model call, optional first-token deadline."""

from __future__ import annotations

import contextvars
import queue
import threading
import time
from collections.abc import Iterator
from typing import Any, NamedTuple

from source.agent.messages import message_text


class FirstTokenTimeout(TimeoutError):
    """No text (or thinking) token arrived before the deadline."""


class RecommendCall(NamedTuple):
    """One model for a recommend stream. `chat` None uses the model factory."""

    model: str
    chat: Any = None
    first_token_sec: float = 0.0


def chunk_text(chunk: Any) -> str:
    return message_text(getattr(chunk, "content", None))


def chunk_thinking(chunk: Any) -> str:
    extra = getattr(chunk, "additional_kwargs", None) or {}
    if not isinstance(extra, dict):
        return ""
    for key in ("reasoning_content", "reasoning", "thinking"):
        text = extra.get(key)
        if isinstance(text, str) and text.strip():
            return text
    return ""


def _chunk_has_token(chunk: Any) -> bool:
    return bool(chunk_text(chunk) or chunk_thinking(chunk))


def _iter_chat_chunks_blocking(chat: Any, messages: list[Any]) -> Iterator[Any]:
    if not hasattr(chat, "stream"):
        yield chat.invoke(messages)
        return
    try:
        stream = chat.stream(messages, stream_usage=True)
    except TypeError:
        stream = chat.stream(messages)
    yield from stream


def iter_chat_chunks(
    chat: Any,
    messages: list[Any],
    *,
    first_token_sec: float = 0.0,
) -> Iterator[Any]:
    if first_token_sec <= 0:
        yield from _iter_chat_chunks_blocking(chat, messages)
        return
    pending: queue.Queue[tuple[str, Any]] = queue.Queue()

    def _run() -> None:
        try:
            for chunk in _iter_chat_chunks_blocking(chat, messages):
                pending.put(("chunk", chunk))
        except Exception as exc:
            pending.put(("err", exc))
        else:
            pending.put(("done", None))

    # The first-token wait runs the stream off-thread. Copy the caller
    # context so LangSmith still records that model call under this node.
    caller = contextvars.copy_context()
    threading.Thread(
        target=caller.run,
        args=(_run,),
        daemon=True,
        name="recommend-ttft",
    ).start()
    deadline = time.monotonic() + first_token_sec
    saw_token = False
    while True:
        remaining = None if saw_token else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise FirstTokenTimeout(
                f"no recommend token in {first_token_sec:g}s"
            )
        try:
            kind, payload = pending.get(timeout=remaining)
        except queue.Empty as exc:
            raise FirstTokenTimeout(
                f"no recommend token in {first_token_sec:g}s"
            ) from exc
        if kind == "err":
            raise payload
        if kind == "done":
            return
        yield payload
        if not saw_token and _chunk_has_token(payload):
            saw_token = True
