"""LangSmith traces for LangGraph Streamlit turns, not ingest."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from typing import Any

DEFAULT_PROJECT = "trippy"
_API_KEY_VARS = ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
_WS = re.compile(r"\s+")
_configured = False


def api_key() -> str:
    for name in _API_KEY_VARS:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


def project_name() -> str:
    return (
        (os.environ.get("LANGSMITH_PROJECT") or "").strip()
        or (os.environ.get("LANGCHAIN_PROJECT") or "").strip()
        or DEFAULT_PROJECT
    )


def tracing_configured() -> bool:
    return bool(api_key()) and _configured


def configure_agent_tracing() -> bool:
    """Turn on LangSmith for this process when an API key is set.

    Call from the Streamlit entrypoint only. Scrape jobs must not
    call this — ingest ChatOpenAI calls would otherwise land in the same
    project as tester turns.
    """
    global _configured
    if not api_key():
        _configured = False
        return False
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    project = project_name()
    os.environ.setdefault("LANGSMITH_PROJECT", project)
    os.environ.setdefault("LANGCHAIN_PROJECT", project)
    _configured = True
    return True


def run_name(user_text: str, *, max_len: int = 80) -> str:
    text = _WS.sub(" ", (user_text or "").strip())
    if not text:
        return "trippy-turn"
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def agent_run_config(
    *,
    thread_id: str,
    channel: str,
    user_text: str = "",
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """RunnableConfig so LangSmith groups turns by browser session."""
    metadata: dict[str, Any] = {
        "channel": channel,
        "thread_id": thread_id,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return {
        "run_name": run_name(user_text),
        "tags": ["trippy", channel],
        "metadata": metadata,
        "configurable": {"thread_id": thread_id},
    }


def tracing_env_on() -> bool:
    flag = (
        os.environ.get("LANGSMITH_TRACING") or os.environ.get("LANGCHAIN_TRACING_V2") or ""
    ).strip().casefold()
    return flag in {"1", "true", "yes"}


def bind_to_current_trace(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Restore the LangSmith parent run inside a ThreadPool worker."""
    if not tracing_env_on():
        return fn
    try:
        from langsmith import get_current_run_tree, tracing_context
    except ImportError:
        return fn
    parent = get_current_run_tree()
    if parent is None:
        return fn

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        with tracing_context(parent=parent):
            return fn(*args, **kwargs)

    return _wrapped


def emit_child_span(
    *,
    name: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    tags: list[str] | None = None,
) -> None:
    """Attach a child run to the current LangSmith trace. No-op if none."""
    if not tracing_env_on():
        return
    try:
        from langsmith import get_current_run_tree
    except ImportError:
        return
    parent = get_current_run_tree()
    if parent is None:
        return
    try:
        child = parent.create_child(
            name=name,
            run_type="chain",
            inputs=inputs,
            tags=tags,
        )
        child.end(outputs=outputs)
        child.post()
    except Exception:
        return
