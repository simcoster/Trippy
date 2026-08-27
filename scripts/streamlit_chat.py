"""
Local Streamlit harness for the LangGraph agent.

Mirrors Telegram's invoke path (HumanMessage → graph.invoke → last AIMessage)
without requiring TELEGRAM_TOKEN or sending replies to Telegram.

Also records a per-turn LangGraph trace: nodes, LLM prompts/responses,
and tool calls (params + returns).

Run from repo root:
  uv run streamlit run scripts/streamlit_chat.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any
from uuid import UUID

# Repo root on sys.path so `source.*` imports work under `streamlit run`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Suppress Pydantic V1 compatibility warning with Python 3.14+
warnings.filterwarnings("ignore", message=".*Pydantic V1.*", category=UserWarning)

import streamlit as st
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
)
from langchain_core.outputs import LLMResult

import source.agent.graph as agent_graph
from source.agent.graph import AGENT_CHAT_MODEL, ChatState, graph

load_dotenv(_ROOT / ".env")

st.set_page_config(
    page_title="Trippy Agent (local)",
    page_icon="⛺",
    layout="wide",
)

# Active turn trace (set while invoke_agent runs)
_current_trace: list[dict[str, Any]] | None = None


def _truncate(text: str, max_len: int = 4000) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n… ({len(text) - max_len} more chars)"


def _content_to_str(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        return str(content)


def _format_lc_message(msg: Any) -> dict[str, Any]:
    if isinstance(msg, BaseMessage):
        row: dict[str, Any] = {
            "type": type(msg).__name__,
            "content": _content_to_str(msg.content),
        }
        if isinstance(msg, ChatMessage):
            row["role"] = getattr(msg, "role", None)
        return row
    if isinstance(msg, dict):
        return {
            "type": msg.get("type") or msg.get("role") or "dict",
            "content": _content_to_str(msg.get("content", msg)),
        }
    return {"type": type(msg).__name__, "content": _content_to_str(msg)}


def _format_prompt_messages(messages: list[Any]) -> list[dict[str, Any]]:
    # Chat model callbacks pass list[list[BaseMessage]] (batch)
    if messages and isinstance(messages[0], list):
        flat: list[Any] = []
        for batch in messages:
            flat.extend(batch)
        messages = flat
    return [_format_lc_message(m) for m in messages]


class TraceCallbackHandler(BaseCallbackHandler):
    """Capture node starts + LLM prompts/responses into the active turn trace."""

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if _current_trace is None:
            return
        metadata = kwargs.get("metadata") or {}
        node = metadata.get("langgraph_node")
        if not node:
            return
        if any(
            e.get("kind") == "node"
            and e.get("name") == node
            and e.get("run_id") == str(run_id)
            for e in _current_trace
        ):
            return
        _current_trace.append(
            {
                "kind": "node",
                "name": node,
                "run_id": str(run_id),
                "phase": "start",
            }
        )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if _current_trace is None:
            return
        metadata = kwargs.get("metadata") or {}
        model = (
            (serialized or {}).get("kwargs", {}).get("model_name")
            or (serialized or {}).get("kwargs", {}).get("model")
            or metadata.get("ls_model_name")
            or "chat_model"
        )
        _current_trace.append(
            {
                "kind": "llm_start",
                "run_id": str(run_id),
                "node": metadata.get("langgraph_node"),
                "model": model,
                "prompt": _format_prompt_messages(messages),
            }
        )

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        if _current_trace is None:
            return
        generations: list[str] = []
        for gen_list in response.generations or []:
            for gen in gen_list:
                text = getattr(gen, "text", None)
                if text is None and getattr(gen, "message", None) is not None:
                    text = _content_to_str(gen.message.content)
                generations.append(text or "")
        _current_trace.append(
            {
                "kind": "llm_end",
                "run_id": str(run_id),
                "response": generations[0] if len(generations) == 1 else generations,
            }
        )

def _install_tool_hooks() -> None:
    """Wrap imperative tool functions so Streamlit can log params/returns."""
    if getattr(agent_graph, "_trippy_streamlit_hooks", False):
        return

    def _wrap(name: str, fn: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            if _current_trace is not None:
                if name == "search_claims":
                    params = {
                        "query": args[0] if args else kwargs.get("query"),
                        "limit": args[1] if len(args) > 1 else kwargs.get("limit", 5),
                    }
                elif name == "search_campsites":
                    params = {
                        "numeric_constraints": args[0]
                        if args
                        else kwargs.get("numeric_constraints"),
                    }
                else:
                    params = {"args": list(args), "kwargs": kwargs}
                _current_trace.append(
                    {
                        "kind": "tool",
                        "name": name,
                        "params": params,
                        "result": result,
                    }
                )
            return result

        wrapper.__name__ = getattr(fn, "__name__", name)
        wrapper.__doc__ = getattr(fn, "__doc__", None)
        return wrapper

    agent_graph.search_claims = _wrap("search_claims", agent_graph.search_claims)
    agent_graph.search_campsites = _wrap(
        "search_campsites", agent_graph.search_campsites
    )
    agent_graph._trippy_streamlit_hooks = True


_install_tool_hooks()


def _init_session() -> None:
    if "graph_messages" not in st.session_state:
        st.session_state.graph_messages = []
    if "display" not in st.session_state:
        st.session_state.display = []


def _message_preview(msg: BaseMessage, max_len: int = 400) -> str:
    content = _content_to_str(getattr(msg, "content", ""))
    if len(content) > max_len:
        return content[:max_len] + "…"
    return content


def _serialize_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for msg in messages:
        row: dict[str, Any] = {
            "type": type(msg).__name__,
            "content": getattr(msg, "content", None),
        }
        if isinstance(msg, ChatMessage):
            row["role"] = getattr(msg, "role", None)
        rows.append(row)
    return rows


def _last_ai_reply(messages: list[BaseMessage]) -> str:
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    for msg in reversed(ai_messages):
        text = _content_to_str(msg.content).strip()
        if text:
            return text
    last = messages[-1] if messages else None
    if last is not None and hasattr(last, "content"):
        text = _content_to_str(last.content).strip()
        if text:
            return text
    return (
        "לא הצלחתי להשלים תשובה כרגע. נסו לנסח שוב עם תאריך, מיקום או העדפה."
    )


def _json_block(data: Any) -> None:
    st.code(
        _truncate(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            max_len=12000,
        ),
        language="json",
    )


def _render_trace(trace: list[dict[str, Any]]) -> None:
    nodes = [
        e["name"]
        for e in trace
        if e.get("kind") == "node" and e.get("phase") != "update"
    ]
    if not nodes:
        nodes = [e["name"] for e in trace if e.get("kind") == "node"]
    seen: set[str] = set()
    path: list[str] = []
    for n in nodes:
        if n not in seen:
            seen.add(n)
            path.append(n)
    if path:
        st.markdown("**Nodes:** " + " → ".join(f"`{n}`" for n in path))

    for i, event in enumerate(trace):
        kind = event.get("kind")
        if kind == "node":
            phase = event.get("phase", "update")
            title = f"{i + 1}. Node `{event['name']}`"
            if phase == "start":
                st.markdown(f"**{title}** _(enter)_")
                continue
            with st.expander(f"{title} · state update", expanded=False):
                _json_block(event.get("update"))
        elif kind == "llm_start":
            node = event.get("node")
            label = f"{i + 1}. LLM prompt · `{event.get('model', 'chat_model')}`"
            if node:
                label += f" _(in `{node}`)_"
            with st.expander(label, expanded=True):
                for msg in event.get("prompt") or []:
                    st.markdown(f"**{msg.get('type')}**")
                    st.code(_truncate(msg.get("content") or ""), language=None)
        elif kind == "llm_end":
            with st.expander(f"{i + 1}. LLM response", expanded=False):
                st.code(
                    _truncate(_content_to_str(event.get("response"))),
                    language=None,
                )
        elif kind == "tool":
            with st.expander(
                f"{i + 1}. Tool `{event.get('name')}`",
                expanded=True,
            ):
                st.markdown("**Params**")
                _json_block(event.get("params"))
                st.markdown("**Result**")
                _json_block(event.get("result"))
        else:
            with st.expander(f"{i + 1}. {kind}", expanded=False):
                _json_block(event)

def invoke_agent(user_text: str) -> tuple[str, list[dict[str, Any]]]:
    """Same contract as main.telegram_webhook, plus a LangGraph turn trace."""
    global _current_trace

    history: list[BaseMessage] = list(st.session_state.graph_messages)
    history.append(HumanMessage(content=user_text))
    state: ChatState = {"messages": history}

    trace: list[dict[str, Any]] = []
    _current_trace = trace
    handler = TraceCallbackHandler()
    config = {"callbacks": [handler]}

    final_messages: list[BaseMessage] | None = None
    try:
        for mode, chunk in graph.stream(
            state,
            config=config,
            stream_mode=["updates", "values"],
        ):
            if mode == "updates" and isinstance(chunk, dict):
                for node_name, update in chunk.items():
                    if isinstance(update, dict) and "messages" in update:
                        serialized_update: Any = {
                            "messages": _serialize_messages(update["messages"])
                        }
                    else:
                        serialized_update = update
                    trace.append(
                        {
                            "kind": "node",
                            "name": node_name,
                            "phase": "update",
                            "update": serialized_update,
                        }
                    )
            elif mode == "values" and isinstance(chunk, dict):
                final_messages = chunk.get("messages")
    finally:
        _current_trace = None

    if final_messages is None:
        result = graph.invoke(state, config=config)
        final_messages = result["messages"]

    st.session_state.graph_messages = final_messages
    return _last_ai_reply(final_messages), trace


_init_session()

st.title("Trippy agent")
st.caption(
    f"Local Streamlit client · `{AGENT_CHAT_MODEL}` via Nebius · "
    "production remains Telegram"
)

with st.sidebar:
    st.header("Session")
    if st.button("Reset conversation", use_container_width=True):
        st.session_state.graph_messages = []
        st.session_state.display = []
        st.rerun()

    st.divider()
    st.subheader("Last turn trace")
    last_trace = None
    for turn in reversed(st.session_state.display):
        if turn.get("role") == "assistant" and turn.get("trace"):
            last_trace = turn["trace"]
            break
    if last_trace:
        _render_trace(last_trace)
        st.download_button(
            "Download last trace (JSON)",
            data=json.dumps(last_trace, ensure_ascii=False, indent=2, default=str),
            file_name="trippy_langgraph_trace.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.info("Send a message to see nodes, prompts, and tools.")

    st.divider()
    st.subheader("Graph messages")
    st.caption(f"{len(st.session_state.graph_messages)} messages in LangGraph state")
    if st.session_state.graph_messages:
        for i, msg in enumerate(st.session_state.graph_messages):
            label = type(msg).__name__
            if isinstance(msg, ChatMessage):
                label = f"ChatMessage({getattr(msg, 'role', '?')})"
            with st.expander(f"{i}. {label}", expanded=False):
                st.code(_message_preview(msg, max_len=2000), language=None)
        st.download_button(
            "Download full state (JSON)",
            data=json.dumps(
                _serialize_messages(st.session_state.graph_messages),
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            file_name="trippy_graph_messages.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.info("Send a message to start a conversation.")

for turn in st.session_state.display:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("role") == "assistant" and turn.get("trace"):
            with st.expander("LangGraph trace", expanded=False):
                _render_trace(turn["trace"])

if prompt := st.chat_input("Ask about campsites…"):
    st.session_state.display.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                reply, trace = invoke_agent(prompt)
            except Exception as e:
                reply = f"Sorry, I encountered an error: {e}"
                trace = []
        st.markdown(reply)
        if trace:
            with st.expander("LangGraph trace", expanded=True):
                _render_trace(trace)
    st.session_state.display.append(
        {"role": "assistant", "content": reply, "trace": trace}
    )
    st.rerun()
