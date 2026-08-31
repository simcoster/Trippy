"""
Local Streamlit harness for the LangGraph agent.

Mirrors Telegram's invoke path (HumanMessage → graph.invoke → last AIMessage)
without requiring TELEGRAM_TOKEN or sending replies to Telegram.

Also records a per-turn LangGraph trace: nodes, LLM prompts/responses,
tool calls (params + returns), token cost, and latency.

Run from repo root:
  uv run streamlit run scripts/streamlit_chat.py

streamlit-mcp cannot drive st.chat_input. The sidebar "MCP prompt" form is
driveable (text_area + submit).
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from collections import defaultdict
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
import source.agent.search as agent_search
from source.agent.graph import AGENT_CHAT_MODEL, ChatState, HeavyThrough, build_graph
from source.scraper.amenity_enrichment.llm import (
    EmbeddingLLMClient,
    LlmUsage,
    chat_usd_per_mtok,
)

HEAVY_PATH_LABELS: dict[HeavyThrough, str] = {
    "extractor": "Extractor only",
    "planner": "Extractor + planner",
    "recommender": "Extractor + planner + recommender",
}

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


def _format_node_input(inputs: Any) -> list[dict[str, Any]]:
    """LangGraph node enter: ChatState (or a messages list) → trace rows."""
    if inputs is None:
        return []
    if isinstance(inputs, dict):
        messages = inputs.get("messages")
        if messages is None:
            return []
        return _format_prompt_messages(list(messages))
    if isinstance(inputs, list):
        return _format_prompt_messages(inputs)
    return []


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _usage_from_mapping(raw: Any) -> dict[str, int]:
    if raw is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if not isinstance(raw, dict):
        raw = {
            "prompt_tokens": getattr(raw, "prompt_tokens", None)
            or getattr(raw, "input_tokens", None),
            "completion_tokens": getattr(raw, "completion_tokens", None)
            or getattr(raw, "output_tokens", None),
            "total_tokens": getattr(raw, "total_tokens", None),
        }
    prompt = _int_or_zero(
        raw.get("prompt_tokens") or raw.get("input_tokens")
    )
    completion = _int_or_zero(
        raw.get("completion_tokens") or raw.get("output_tokens")
    )
    total = _int_or_zero(raw.get("total_tokens")) or (prompt + completion)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _usage_from_llm_result(response: LLMResult) -> dict[str, int]:
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    llm_output = getattr(response, "llm_output", None) or {}
    if isinstance(llm_output, dict):
        usage = _usage_from_mapping(
            llm_output.get("token_usage") or llm_output.get("usage")
        )
    if usage["total_tokens"]:
        return usage
    for gen_list in response.generations or []:
        for gen in gen_list:
            msg = getattr(gen, "message", None)
            meta = getattr(msg, "usage_metadata", None) if msg else None
            if meta:
                usage = _usage_from_mapping(meta)
                if usage["total_tokens"]:
                    return usage
            info = getattr(gen, "generation_info", None) or {}
            if isinstance(info, dict):
                usage = _usage_from_mapping(
                    info.get("token_usage") or info.get("usage")
                )
                if usage["total_tokens"]:
                    return usage
    return usage


def _chat_cost_usd(model: str | None, prompt: int, completion: int) -> float:
    in_rate, out_rate = chat_usd_per_mtok(model)
    return (prompt * in_rate + completion * out_rate) / 1_000_000


def _embed_cost_usd(tokens: int) -> float:
    return tokens * EmbeddingLLMClient.INPUT_USD_PER_MTOK / 1_000_000


def _format_latency(ms: float | None) -> str:
    if ms is None:
        return "—"
    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms / 1000:.2f} s"


class TraceCallbackHandler(BaseCallbackHandler):
    """Capture node starts + LLM prompts/responses into the active turn trace."""

    def __init__(self) -> None:
        super().__init__()
        self.node_started_at: dict[str, list[float]] = defaultdict(list)
        self._llm_started_at: dict[str, float] = {}

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
        self.node_started_at[node].append(time.perf_counter())
        _current_trace.append(
            {
                "kind": "node",
                "name": node,
                "run_id": str(run_id),
                "phase": "start",
                "input": _format_node_input(inputs),
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
        self._llm_started_at[str(run_id)] = time.perf_counter()
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
        started = self._llm_started_at.pop(str(run_id), None)
        node = None
        model = None
        for event in reversed(_current_trace):
            if event.get("kind") == "llm_start" and event.get("run_id") == str(run_id):
                node = event.get("node")
                model = event.get("model")
                break
        usage = _usage_from_llm_result(response)
        latency_ms = (
            (time.perf_counter() - started) * 1000 if started is not None else None
        )
        _current_trace.append(
            {
                "kind": "llm_end",
                "run_id": str(run_id),
                "node": node,
                "model": model,
                "response": generations[0] if len(generations) == 1 else generations,
                "usage": usage,
                "latency_ms": latency_ms,
                "cost_usd": _chat_cost_usd(
                    model,
                    usage["prompt_tokens"],
                    usage["completion_tokens"],
                ),
            }
        )

def _install_tool_hooks() -> None:
    """Wrap imperative tool functions so Streamlit can log params/returns."""
    if getattr(agent_graph, "_trippy_streamlit_hooks", False):
        return

    def _wrap(name: str, fn: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            result = fn(*args, **kwargs)
            latency_ms = (time.perf_counter() - started) * 1000
            if _current_trace is not None:
                if name in (
                    "search_claims",
                    "search_review_claims",
                    "search_stated_amenities",
                ):
                    params = {
                        "query": args[0] if args else kwargs.get("query"),
                        "limit": args[1] if len(args) > 1 else kwargs.get("limit", 5),
                    }
                    if name == "search_stated_amenities":
                        params["accommodation_type_ids"] = kwargs.get(
                            "accommodation_type_ids"
                        )
                elif name == "lookup_campsite_by_name":
                    params = {"name": args[0] if args else kwargs.get("name")}
                elif name == "search_open_slots":
                    params = {
                        "date_range": kwargs.get("date_range"),
                        "site_id": kwargs.get("site_id"),
                        "party_size": kwargs.get("party_size"),
                        "numeric_constraints": kwargs.get("numeric_constraints"),
                    }
                    last = getattr(agent_search, "_LAST_OPEN_SLOTS_QUERY", None)
                    if isinstance(last, dict):
                        params.update(last)
                elif name == "search_availability":
                    params = {
                        "hotel_id": args[0] if args else kwargs.get("hotel_id"),
                        "date_range": kwargs.get("date_range"),
                        "party_size": kwargs.get("party_size"),
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
                        "latency_ms": latency_ms,
                    }
                )
            return result

        wrapper.__name__ = getattr(fn, "__name__", name)
        wrapper.__doc__ = getattr(fn, "__doc__", None)
        return wrapper

    agent_search.search_claims = _wrap("search_claims", agent_search.search_claims)
    agent_search.search_review_claims = _wrap(
        "search_review_claims", agent_search.search_review_claims
    )
    agent_search.search_stated_amenities = _wrap(
        "search_stated_amenities", agent_search.search_stated_amenities
    )
    agent_search.lookup_campsite_by_name = _wrap(
        "lookup_campsite_by_name", agent_search.lookup_campsite_by_name
    )
    agent_search.search_open_slots = _wrap(
        "search_open_slots", agent_search.search_open_slots
    )
    agent_search.search_availability = _wrap(
        "search_availability", agent_search.search_availability
    )
    agent_search.search_campsites = _wrap(
        "search_campsites", agent_search.search_campsites
    )
    agent_graph.search_claims = agent_search.search_claims
    agent_graph.search_review_claims = agent_search.search_review_claims
    agent_graph.search_stated_amenities = agent_search.search_stated_amenities
    agent_graph.lookup_campsite_by_name = agent_search.lookup_campsite_by_name
    agent_graph.search_open_slots = agent_search.search_open_slots
    agent_graph.search_availability = agent_search.search_availability
    agent_graph.search_campsites = agent_search.search_campsites

    embedder = getattr(agent_search, "_claims_embedder", None)
    orig_embed = getattr(embedder, "embed", None)
    if orig_embed is not None:

        def embed_wrapper(
            texts: list[str],
            *,
            usage: LlmUsage | None = None,
            **kwargs: Any,
        ) -> Any:
            local = LlmUsage()
            started = time.perf_counter()
            result = orig_embed(texts, usage=local, **kwargs)
            if usage is not None:
                usage.merge(local)
            if _current_trace is not None:
                tokens = local.embed_prompt_tokens
                _current_trace.append(
                    {
                        "kind": "embed",
                        "name": "embed_query",
                        "node": "planner",
                        "prompt_tokens": tokens,
                        "latency_ms": (time.perf_counter() - started) * 1000,
                        "cost_usd": _embed_cost_usd(tokens),
                    }
                )
            return result

        embedder.embed = embed_wrapper

    agent_graph._trippy_streamlit_hooks = True


def _install_node_input_hooks() -> None:
    """Stamp each node's enter event with the ChatState it actually received."""
    if getattr(agent_graph, "_trippy_streamlit_node_input_hooks", False):
        return

    def _wrap_node(name: str, fn: Any) -> Any:
        def wrapper(state: Any, *args: Any, **kwargs: Any) -> Any:
            if _current_trace is not None:
                incoming = _format_node_input(state)
                for event in reversed(_current_trace):
                    if (
                        event.get("kind") == "node"
                        and event.get("name") == name
                        and event.get("phase") == "start"
                    ):
                        event["input"] = incoming
                        break
            return fn(state, *args, **kwargs)

        wrapper.__name__ = getattr(fn, "__name__", name)
        wrapper.__doc__ = getattr(fn, "__doc__", None)
        return wrapper

    agent_graph.light_node = _wrap_node("light", agent_graph.light_node)
    agent_graph.extractor_node = _wrap_node("extractor", agent_graph.extractor_node)
    agent_graph.planner_node = _wrap_node("planner", agent_graph.planner_node)
    agent_graph.recommender_node = _wrap_node(
        "recommender", agent_graph.recommender_node
    )
    agent_graph._trippy_streamlit_node_input_hooks = True


_install_tool_hooks()
_install_node_input_hooks()


def _init_session() -> None:
    if "graph_messages" not in st.session_state:
        st.session_state.graph_messages = []
    if "display" not in st.session_state:
        st.session_state.display = []
    if "heavy_path" not in st.session_state:
        st.session_state.heavy_path = "extractor"


def _reset_conversation() -> None:
    st.session_state.graph_messages = []
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


def _tool_queries_since_node_start(
    trace: list[dict[str, Any]], node_name: str
) -> list[dict[str, Any]]:
    start_at: int | None = None
    for i, event in enumerate(trace):
        if (
            event.get("kind") == "node"
            and event.get("name") == node_name
            and event.get("phase") == "start"
        ):
            start_at = i
    if start_at is None:
        return []
    queries: list[dict[str, Any]] = []
    for event in trace[start_at + 1 :]:
        if event.get("kind") != "tool":
            continue
        row: dict[str, Any] = {"tool": event.get("name")}
        params = event.get("params")
        if isinstance(params, dict):
            row.update(params)
        queries.append(row)
    return queries


def _planner_queries_reply(trace: list[dict[str, Any]]) -> str | None:
    for event in reversed(trace):
        if (
            event.get("kind") == "node"
            and event.get("name") == "planner"
            and event.get("phase") == "update"
        ):
            update = event.get("update") or {}
            if "fits" in update:
                body = {
                    "fits": update.get("fits"),
                    "rejected": update.get("rejected"),
                    "rejected_count": update.get("rejected_count"),
                    "queries": update.get("queries"),
                }
                for key in ("error", "skipped", "open_slots_query"):
                    if update.get(key) is not None:
                        body[key] = update[key]
                return json.dumps(body, ensure_ascii=False, indent=2, default=str)
            queries = update.get("queries")
            if queries is None:
                return None
            return json.dumps(queries, ensure_ascii=False, indent=2, default=str)
    return None


def _last_ai_reply(
    messages: list[BaseMessage],
    *,
    stop_after: HeavyThrough = "recommender",
) -> str:
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


def _empty_node_row(name: str) -> dict[str, Any]:
    return {
        "node": name,
        "latency_ms": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "embed_tokens": 0,
        "cost_usd": 0.0,
    }


def _compute_trace_summary(
    trace: list[dict[str, Any]],
    *,
    total_latency_ms: float | None = None,
) -> dict[str, Any]:
    by_node: dict[str, dict[str, Any]] = {}
    path: list[str] = []

    def bucket(name: str | None) -> dict[str, Any]:
        key = name or "unknown"
        if key not in by_node:
            by_node[key] = _empty_node_row(key)
            path.append(key)
        return by_node[key]

    for event in trace:
        kind = event.get("kind")
        if kind == "summary":
            continue
        if kind == "node":
            name = event.get("name")
            if not name:
                continue
            row = bucket(name)
            if event.get("phase") == "update" and event.get("latency_ms") is not None:
                row["latency_ms"] += float(event["latency_ms"])
        elif kind == "llm_end":
            usage = event.get("usage") or {}
            prompt = _int_or_zero(usage.get("prompt_tokens"))
            completion = _int_or_zero(usage.get("completion_tokens"))
            row = bucket(event.get("node"))
            row["prompt_tokens"] += prompt
            row["completion_tokens"] += completion
            row["cost_usd"] += float(
                event.get("cost_usd")
                or _chat_cost_usd(event.get("model"), prompt, completion)
            )
        elif kind == "embed":
            tokens = _int_or_zero(event.get("prompt_tokens"))
            row = bucket(event.get("node") or "planner")
            row["embed_tokens"] += tokens
            row["cost_usd"] += float(event.get("cost_usd") or _embed_cost_usd(tokens))

    rows = [by_node[name] for name in path if name in by_node]
    if total_latency_ms is None:
        total_latency_ms = sum(r["latency_ms"] for r in rows)
    return {
        "latency_ms": total_latency_ms,
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "completion_tokens": sum(r["completion_tokens"] for r in rows),
        "embed_tokens": sum(r["embed_tokens"] for r in rows),
        "cost_usd": sum(r["cost_usd"] for r in rows),
        "by_node": rows,
    }


def _trace_summary(trace: list[dict[str, Any]]) -> dict[str, Any]:
    existing = next((e for e in trace if e.get("kind") == "summary"), None)
    if existing and "by_node" in existing:
        return existing
    return _compute_trace_summary(trace)


def _render_trace_metrics(trace: list[dict[str, Any]]) -> None:
    summary = _trace_summary(trace)
    nodes = [
        e["name"]
        for e in trace
        if e.get("kind") == "node" and e.get("phase") != "update"
    ]
    if not nodes:
        nodes = [e["name"] for e in trace if e.get("kind") == "node"]
    seen: set[str] = set()
    path: list[str] = []
    for name in nodes:
        if name not in seen:
            seen.add(name)
            path.append(name)
    if path:
        st.markdown("**Nodes:** " + " → ".join(f"`{n}`" for n in path))

    with st.container(horizontal=True):
        st.metric("Latency", _format_latency(summary.get("latency_ms")), border=True)
        st.metric("Cost", f"${float(summary.get('cost_usd') or 0):.5f}", border=True)
        st.metric(
            "Tokens in",
            f"{int(summary.get('prompt_tokens') or 0):,}",
            border=True,
        )
        st.metric(
            "Tokens out",
            f"{int(summary.get('completion_tokens') or 0):,}",
            border=True,
        )
        embed = int(summary.get("embed_tokens") or 0)
        if embed:
            st.metric("Embed tokens", f"{embed:,}", border=True)

    rows = summary.get("by_node") or []
    if rows:
        table = [
            {
                "Node": row["node"],
                "Latency (s)": round(float(row["latency_ms"]) / 1000, 3),
                "Prompt": row["prompt_tokens"],
                "Completion": row["completion_tokens"],
                "Embed": row["embed_tokens"],
                "Cost (USD)": row["cost_usd"],
            }
            for row in rows
        ]
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "Latency (s)": st.column_config.NumberColumn(
                    "Latency (s)", format="%.2f"
                ),
                "Prompt": st.column_config.NumberColumn("Prompt", format="%d"),
                "Completion": st.column_config.NumberColumn(
                    "Completion", format="%d"
                ),
                "Embed": st.column_config.NumberColumn("Embed", format="%d"),
                "Cost (USD)": st.column_config.NumberColumn(
                    "Cost (USD)", format="$%.5f"
                ),
            },
        )


def _render_trace(trace: list[dict[str, Any]]) -> None:
    _render_trace_metrics(trace)

    for i, event in enumerate(trace):
        kind = event.get("kind")
        if kind == "summary":
            continue
        if kind == "node":
            phase = event.get("phase", "update")
            title = f"{i + 1}. Node `{event['name']}`"
            latency = _format_latency(event.get("latency_ms"))
            if phase == "start":
                incoming = event.get("input") or []
                if incoming:
                    with st.expander(f"{title} _(enter)_", expanded=True):
                        for msg in incoming:
                            label = msg.get("type") or "message"
                            role = msg.get("role")
                            if role:
                                label = f"{label} ({role})"
                            st.markdown(f"**{label}**")
                            st.code(
                                _truncate(msg.get("content") or ""),
                                language=None,
                            )
                else:
                    st.markdown(f"**{title}** _(enter)_")
                continue
            update = event.get("update")
            is_queries = (
                event.get("name") == "planner"
                and isinstance(update, dict)
                and "queries" in update
            )
            label = "queries" if is_queries else "state update"
            if is_queries and isinstance(update, dict) and "fits" in update:
                label = "fits"
            with st.expander(
                f"{title} · {label} · {latency}",
                expanded=is_queries,
            ):
                _json_block(update)
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
            usage = event.get("usage") or {}
            bits = [
                _format_latency(event.get("latency_ms")),
                f"{_int_or_zero(usage.get('prompt_tokens'))} in",
                f"{_int_or_zero(usage.get('completion_tokens'))} out",
            ]
            if event.get("cost_usd") is not None:
                bits.append(f"${float(event['cost_usd']):.5f}")
            with st.expander(
                f"{i + 1}. LLM response · {' · '.join(bits)}",
                expanded=False,
            ):
                st.code(
                    _truncate(_content_to_str(event.get("response"))),
                    language=None,
                )
        elif kind == "tool":
            latency = _format_latency(event.get("latency_ms"))
            with st.expander(
                f"{i + 1}. Tool `{event.get('name')}` · {latency}",
                expanded=True,
            ):
                st.markdown("**Params**")
                _json_block(event.get("params"))
                st.markdown("**Result**")
                _json_block(event.get("result"))
        elif kind == "embed":
            tokens = _int_or_zero(event.get("prompt_tokens"))
            with st.expander(
                f"{i + 1}. Embed query · {_format_latency(event.get('latency_ms'))} · "
                f"{tokens} tokens · ${float(event.get('cost_usd') or 0):.5f}",
                expanded=False,
            ):
                _json_block(
                    {
                        "prompt_tokens": tokens,
                        "latency_ms": event.get("latency_ms"),
                        "cost_usd": event.get("cost_usd"),
                    }
                )
        else:
            with st.expander(f"{i + 1}. {kind}", expanded=False):
                _json_block(event)

def invoke_agent(
    user_text: str,
    *,
    stop_after: HeavyThrough,
) -> tuple[str, list[dict[str, Any]]]:
    """Same contract as main.telegram_webhook, plus a LangGraph turn trace."""
    global _current_trace

    compiled = build_graph(stop_after=stop_after)
    history: list[BaseMessage] = list(st.session_state.graph_messages)
    history.append(HumanMessage(content=user_text))
    state: ChatState = {"messages": history}

    trace: list[dict[str, Any]] = []
    _current_trace = trace
    handler = TraceCallbackHandler()
    config = {"callbacks": [handler]}

    final_messages: list[BaseMessage] | None = None
    turn_started = time.perf_counter()
    try:
        for mode, chunk in compiled.stream(
            state,
            config=config,
            stream_mode=["updates", "values"],
        ):
            if mode == "updates" and isinstance(chunk, dict):
                for node_name, update in chunk.items():
                    if node_name == "planner" and isinstance(update, dict):
                        queries = _tool_queries_since_node_start(trace, "planner")
                        fits_payload: dict[str, Any] | None = None
                        for msg in update.get("messages") or []:
                            raw = _content_to_str(getattr(msg, "content", ""))
                            try:
                                data = json.loads(raw)
                            except json.JSONDecodeError:
                                continue
                            if isinstance(data, dict) and "fits" in data:
                                fits_payload = data
                                break
                        serialized_update = {"queries": queries}
                        if fits_payload is not None:
                            serialized_update["fits"] = fits_payload.get("fits")
                            serialized_update["rejected"] = fits_payload.get(
                                "rejected"
                            )
                            serialized_update["rejected_count"] = (
                                fits_payload.get("rejected_count")
                            )
                            for key in (
                                "error",
                                "skipped",
                                "constraints",
                                "open_slots_query",
                            ):
                                if fits_payload.get(key) is not None:
                                    serialized_update[key] = fits_payload[key]
                    elif isinstance(update, dict) and "messages" in update:
                        serialized_update = {
                            "messages": _serialize_messages(update["messages"])
                        }
                    else:
                        serialized_update = update
                    started_stack = handler.node_started_at.get(node_name) or []
                    started = started_stack.pop(0) if started_stack else None
                    latency_ms = (
                        (time.perf_counter() - started) * 1000
                        if started is not None
                        else None
                    )
                    trace.append(
                        {
                            "kind": "node",
                            "name": node_name,
                            "phase": "update",
                            "update": serialized_update,
                            "latency_ms": latency_ms,
                        }
                    )
            elif mode == "values" and isinstance(chunk, dict):
                final_messages = chunk.get("messages")
    finally:
        _current_trace = None

    if final_messages is None:
        result = compiled.invoke(state, config=config)
        final_messages = result["messages"]

    total_ms = (time.perf_counter() - turn_started) * 1000
    summary = _compute_trace_summary(trace, total_latency_ms=total_ms)
    trace.append({"kind": "summary", **summary})

    st.session_state.graph_messages = final_messages
    if stop_after == "planner":
        reply = _planner_queries_reply(trace)
        if reply:
            return reply, trace
    return _last_ai_reply(final_messages, stop_after=stop_after), trace


_init_session()

st.title("Trippy agent")
st.caption(
    f"Local Streamlit client · `{AGENT_CHAT_MODEL}` via Nebius · "
    "production remains Telegram"
)

with st.sidebar:
    st.header("Session")
    stop_after: HeavyThrough = (
        st.radio(
            "Heavy path",
            options=list(HEAVY_PATH_LABELS.keys()),
            format_func=lambda key: HEAVY_PATH_LABELS[key],
            captions=[
                "Constraints JSON",
                "Plus search / RAG tools",
                "Plus recommendation reply",
            ],
            key="heavy_path",
            on_change=_reset_conversation,
            help=(
                "Light router always runs first. "
                "Changing this clears the conversation so node outputs do not mix."
            ),
            width="stretch",
        )
        or "extractor"
    )
    st.caption("Telegram still uses the full path.")
    if st.button("Reset conversation", width="stretch"):
        _reset_conversation()
        st.rerun()

    st.divider()
    st.subheader("MCP prompt")
    st.caption("streamlit-mcp cannot drive chat_input. Send from here.")
    with st.form("agent_prompt_form", clear_on_submit=True, border=False):
        mcp_text = st.text_area(
            "Prompt",
            key="agent_prompt",
            placeholder="Ask about campsites…",
            height=80,
        )
        agent_send = st.form_submit_button(
            "Send prompt",
            key="agent_send",
            icon=":material/send:",
            width="stretch",
        )
    mcp_prompt = (mcp_text or "").strip() if agent_send else ""

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
            width="stretch",
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
            width="stretch",
        )
    else:
        st.info("Send a message to start a conversation.")

for turn in st.session_state.display:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("role") == "assistant" and turn.get("trace"):
            with st.expander("LangGraph trace", expanded=False):
                _render_trace(turn["trace"])

prompt = st.chat_input("Ask about campsites…") or mcp_prompt
if prompt:
    st.session_state.display.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                reply, trace = invoke_agent(prompt, stop_after=stop_after)
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
