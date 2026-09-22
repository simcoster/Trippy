"""
Local Streamlit harness for the LangGraph agent.

Mirrors a chat turn (HumanMessage → graph.stream → last AIMessage).

Also records a per-turn LangGraph trace: nodes, LLM prompts/responses,
tool calls (params + returns), token cost, and latency.

Run from repo root:
  uv run python -m streamlit run scripts/streamlit_chat.py

`TRIPPY_PUBLIC_UI=1` hides traces, MCP, and the heavy-path selector (cloud).

streamlit-mcp cannot drive st.chat_input. The sidebar "MCP prompt" form is
driveable (text_area + submit).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

# Repo root on sys.path so `source.*` imports work under `streamlit run`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Patch ssl.create_default_context before LangChain imports (Windows OpenSSL).
import source.scraper.tls as _tls  # noqa: F401
from source.price_sandbox.client import require_healthy_sandbox

# Suppress Pydantic V1 compatibility warning with Python 3.14+
warnings.filterwarnings("ignore", message=".*Pydantic V1.*", category=UserWarning)

import streamlit as st
from dotenv import dotenv_values, load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
)
from langchain_core.outputs import LLMResult

load_dotenv(_ROOT / ".env")

require_healthy_sandbox()

# Read from the file on every Streamlit rerun. load_dotenv() does not
# override a value already in os.environ, so editing .env used to no-op.
_PUBLIC_UI = (
    (
        dotenv_values(_ROOT / ".env").get("TRIPPY_PUBLIC_UI")
        or os.environ.get("TRIPPY_PUBLIC_UI")
        or ""
    )
    .strip()
    .casefold()
    in {
        "1",
        "true",
        "yes",
    }
)
print(f"trippy public_ui={int(_PUBLIC_UI)}", flush=True)

import importlib

try:
    import db.connect as _db_connect

    if not hasattr(_db_connect, "DatabaseUnavailable"):
        _db_connect = importlib.reload(_db_connect)
    DatabaseUnavailable = _db_connect.DatabaseUnavailable
    ping = _db_connect.ping
except Exception:
    traceback.print_exc()
    print("error: failed to load db.connect", flush=True)

    class DatabaseUnavailable(Exception):
        """Fallback if db.connect did not import (stale Streamlit module)."""

    def ping(**_kwargs: Any) -> None:
        raise DatabaseUnavailable("db.connect failed to load")

import source.agent.recommender.timing as _recommender_timing

if not hasattr(_recommender_timing, "last_recommend_timing"):
    importlib.reload(_recommender_timing)

import source.agent.graph as agent_graph
import source.demo_quota as _demo_quota
from source.agent.graph import AGENT_CHAT_MODEL, ChatState, HeavyThrough, build_graph
from source.agent.keepalive import ping_new_session
from source.agent.recommender.recommend import listen_recommend_text
from source.agent.recommender.timing import last_recommend_timing
from source.agent.search import amenities, availability, campsites, claims, embed, rules
from source.agent.timing import collect_stages, format_stages
from source.agent.tracing import (
    agent_run_config,
    configure_agent_tracing,
    project_name,
    tracing_configured,
)
from source.agent.turn_status import SEARCHING, set_turn_status
from source.scraper.amenity_enrichment.llm import (
    EmbeddingLLMClient,
    LlmUsage,
    chat_usd_per_mtok,
    collect_llm_usage,
)

importlib.reload(_demo_quota)
QUOTA_USED = _demo_quota.QUOTA_USED
claim_query = _demo_quota.claim_query
public_visitor_hash = _demo_quota.public_visitor_hash
quota_remaining = _demo_quota.quota_remaining
remaining_caption = _demo_quota.remaining_caption

HEAVY_PATH_LABELS: dict[HeavyThrough, str] = {
    "extractor": "Extractor only",
    "planner": "Extractor + planner",
    "recommender": "Extractor + planner + recommender",
}

st.set_page_config(
    page_title="Trippy camping" if _PUBLIC_UI else "Trippy camping (local)",
    page_icon="⛺",
    layout="wide",
)
# Streamlit binds "c" to Clear cache; Ctrl+C in the browser opens that dialog.
st.set_option("client.toolbarMode", "viewer")
# A Hebrew paragraph starts on the right; an English one stays on the left.
# plaintext takes the direction from the first strong letter in that block.
st.markdown(
    """
<style>
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li {
    unicode-bidi: plaintext;
    text-align: start;
}
.trippy-questions-left {
    font-size: 1.15rem;
    line-height: 1.3;
    margin: 0 0 0.75rem 0;
}
.trippy-questions-left span {
    font-size: 1.45rem;
    font-weight: 700;
    color: #ff4b4b;
}
.st-key-reset_chat button {
    background-color: #21c354 !important;
    border-color: #21c354 !important;
}
.st-key-reset_chat button:hover,
.st-key-reset_chat button:focus {
    background-color: #1a9e43 !important;
    border-color: #1a9e43 !important;
    color: #fff !important;
}
[data-testid="stSidebarContent"] {
    display: flex;
    flex-direction: column;
}
[data-testid="stSidebarUserContent"] {
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
    padding-bottom: 1rem !important;
}
[data-testid="stSidebarUserContent"] > div {
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
}
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] {
    flex: 1 1 auto;
}
.st-key-github_readme {
    margin-top: auto;
}
.st-key-github_readme a {
    min-height: 4.5rem;
    padding-top: 1.15rem;
    padding-bottom: 1.15rem;
    font-size: 1.2rem;
    font-weight: 650;
}
[data-testid="stMainBlockContainer"] {
    position: relative;
}
[data-testid="stMainBlockContainer"] h1 {
    padding-right: 7rem;
}
.st-key-github_readme_top {
    position: absolute;
    top: 8rem;
    right: 1rem;
    width: auto !important;
    z-index: 2;
}
@media (min-width: calc(736px + 8rem)) {
    .st-key-github_readme_top {
        right: 5rem;
    }
}
.st-key-github_readme_top a {
    white-space: nowrap;
    width: auto;
    min-height: 3.625rem;
    padding: 0 0.9rem;
    font-size: 1rem;
    line-height: 1.2;
}
</style>
""",
    unsafe_allow_html=True,
)
if configure_agent_tracing():
    print(f"langsmith tracing project={project_name()}", flush=True)

_USER_ERROR = "Something went wrong."
logger = logging.getLogger("trippy.streamlit")


def _report_error(exc: BaseException) -> str:
    """Log + print the real error; UI only gets a generic line."""
    logger.exception("%s", exc)
    traceback.print_exc()
    print(f"error: {type(exc).__name__}: {exc}", flush=True)
    return _USER_ERROR


@st.cache_data(ttl=8, show_spinner=False)
def _cached_postgres_error() -> str:
    try:
        ping()
        return ""
    except Exception as exc:
        _report_error(exc)
        return _USER_ERROR


_db_error = _cached_postgres_error()

# Active turn trace (set while invoke_agent runs)
_current_trace: list[dict[str, Any]] | None = None
_turn_t0: float | None = None
_progress_ui: Any = None


def _turn_log(msg: str) -> None:
    """Print live stage lines; update the assistant placeholder when local."""
    elapsed = (
        f"+{time.perf_counter() - _turn_t0:.1f}s"
        if _turn_t0 is not None
        else ""
    )
    line = f"turn {elapsed} {msg}".strip()
    print(line, flush=True)
    box = _progress_ui
    if box is None:
        return
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx(suppress_warning=True) is None:
            return
        box.caption(line)
    except Exception:
        return


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
        _turn_log(f"{node} …")

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
        node = metadata.get("langgraph_node")
        _current_trace.append(
            {
                "kind": "llm_start",
                "run_id": str(run_id),
                "node": node,
                "model": model,
                "prompt": _format_prompt_messages(messages),
            }
        )
        _turn_log(f"{node or '?'} LLM {model} …")

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
        if latency_ms is not None:
            _turn_log(
                f"{node or '?'} LLM done {latency_ms / 1000:.1f}s"
            )

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = (
            kwargs.get("name")
            or (serialized or {}).get("name")
            or "tool"
        )
        _turn_log(f"{name} …")

def _install_tool_hooks() -> None:
    """Wrap imperative tool functions so Streamlit can log params/returns."""
    if getattr(agent_graph, "_trippy_streamlit_hooks", False):
        return

    def _wrap(name: str, fn: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if _current_trace is not None:
                _turn_log(f"{name} …")
            started = time.perf_counter()
            result = fn(*args, **kwargs)
            latency_ms = (time.perf_counter() - started) * 1000
            if _current_trace is not None:
                _turn_log(f"{name} {latency_ms / 1000:.1f}s")
            if _current_trace is not None:
                if name in (
                    "search_claims",
                    "search_review_claims",
                    "search_site_amenities",
                    "search_stated_amenities",
                    "search_campsite_rules",
                ):
                    params = {
                        "query": args[0] if args else kwargs.get("query"),
                        "limit": args[1] if len(args) > 1 else kwargs.get("limit", 5),
                    }
                    if name == "search_stated_amenities":
                        params["accommodation_type_ids"] = kwargs.get(
                            "accommodation_type_ids"
                        )
                    elif name in (
                        "search_site_amenities",
                        "search_review_claims",
                        "search_campsite_rules",
                    ):
                        params["campsite_ids"] = kwargs.get("campsite_ids")
                elif name == "lookup_campsite_by_name":
                    params = {"name": args[0] if args else kwargs.get("name")}
                elif name == "search_open_slots":
                    params = {
                        "date_range": kwargs.get("date_range"),
                        "site_id": kwargs.get("site_id"),
                        "party_size": kwargs.get("party_size"),
                        "numeric_constraints": kwargs.get("numeric_constraints"),
                    }
                    last = getattr(availability, "_LAST_OPEN_SLOTS_QUERY", None)
                    if isinstance(last, dict):
                        params.update(last)
                    sandbox = params.pop("sandbox", None)
                    if isinstance(sandbox, dict):
                        _current_trace.append(
                            {
                                "kind": "sandbox",
                                "name": "price_sandbox_quote",
                                "url": sandbox.get("url"),
                                "skipped": sandbox.get("skipped"),
                                "calls": sandbox.get("calls") or [],
                                "latency_ms": sandbox.get("latency_ms"),
                            }
                        )
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

    claims.search_claims = _wrap("search_claims", claims.search_claims)
    claims.search_review_claims = _wrap(
        "search_review_claims", claims.search_review_claims
    )
    amenities.search_stated_amenities = _wrap(
        "search_stated_amenities", amenities.search_stated_amenities
    )
    amenities.search_site_amenities = _wrap(
        "search_site_amenities", amenities.search_site_amenities
    )
    rules.search_campsite_rules = _wrap(
        "search_campsite_rules", rules.search_campsite_rules
    )
    campsites.lookup_campsite_by_name = _wrap(
        "lookup_campsite_by_name", campsites.lookup_campsite_by_name
    )
    availability.search_open_slots = _wrap(
        "search_open_slots", availability.search_open_slots
    )
    availability.search_availability = _wrap(
        "search_availability", availability.search_availability
    )
    campsites.search_campsites = _wrap(
        "search_campsites", campsites.search_campsites
    )
    agent_graph.search_claims = claims.search_claims
    agent_graph.search_review_claims = claims.search_review_claims
    agent_graph.search_stated_amenities = amenities.search_stated_amenities
    agent_graph.search_site_amenities = amenities.search_site_amenities
    agent_graph.search_campsite_rules = rules.search_campsite_rules
    agent_graph.lookup_campsite_by_name = campsites.lookup_campsite_by_name
    agent_graph.search_open_slots = availability.search_open_slots
    agent_graph.search_availability = availability.search_availability
    agent_graph.search_campsites = campsites.search_campsites

    embedder = embed._claims_embedder
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
    if "langsmith_thread_id" not in st.session_state:
        st.session_state.langsmith_thread_id = str(uuid4())
    ping_new_session(st.session_state)


_CHAT_INPUT_KEY = "chat_prompt"
_EXAMPLE_PROMPT_KEY = "example_prompt"
_EXAMPLE_PROMPTS = (
    "אנחנו מחפשים מקום לשני מבוגרים בשבוע הבא בין שלישי לחמישי ללילה אחד, עם לפחות 2 שירותי נכים.",
    "we're looking for a place for 2 adults and 2 kids for one next week somewhere on Tuesday-Thursday , with pools for the kids and a fridge for up to 300 nis",
)


def _reset_conversation() -> None:
    st.session_state.graph_messages = []
    st.session_state.display = []
    st.session_state.langsmith_thread_id = str(uuid4())
    st.session_state.pop(_EXAMPLE_PROMPT_KEY, None)
    st.session_state.pop(_CHAT_INPUT_KEY, None)


def _apply_example_prompt() -> None:
    picked = st.session_state.get(_EXAMPLE_PROMPT_KEY)
    if isinstance(picked, str) and picked:
        st.session_state[_CHAT_INPUT_KEY] = picked


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


def _planner_queries_reply(trace: list[dict[str, Any]]) -> str | None:
    for event in reversed(trace):
        if (
            event.get("kind") == "node"
            and event.get("name") == "planner"
            and event.get("phase") == "update"
        ):
            update = event.get("update") or {}
            if "fits" not in update:
                return None
            body = {
                "fits": update.get("fits"),
                "rejected": update.get("rejected"),
                "rejected_count": update.get("rejected_count"),
            }
            for key in ("error", "skipped", "open_slots_query"):
                if update.get(key) is not None:
                    body[key] = update[key]
            return json.dumps(body, ensure_ascii=False, indent=2, default=str)
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
        st.metric(
            "TTFT",
            _format_latency(summary.get("turn_ttft_spoken_ms")),
            border=True,
        )
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
        judge_n = int(summary.get("judge_calls") or 0)
        if judge_n or summary.get("stages"):
            st.metric("Judge calls", f"{judge_n}", border=True)
    stages_line = format_stages(summary.get("stages"))
    if stages_line:
        st.caption(stages_line)
    rec_chunk = summary.get("recommend_ttft_chunk_ms")
    rec_spoken = summary.get("recommend_ttft_spoken_ms")
    if rec_chunk is not None or rec_spoken is not None:
        bits = []
        if rec_chunk is not None:
            bits.append(f"chunk {_format_latency(rec_chunk)}")
        if rec_spoken is not None:
            bits.append(f"spoken {_format_latency(rec_spoken)}")
        st.caption("Recommend TTFT: " + " · ".join(bits))
    reasoning_n = int(summary.get("recommend_reasoning_tokens") or 0)
    if reasoning_n or summary.get("recommend_thinking_stream"):
        st.warning(
            "Recommender thinking may be on "
            f"(reasoning_tokens={reasoning_n}, "
            f"thinking_stream={summary.get('recommend_thinking_stream')})."
        )

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
            is_fits = (
                event.get("name") == "planner"
                and isinstance(update, dict)
                and "fits" in update
            )
            label = "fits" if is_fits else "state update"
            with st.expander(
                f"{title} · {label} · {latency}",
                expanded=is_fits,
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
        elif kind == "sandbox":
            calls = event.get("calls") or []
            latency = _format_latency(event.get("latency_ms"))
            skipped = event.get("skipped")
            title = f"{i + 1}. Price sandbox · {len(calls)} call(s) · {latency}"
            if skipped:
                title += f" · skipped ({skipped})"
            with st.expander(title, expanded=True):
                st.markdown(f"**URL** `{event.get('url') or '(unset)'}`")
                if skipped:
                    st.warning(skipped)
                _json_block(calls)
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
    """Run one user turn through the compiled graph and return a LangGraph trace."""
    global _current_trace, _turn_t0

    compiled = build_graph(stop_after=stop_after)
    _turn_t0 = time.perf_counter()
    _turn_log("start")
    history: list[BaseMessage] = list(st.session_state.graph_messages)
    history.append(HumanMessage(content=user_text))
    state: ChatState = {"messages": history}

    trace: list[dict[str, Any]] = []
    _current_trace = trace
    handler = TraceCallbackHandler()
    config = agent_run_config(
        thread_id=st.session_state.langsmith_thread_id,
        channel="streamlit",
        user_text=user_text,
        extra_metadata={
            "public_ui": _PUBLIC_UI,
            "stop_after": stop_after,
        },
    )
    config["callbacks"] = [handler]

    final_messages: list[BaseMessage] | None = None
    turn_started = time.perf_counter()
    stages_snap: dict[str, dict[str, float | int]] | None = None
    judge_calls = 0
    try:
        with collect_stages() as clock, collect_llm_usage() as usage:
            for mode, chunk in compiled.stream(
                state,
                config=config,
                stream_mode=["updates", "values"],
            ):
                if mode == "updates" and isinstance(chunk, dict):
                    for node_name, update in chunk.items():
                        if node_name == "planner" and isinstance(update, dict):
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
                            if fits_payload is not None:
                                serialized_update = {
                                    "fits": fits_payload.get("fits"),
                                    "rejected": fits_payload.get("rejected"),
                                    "rejected_count": fits_payload.get(
                                        "rejected_count"
                                    ),
                                }
                                for key in (
                                    "error",
                                    "skipped",
                                    "constraints",
                                    "open_slots_query",
                                ):
                                    if fits_payload.get(key) is not None:
                                        serialized_update[key] = fits_payload[key]
                            elif "messages" in update:
                                serialized_update = {
                                    "messages": _serialize_messages(
                                        update["messages"]
                                    )
                                }
                            else:
                                serialized_update = update
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
                        if latency_ms is not None:
                            _turn_log(
                                f"{node_name} done {latency_ms / 1000:.1f}s"
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
            if final_messages is None:
                result = compiled.invoke(state, config=config)
                final_messages = result["messages"]
            stages_snap = clock.snapshot()
            judge_calls = sum(
                int(b.calls)
                for b in usage.by_role()
                if b.role == "claim_judge"
            )
    finally:
        _current_trace = None

    total_ms = (time.perf_counter() - turn_started) * 1000
    summary = _compute_trace_summary(trace, total_latency_ms=total_ms)
    if stages_snap is not None:
        summary["stages"] = stages_snap
        summary["judge_calls"] = judge_calls
        stages_line = format_stages(stages_snap)
        print(
            f"judge={judge_calls} {stages_line} wall={total_ms / 1000:.1f}s",
            flush=True,
        )
    timing = last_recommend_timing()
    if timing:
        summary["recommend_ttft_chunk_ms"] = timing.get("chunk_ms")
        summary["recommend_ttft_spoken_ms"] = timing.get("spoken_ms")
        summary["recommend_elapsed_ms"] = timing.get("total_ms")
        summary["recommend_reasoning_tokens"] = timing.get("reasoning_tokens")
        summary["recommend_thinking_stream"] = timing.get("thinking_stream")
        chunk_s = timing.get("chunk_ms")
        spoken_s = timing.get("spoken_ms")
        print(
            "recommend "
            f"ttft_chunk={None if chunk_s is None else f'{float(chunk_s) / 1000:.1f}s'} "
            f"ttft_spoken={None if spoken_s is None else f'{float(spoken_s) / 1000:.1f}s'} "
            f"reasoning={timing.get('reasoning_tokens')} "
            f"thinking_stream={timing.get('thinking_stream')} "
            f"empty_prefix={timing.get('empty_prefix')}",
            flush=True,
        )
        before_ms = 0.0
        for event in trace:
            if event.get("kind") != "node" or event.get("phase") != "update":
                continue
            if event.get("name") == "recommender":
                break
            if event.get("latency_ms") is not None:
                before_ms += float(event["latency_ms"])
        spoken = timing.get("spoken_ms")
        if spoken is not None:
            summary["turn_ttft_spoken_ms"] = before_ms + float(spoken)
        chunk = timing.get("chunk_ms")
        if chunk is not None:
            summary["turn_ttft_chunk_ms"] = before_ms + float(chunk)
    else:
        print("recommend timing missing", flush=True)
    trace.append({"kind": "summary", **summary})

    st.session_state.graph_messages = final_messages
    if stop_after == "planner":
        reply = _planner_queries_reply(trace)
        if reply:
            return reply, trace
    return _last_ai_reply(final_messages, stop_after=stop_after), trace


_init_session()

_answered = any(turn.get("role") == "assistant" for turn in st.session_state.display)
_ASK_PLACEHOLDER = (
    "Ask me about a camping stay | שאל אותי על שהייה באתרי קמפינג"
)

def _show_questions_left(remaining: int) -> None:
    count, _, rest = remaining_caption(remaining).partition(" ")
    st.markdown(
        f"<p class='trippy-questions-left'><span>{count}</span> {rest}</p>",
        unsafe_allow_html=True,
    )


_quota_open = True
_visitor_hash: str | None = None
_quota_left: int | None = None
if _PUBLIC_UI and not _db_error:
    _visitor_hash = public_visitor_hash(st.context.headers)
    if _visitor_hash is None:
        _quota_open = False
    else:
        try:
            _quota_left = quota_remaining(_visitor_hash)
        except Exception as exc:
            _report_error(exc)
            _visitor_hash = None
            _quota_open = False
        else:
            _quota_open = _quota_left > 0

st.link_button(
    "README",
    "https://github.com/simcoster/Trippy",
    icon=":material/menu_book:",
    key="github_readme_top",
)
st.title("Trippy camping ⛺" if _PUBLIC_UI else "Trippy camping ⛺ (local)")
if _db_error:
    st.error(_db_error)
elif _PUBLIC_UI and _visitor_hash is None:
    st.error(_USER_ERROR)
if _quota_left is not None:
    _show_questions_left(_quota_left)
elif not _PUBLIC_UI:
    st.caption(
        f"Local Streamlit client · `{AGENT_CHAT_MODEL}` via Nebius · "
        f"`{(os.environ.get('TRIPPY_SCHEMA') or 'public')}`."
        f"`{(os.environ.get('TRIPPY_AVAILABILITY_TABLE') or 'availability')}`"
    )

mcp_prompt = ""
stop_after: HeavyThrough = "recommender"
with st.sidebar:
    st.header("Session")
    if _quota_left is not None:
        _show_questions_left(_quota_left)
    if not _PUBLIC_UI:
        stop_after = (
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
        st.caption("Local harness: traces stay in this sidebar.")
        if tracing_configured():
            st.caption(f"LangSmith project `{project_name()}`.")
        else:
            st.caption("LangSmith off — set `LANGSMITH_API_KEY` to record turns.")

    if not _PUBLIC_UI:
        st.divider()
        st.subheader("MCP prompt")
        st.caption("streamlit-mcp cannot drive chat_input. Send from here.")
        with st.form("agent_prompt_form", clear_on_submit=True, border=False):
            mcp_text = st.text_area(
                "Prompt",
                key="agent_prompt",
                placeholder=_ASK_PLACEHOLDER,
                height=80,
                disabled=_answered,
            )
            agent_send = st.form_submit_button(
                "Send prompt",
                key="agent_send",
                icon=":material/send:",
                width="stretch",
                disabled=_answered,
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

    st.link_button(
        "GitHub README",
        "https://github.com/simcoster/Trippy",
        icon=":material/menu_book:",
        width="stretch",
        key="github_readme",
    )

for turn in st.session_state.display:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if (
            not _PUBLIC_UI
            and turn.get("role") == "assistant"
            and turn.get("trace")
        ):
            with st.expander("LangGraph trace", expanded=False):
                _render_trace(turn["trace"])

_can_ask = not _answered and _quota_open
with st.bottom:
    if _answered:
        if st.button("Try another question!", type="primary", width="stretch", key="reset_chat"):
            _reset_conversation()
            st.rerun()
    elif _can_ask:
        st.selectbox(
            "Example prompts",
            _EXAMPLE_PROMPTS,
            index=None,
            placeholder="Example prompts",
            label_visibility="collapsed",
            key=_EXAMPLE_PROMPT_KEY,
            on_change=_apply_example_prompt,
        )

submitted = (
    st.chat_input(_ASK_PLACEHOLDER, key=_CHAT_INPUT_KEY) if _can_ask else None
)
prompt = submitted or mcp_prompt
if prompt:
    st.session_state.display.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    _blocked: str | None = None
    if _PUBLIC_UI and not _db_error:
        if _visitor_hash is None:
            _blocked = _USER_ERROR
        elif (_quota_left or 0) <= 0:
            _blocked = QUOTA_USED
        else:
            try:
                _decision = claim_query(_visitor_hash)
            except Exception as exc:
                _blocked = _report_error(exc)
            else:
                if not _decision.allowed:
                    _blocked = QUOTA_USED
    if _blocked is not None:
        with st.chat_message("assistant"):
            st.markdown(_blocked)
        st.session_state.display.append(
            {"role": "assistant", "content": _blocked, "trace": []}
        )
        st.rerun()

    with st.chat_message("assistant"):
        phase = (
            st.status(SEARCHING, expanded=False, state="running")
            if stop_after != "extractor"
            else None
        )
        reply_box = st.empty()
        if not _PUBLIC_UI:
            _progress_ui = st.empty()

        def _show_phase(text: str) -> None:
            if phase is not None:
                phase.update(label=text, state="running")

        set_turn_status(_show_phase if phase is not None else None)
        failed = False
        try:
            try:
                if _db_error:
                    reply, trace = _USER_ERROR, []
                else:
                    with listen_recommend_text(reply_box.markdown):
                        reply, trace = invoke_agent(
                            prompt, stop_after=stop_after
                        )
            except Exception as e:
                reply = _report_error(e)
                trace = []
                failed = True
                if phase is not None:
                    phase.update(state="error")
        finally:
            set_turn_status(None)
            _progress_ui = None
            if phase is not None and not failed:
                phase.update(state="complete")
        reply_box.markdown(reply)
        if trace and not _PUBLIC_UI:
            with st.expander("LangGraph trace", expanded=True):
                _render_trace(trace)
    st.session_state.display.append(
        {"role": "assistant", "content": reply, "trace": trace}
    )
    st.rerun()
