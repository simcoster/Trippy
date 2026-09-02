"""LangGraph nodes and wiring. Search, dates, and planner logic live elsewhere."""

from __future__ import annotations

import json
import logging
import warnings
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from source.agent.constraints import (
    EMPTY_CONSTRAINTS,
    attach_campsite,
    compact_date_intent,
    constraints_from_tool_calls,
    intent_nonempty,
    latest_constraints_json,
    normalize_constraints,
    parse_constraints_dict,
    today_il,
)
from source.agent.dates import intent_tool_args, resolve_dates_tool
from source.agent.messages import is_keep_decision, latest_user_text, message_text
from source.agent.planner import _semantic_evidence_payload, planner_fits_payload
from source.agent.prompts import (
    EMPTY_REPLY_FALLBACK,
    NOT_TRIP_REPLY,
    RECOMMENDER_SYSTEM_PROMPT,
    TRIVIAL_PATTERNS,
    format_cleaning_prompt,
    format_extractor_system_prompt,
)
from source.agent.search import (
    _claims_embedder,
    _open_slots_sql,
    _price_matches,
    _price_per_night_constraint,
    _query_vec_literal,
    _rate_period_for_stay,
    _render_sql,
    lookup_campsite_by_name,
    search_availability,
    search_campsites,
    search_claims,
    search_open_slots,
    search_review_claims,
    search_site_amenities,
    search_stated_amenities,
)
from source.scraper.amenity_enrichment.llm import (
    QWEN_INSTRUCT_30B_MODEL,
    QWEN_INSTRUCT_MODEL,
    make_agent_chat_model,
)

__all__ = [
    "_claims_embedder",
    "_open_slots_sql",
    "_price_matches",
    "_price_per_night_constraint",
    "_query_vec_literal",
    "_rate_period_for_stay",
    "_render_sql",
    "_semantic_evidence_payload",
    "lookup_campsite_by_name",
    "planner_fits_payload",
    "search_availability",
    "search_campsites",
    "search_claims",
    "search_open_slots",
    "search_review_claims",
    "search_site_amenities",
    "search_stated_amenities",
]

load_dotenv()

# Suppress Pydantic V1 compatibility warning with Python 3.14+
warnings.filterwarnings("ignore", message=".*Pydantic V1.*", category=UserWarning)

logger = logging.getLogger(__name__)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Re-exports so Streamlit wraps and older tests keep `source.agent.graph.*`.
# Planner calls `source.agent.search` directly — wrap/patch that module.

# Do NOT bind_tools on the chat models: extractor/recommender need text/JSON
# replies. Tools are invoked imperatively in planner_node. Binding tools made
# Qwen return empty content + tool_calls, which surfaced as blank agent replies.
light_model = make_agent_chat_model(temperature=0.7)
heavy_model = make_agent_chat_model(temperature=0.7)
planner_model = make_agent_chat_model(
    temperature=0, model=QWEN_INSTRUCT_30B_MODEL
)
AGENT_CHAT_MODEL = QWEN_INSTRUCT_MODEL
PLANNER_CHAT_MODEL = QWEN_INSTRUCT_30B_MODEL


def router(state: ChatState) -> str:
    """Decide if the message is trivial (like "thanks!") or needs processing."""
    last_message = message_text(state["messages"][-1].content).lower().strip()

    if any(pattern in last_message for pattern in TRIVIAL_PATTERNS) and len(last_message) < 50:
        return "trivial"
    return "non_trivial"


def light_node(state: ChatState) -> ChatState:
    """
    Light model handles:
    - Trivial prompts: answer directly
    - Non-trivial prompts: clean message to extract only trip-planning related content
    """
    last_message = state["messages"][-1]
    last_content = message_text(last_message.content)

    is_trivial = (
        any(pattern in last_content.lower() for pattern in TRIVIAL_PATTERNS)
        and len(last_content) < 50
    )

    if is_trivial:
        response = light_model.invoke(state["messages"])
        text = message_text(response.content).strip() or EMPTY_REPLY_FALLBACK
        return {"messages": [AIMessage(content=text)]}

    conversation_context = "\n".join(
        [
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {message_text(msg.content)}"
            for msg in state["messages"][:-1]
        ]
    )

    cleaning_prompt = format_cleaning_prompt(
        conversation_context=conversation_context,
        last_content=last_content,
    )
    cleaning_messages = state["messages"][:-1] + [HumanMessage(content=cleaning_prompt)]
    cleaned_response = light_model.invoke(cleaning_messages)
    cleaned_content = message_text(cleaned_response.content)

    if is_keep_decision(cleaned_content):
        return {}
    return {"messages": [AIMessage(content=NOT_TRIP_REPLY)]}


def check_after_cleaning(state: ChatState) -> str:
    """Check if there's still content after cleaning that needs heavy model."""
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage):
        return "end"

    if isinstance(last_message, HumanMessage) and message_text(last_message.content).strip():
        return "heavy"
    return "end"


def extractor_node(state: ChatState) -> ChatState:
    """Extract constraint *intent*, then resolve dates via resolve_dates."""
    today = today_il()
    system_msg = SystemMessage(content=format_extractor_system_prompt(today))

    response = planner_model.invoke([system_msg] + state["messages"])
    raw = message_text(response.content)
    tool_calls = getattr(response, "tool_calls", None) or []
    user_text = latest_user_text(state["messages"])
    parsed = parse_constraints_dict(raw)

    for tc in tool_calls or []:
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
        if name == "resolve_dates" and isinstance(args, dict):
            parsed["date_intent"] = {**(parsed.get("date_intent") or {}), **args}

    intent = parsed.get("date_intent")
    if not intent_nonempty(intent):
        intent = {}

    compact = compact_date_intent(intent) if isinstance(intent, dict) else {}
    resolved = None
    tool_args = intent_tool_args(intent) if isinstance(intent, dict) else {}
    if tool_args:
        resolved = resolve_dates_tool.invoke(tool_args)

    logger.info(
        "extractor date_intent=%s resolved=%s",
        json.dumps(compact, ensure_ascii=False) if compact else None,
        json.dumps(resolved, ensure_ascii=False) if resolved else None,
    )

    constraints_json = normalize_constraints(
        parsed,
        today=today,
        user_text=user_text,
        resolved=resolved,
    )
    constraints_json = attach_campsite(constraints_json, parsed)
    if (
        not constraints_json.get("semantic_constraints")
        and not constraints_json.get("numeric_constraints")
        and constraints_json.get("date") is None
        and tool_calls
    ):
        constraints_json = constraints_from_tool_calls(tool_calls)
        constraints_json = normalize_constraints(
            constraints_json,
            today=today,
            user_text=user_text,
            resolved=resolved,
        )
        constraints_json = attach_campsite(constraints_json, parsed)

    if compact:
        constraints_json["date_intent"] = compact
    constraints_payload = json.dumps(constraints_json, ensure_ascii=False)
    if not constraints_payload.strip():
        constraints_payload = json.dumps(EMPTY_CONSTRAINTS, ensure_ascii=False)
    return {"messages": [AIMessage(content=constraints_payload)]}


def planner_node(state: ChatState) -> ChatState:
    """Vacancies + prices, then semantic intersection with evidence."""
    constraints_json = latest_constraints_json(state["messages"])
    payload = planner_fits_payload(constraints_json)
    return {
        "messages": [
            ChatMessage(
                content=json.dumps(payload, ensure_ascii=False, default=str),
                role="assistant",
            )
        ]
    }


def recommender_node(state: ChatState) -> ChatState:
    """Generate final recommendation based on constraints and tool responses."""
    system_msg = SystemMessage(content=RECOMMENDER_SYSTEM_PROMPT)

    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    tool_messages = [msg for msg in state["messages"] if isinstance(msg, ChatMessage)]

    recommendation = heavy_model.invoke([system_msg] + user_messages + tool_messages)
    text = message_text(recommendation.content).strip() or EMPTY_REPLY_FALLBACK

    return {"messages": [AIMessage(content=text)]}


HeavyThrough = Literal["extractor", "planner", "recommender"]


def build_graph(*, stop_after: HeavyThrough = "recommender"):
    """Compile the agent. `stop_after` is the last heavy node to run (Streamlit)."""
    builder = StateGraph(ChatState)

    builder.add_node("light", light_node)
    builder.add_node("extractor", extractor_node)
    if stop_after in ("planner", "recommender"):
        builder.add_node("planner", planner_node)
    if stop_after == "recommender":
        builder.add_node("recommender", recommender_node)

    builder.add_conditional_edges(
        START,
        router,
        {
            "trivial": "light",
            "non_trivial": "light",
        },
    )
    builder.add_conditional_edges(
        "light",
        check_after_cleaning,
        {
            "heavy": "extractor",
            "end": END,
        },
    )

    if stop_after == "extractor":
        builder.add_edge("extractor", END)
    elif stop_after == "planner":
        builder.add_edge("extractor", "planner")
        builder.add_edge("planner", END)
    else:
        builder.add_edge("extractor", "planner")
        builder.add_edge("planner", "recommender")
        builder.add_edge("recommender", END)

    return builder.compile()


graph = build_graph()


def __getattr__(name: str):
    """Live binding for search globals that tests/Streamlit still read on graph."""
    if name == "_LAST_OPEN_SLOTS_QUERY":
        from source.agent import search as search_mod

        return search_mod._LAST_OPEN_SLOTS_QUERY
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
