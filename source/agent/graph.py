import os
import warnings
from typing import Annotated, Literal, TypedDict

# Suppress Pydantic V1 compatibility warning with Python 3.14+
warnings.filterwarnings("ignore", message=".*Pydantic V1.*", category=UserWarning)

import psycopg
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pgvector.psycopg import register_vector

from source.agent.constraints import (
    EMPTY_CONSTRAINTS,
    amenity_search_queries,
    normalize_constraints,
    parse_constraints_dict,
    today_il,
)
from source.scraper.amenity_enrichment.llm import (
    QWEN_INSTRUCT_30B_MODEL,
    QWEN_INSTRUCT_MODEL,
    ClaimsEmbeddingLLMClient,
    make_agent_chat_model,
)

# Load environment variables
load_dotenv()


# ---- 1. State type for LangGraph ----

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---- 2. RAG Tool for Claims Search ----

_claims_embedder = ClaimsEmbeddingLLMClient()


def search_campsites(numeric_constraints):
    """
    List campsites from the 'campsites' table (id, name, url).
    Numeric filters (price / ride time) are not on this table yet;
    they will come from availability data later. `numeric_constraints`
    is accepted for API compatibility with the planner node.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return "Error: DATABASE_URL not configured"
    _ = numeric_constraints  # reserved for future availability filters
    sql = """
        SELECT id, name, url
        FROM campsites
        ORDER BY id
        LIMIT 50
    """
    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
                if not rows:
                    return "No campsites found"
                return [
                    {"id": row[0], "name": row[1], "url": row[2]}
                    for row in rows
                ]
    except Exception as e:
        return f"Error during search_campsites: {e}"


def search_claims(query: str, limit: int = 5) -> str:
    """
    Search for review claims using vector similarity.
    
    Args:
        query: The search query (e.g., "fit for stargazing", "has hot water")
        limit: Maximum number of results to return (default: 5)
    
    Returns:
        A formatted string with matching claims, their campsite IDs, and relevance scores.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return "Error: DATABASE_URL not configured"
    
    try:
        embedding = _claims_embedder.embed([query])[0]
        vec_literal = "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"
        
        # Search in database
        with psycopg.connect(db_url) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                # Use cosine distance (<#>), order by distance ascending
                cur.execute(
                    """
                    SELECT campsite_id, claim_en, claim_he, 
                           embedding <#> %s::vector AS distance
                    FROM claims
                    WHERE claim_en IS NOT NULL OR claim_he IS NOT NULL
                    ORDER BY embedding <#> %s::vector
                    LIMIT %s
                    """,
                    (vec_literal, vec_literal, limit)
                )
                rows = cur.fetchall()
                
                if not rows:
                    return f"No claims found matching: {query}"
                
                # Format results
                results = []
                for campsite_id, claim_en, claim_he, distance in rows:
                    claim_text = claim_en or claim_he or "N/A"
                    results.append(
                        f"Campsite: {campsite_id}\n"
                        f"Claim: {claim_text}\n"
                        f"Relevance: {distance:.4f}\n"
                    )
                
                return "\n---\n".join(results)
    
    except Exception as e:
        return f"Error searching claims: {str(e)}"


# Create the tool
claims_search_tool = StructuredTool.from_function(
    func=search_claims,
    name="search_claims",
    description=(
        "Search for review claims about campsites using semantic similarity. "
        "Use this when users ask about specific features, amenities, or experiences "
        "at campsites (e.g., 'has hot water', 'good for stargazing', 'clean facilities') "
        "that are not numeric (like 'price < 100', 'rating > 4.5', 'distance < 100km', etc.). "
        "Returns matching claims with campsite IDs and relevance scores."
    ),
)


# ---- 3. Two models: light + heavy (Nebius Qwen instruct) ----

# Do NOT bind_tools on the chat models: extractor/recommender need text/JSON
# replies. Tools are invoked imperatively in planner_node. Binding tools made
# Qwen return empty content + tool_calls, which surfaced as blank agent replies.
light_model = make_agent_chat_model(temperature=0.7)
heavy_model = make_agent_chat_model(temperature=0.7)
# Query-constraint extract (the LLM that used to live in planner). Flip to
# QWEN_INSTRUCT_MODEL when we want 235B here too.
planner_model = make_agent_chat_model(
    temperature=0.7, model=QWEN_INSTRUCT_30B_MODEL
)
AGENT_CHAT_MODEL = QWEN_INSTRUCT_MODEL
PLANNER_CHAT_MODEL = QWEN_INSTRUCT_30B_MODEL

_EMPTY_REPLY_FALLBACK = (
    "לא הצלחתי להשלים תשובה כרגע. נסו לנסח שוב עם תאריך, מיקום או העדפה (למשל מים זורמים)."
)


def _message_text(content) -> str:
    """Normalize LC message content (str | list blocks | None) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
            else:
                parts.append(getattr(block, "text", None) or str(block))
        return "".join(parts)
    return str(content)


def _is_keep_decision(text: str) -> bool:
    """True if the cleaner decided to keep the message for the planner."""
    t = _message_text(text).strip().lower().strip("\"'`")
    if not t:
        return False
    first_line = t.splitlines()[0].strip()
    first_word = first_line.split()[0].strip(".,!?;:") if first_line else ""
    if first_word == "keep":
        return True
    if first_word == "drop":
        return False
    return t == "keep" or t.startswith("keep ")


def _parse_constraints_json(raw: str) -> dict:
    """Parse extractor JSON then normalize to date/amenities schema."""
    parsed = parse_constraints_dict(raw)
    if not parsed:
        return dict(EMPTY_CONSTRAINTS)
    return normalize_constraints(parsed)


def _constraints_from_tool_calls(tool_calls) -> dict:
    semantic: list[dict] = []
    for tc in tool_calls or []:
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
        if name == "search_claims" and isinstance(args, dict):
            query = args.get("query")
            if query:
                semantic.append({"query": query})
    return normalize_constraints(
        {
            "semantic_constraints": semantic,
            "numeric_constraints": [],
            "amenities": [],
            "date": None,
        }
    )


def _latest_user_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            text = _message_text(msg.content).strip()
            if text:
                return text
    return ""


# ---- 4. Nodes ----

def router(state: ChatState) -> str:
    """
    Decide if the message is trivial (like "thanks!") or needs processing.
    """
    last_message = _message_text(state["messages"][-1].content).lower().strip()
    
    # Trivial responses that light model can handle directly
    trivial_patterns = [
        "thanks", "thank you", "thx", "ty", "appreciate it",
        "ok", "okay", "got it", "sounds good", "perfect",
        "bye", "goodbye", "see you", "later"
    ]
    
    if any(pattern in last_message for pattern in trivial_patterns) and len(last_message) < 50:
        return "trivial"
    else:
        return "non_trivial"


def light_node(state: ChatState) -> ChatState:
    """
    Light model handles:
    - Trivial prompts: answer directly
    - Non-trivial prompts: clean message to extract only trip-planning related content
    """
    last_message = state["messages"][-1]
    last_content = _message_text(last_message.content)
    
    # Check if this is a trivial response (from router)
    # We'll use a simple heuristic: if it's very short and matches trivial patterns
    trivial_patterns = [
        "thanks", "thank you", "thx", "ty", "appreciate it",
        "ok", "okay", "got it", "sounds good", "perfect",
        "bye", "goodbye", "see you", "later"
    ]
    is_trivial = any(pattern in last_content.lower() for pattern in trivial_patterns) and len(last_content) < 50
    
    if is_trivial:
        # Answer trivial prompts directly
        response = light_model.invoke(state["messages"])
        text = _message_text(response.content).strip() or _EMPTY_REPLY_FALLBACK
        return {"messages": [AIMessage(content=text)]}
    else:
        # Clean non-trip-planning content from the message
        # IMPORTANT: Consider the full conversation context - references to previous
        # messages (like "not redplace" after "redplace" was mentioned) are trip-planning related
        
        # Format conversation history for context
        conversation_context = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {_message_text(msg.content)}"
            for msg in state["messages"][:-1]
        ])
        
        cleaning_prompt = f"""
        You are a filter before a trip-planning assistant.
        You see the full conversation and the last user message (in Hebrew).

        Your job:
        - If the last user message is related to planning or updating a trip 
          (destinations, dates, people coming, budget, rides, packing, logistics,
          amenities such as running water, electricity, showers, etc.)
          OR it clearly refers to something trip-related mentioned earlier in the conversation
          (like "יהודה?" referring to "מדבר יהודה"),
          answer: "keep".
        - Otherwise (small talk, jokes, family, work, anything unrelated), answer: "drop".

        Always answer with exactly one word: "keep" or "drop".

        Examples:
        Conversation:
            User: "איפה יש מלון יפה?"
            Assistant: "יש מלון יפה במדבר יהודה"
            User: "יהודה?"
            Assistant: 
            → keep

        Conversation:
            User: "אני רוצה משהו לשישי הבא עם מים זורמים"
            Assistant:
            → keep

        Conversation:
            User: "מה שלומך?"
            Assistant: "בסדר, איך אני יכול לעזור?"
            User: "איפה אמא?"
            Assistant:
            → drop

        Conversation history:
        {conversation_context}

        Latest message: {last_content}
        """.strip()
        
        # Include full conversation history so cleaning can understand context
        cleaning_messages = state["messages"][:-1] + [HumanMessage(content=cleaning_prompt)]
        cleaned_response = light_model.invoke(cleaning_messages)
        cleaned_content = _message_text(cleaned_response.content)
        
        # On keep: add no messages so the user's HumanMessage stays last and
        # check_after_cleaning routes to the extractor (avoid injecting "keep").
        if _is_keep_decision(cleaned_content):
            return {}
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I didn't find any trip-planning related questions in your "
                        "message. How can I help you plan your trip?"
                    )
                )
            ]
        }


def check_after_cleaning(state: ChatState) -> str:
    """
    Check if there's still content after cleaning that needs heavy model.
    """
    last_message = state["messages"][-1]

    # If the last message is an AI response (from light model), we're done
    if isinstance(last_message, AIMessage):
        return "end"

    # If there's a cleaned human message, route to heavy model
    if isinstance(last_message, HumanMessage) and _message_text(last_message.content).strip():
        return "heavy"
    else:
        return "end"


def extractor_node(state: ChatState) -> ChatState:
    """Extract structured constraints from the user query (JSON only; no tools).

    Uses planner_model (30B). Flip that binding to 235B when we want it.
    """
    import json

    today = today_il()
    system_msg = SystemMessage(
        content=(
            f"""
            You are a structured query extractor for a campsite recommendation system called Trippy.
            Analyze the user query and extract constraints as JSON only (no commentary).

            Today's date (Asia/Jerusalem): {today.isoformat()}
            (weekday: {today.strftime("%A")})

            Schema (all keys required; use empty arrays / null when absent):
            {{
              "date": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} | null,
              "amenities": [
                "running water",
                {{"op": "or", "values": ["near the sea", "near a body of water"]}}
              ],
              "numeric_constraints": [
                {{"field": "price_per_night", "operator": "<=", "value": 500}}
              ],
              "semantic_constraints": [
                {{"query": "quiet"}}
              ]
            }}

            Rules:
            1. Output ONLY JSON.
            2. date: check-in/check-out as ISO start/end (end is exclusive).
               One night ⇒ end is the day after start. Always at least one night
               (never start == end). "this Friday" / "next Friday" → Friday check-in,
               Saturday check-out.
               Resolve relative phrases using today's date.
               Do NOT put dates in numeric_constraints or semantic_constraints.
            3. amenities: site/unit features and location-near-water prefs.
               Top-level list is AND. Use {{"op":"or","values":[...]}} for alternatives
               (e.g. "near the sea or some body of water").
               Prefer English canonical labels: "running water", "near the sea",
               "near a body of water".
            4. numeric_constraints: price, party size, distance (km), rating only — never dates.
            5. semantic_constraints: soft vibes not covered by amenities (quiet, good for kids).
               Each item: {{"query": "..."}}.
            6. Preserve negation in wording when stated.
            7. Do not invent constraints the user did not imply.

            Example:
            Input: "next friday, near the sea or some body of water to swim in"
            Output:
            {{
              "date": {{"start": "<that Friday ISO>", "end": "<Saturday after that Friday>"}},
              "amenities": [
                {{"op": "or", "values": ["near the sea", "near a body of water"]}}
              ],
              "numeric_constraints": [],
              "semantic_constraints": []
            }}
            """.strip().replace("            ", "")
        )
    )

    response = planner_model.invoke([system_msg] + state["messages"])
    raw = _message_text(response.content)
    tool_calls = getattr(response, "tool_calls", None) or []
    user_text = _latest_user_text(state["messages"])

    constraints_json = normalize_constraints(
        parse_constraints_dict(raw),
        today=today,
        user_text=user_text,
    )
    # If the model emitted tool calls instead of JSON, recover queries from args.
    if (
        not constraints_json.get("amenities")
        and not constraints_json.get("semantic_constraints")
        and not constraints_json.get("numeric_constraints")
        and constraints_json.get("date") is None
        and tool_calls
    ):
        constraints_json = _constraints_from_tool_calls(tool_calls)
        constraints_json = normalize_constraints(
            constraints_json, today=today, user_text=user_text
        )

    constraints_payload = json.dumps(constraints_json, ensure_ascii=False)
    if not constraints_payload.strip():
        constraints_payload = json.dumps(EMPTY_CONSTRAINTS, ensure_ascii=False)
    return {"messages": [AIMessage(content=constraints_payload)]}


def _latest_constraints_json(messages: list[BaseMessage]) -> dict:
    """Read the most recent constraints AIMessage (from extractor_node)."""
    import json

    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        raw = _message_text(msg.content).strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = parse_constraints_dict(raw)
        if isinstance(data, dict) and (
            "semantic_constraints" in data
            or "numeric_constraints" in data
            or "date" in data
            or "amenities" in data
        ):
            return normalize_constraints(data)
    return dict(EMPTY_CONSTRAINTS)


def planner_node(state: ChatState) -> ChatState:
    """Run RAG / campsite searches from extractor constraints. Tools only."""
    constraints_json = _latest_constraints_json(state["messages"])

    tool_messages = []

    # Amenities: AND across groups; OR within a group (merge results)
    for queries in amenity_search_queries(constraints_json.get("amenities") or []):
        chunks: list[str] = []
        for query in queries:
            result = search_claims(query, limit=5)
            text = _message_text(result) or f"No claims found matching: {query}"
            chunks.append(f"[{query}]\n{text}")
        tool_messages.append(
            ChatMessage(content="\n---\n".join(chunks), role="assistant")
        )

    for semantic in constraints_json.get("semantic_constraints", []):
        query = semantic.get("query") if isinstance(semantic, dict) else None
        if query:
            result = search_claims(query, limit=5)
            tool_messages.append(
                ChatMessage(
                    content=_message_text(result) or f"No claims found matching: {query}",
                    role="assistant",
                )
            )

    if constraints_json.get("numeric_constraints"):
        campsites_result = search_campsites(constraints_json["numeric_constraints"])
        tool_messages.append(
            ChatMessage(
                content=_message_text(str(campsites_result)) or "No campsites found",
                role="assistant",
            )
        )

    # Surface date to recommender via a short tool-style note (no availability SQL yet)
    date_range = constraints_json.get("date")
    if isinstance(date_range, dict) and date_range.get("start"):
        tool_messages.append(
            ChatMessage(
                content=(
                    "Requested stay nights: "
                    f"{date_range.get('start')} .. {date_range.get('end', date_range.get('start'))}"
                ),
                role="assistant",
            )
        )

    return {"messages": tool_messages}


def recommender_node(state: ChatState) -> ChatState:
    """Generate final recommendation based on constraints and tool responses."""
    
    system_msg = SystemMessage(
        content=(
            "You are a helpful trip-planning assistant for Trippy. "
            "Based on the user's constraints and the search results provided, "
            "recommend specific campsites that match their preferences. "
            "Only use information from the search results - do not hallucinate or invent details. "
            "If no campsites match or search results are empty, say so clearly and ask a "
            "short follow-up question (dates, area, budget, amenities). "
            "Never reply with an empty message. "
            "Respond in the same language as the user's query."
        )
    )
    
    # Get the original user messages and tool results from state
    # The state contains: user messages + constraint extraction message + tool messages
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    tool_messages = [msg for msg in state["messages"] if isinstance(msg, ChatMessage)]
    
    # Generate recommendation using original user query + tool results
    recommendation = heavy_model.invoke([system_msg] + user_messages + tool_messages)
    text = _message_text(recommendation.content).strip() or _EMPTY_REPLY_FALLBACK

    return {"messages": [AIMessage(content=text)]}


# ---- 5. Build the graph ----

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


# Production / Telegram: full path
graph = build_graph()
