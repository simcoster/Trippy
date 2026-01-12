import warnings
import os

# Suppress Pydantic V1 compatibility warning with Python 3.14+
warnings.filterwarnings("ignore", message=".*Pydantic V1.*", category=UserWarning)

from typing_extensions import TypedDict, Annotated  # noqa: E402

from langgraph.graph import StateGraph, START, END  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ChatMessage  # noqa: E402
from langchain_core.tools import StructuredTool  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
import psycopg  # noqa: E402
from pgvector.psycopg import register_vector  # noqa: E402
from openai import OpenAI  # noqa: E402

# Load environment variables
load_dotenv()


# ---- 1. State type for LangGraph ----

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---- 2. RAG Tool for Claims Search ----

MODEL = "text-embedding-3-small"  # 1536 dims by default

def search_campsites(numeric_constraints):
    """
    Search for campsites in the 'campsites' table using a list of numeric constraints.
    Each constraint in numeric_constraints should have: field, operator, value.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return "Error: DATABASE_URL not configured"
    # Allowed fields and mapping
    field_map = {
        "price": "price",
        "price_per_night": "price",
        "ride_time_from_tlv": "ride_time_from_tlv",
        "ride_time": "ride_time_from_tlv",
        "region": "region",
        "campsite_id": "campsite_id"
    }
    allowed_ops = {"<", "<=", ">", ">=", "=", "==", "!="}
    where_clauses = []
    values = []
    for constraint in numeric_constraints:
        field = constraint.get("field")
        operator = constraint.get("operator")
        value = constraint.get("value")
        # Defensive: only allow mapped fields and safe operators
        db_field = field_map.get(field)
        # Support synonyms, skip any unknown fields
        if not db_field or not operator or db_field not in ["price", "ride_time_from_tlv"]:
            continue
        op = operator if operator in allowed_ops else "="
        where_clauses.append(f"{db_field} {op} %s")
        values.append(value)
    if not where_clauses:
        return "No valid numeric constraints provided"
    sql = f"""
        SELECT campsite_id, region, price, ride_time_from_tlv
        FROM campsites
        WHERE {' AND '.join(where_clauses)}
        ORDER BY price ASC
        LIMIT 10
    """
    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, values)
                rows = cur.fetchall()
                if not rows:
                    return "No campsites found matching numeric constraints"
                result = []
                for row in rows:
                    result.append({
                        "campsite_id": row[0],
                        "region": row[1],
                        "price": row[2],
                        "ride_time_from_tlv": row[3]
                    })
                return result
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
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    db_url = os.environ.get("DATABASE_URL")
    
    if not openai_api_key:
        return "Error: OPENAI_API_KEY not configured"
    if not db_url:
        return "Error: DATABASE_URL not configured"
    
    try:
        # Get embedding from OpenAI
        client = OpenAI(api_key=openai_api_key)
        resp = client.embeddings.create(
            model=MODEL,
            input=query,
            encoding_format="float",
        )
        embedding = resp.data[0].embedding
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


# ---- 3. Two models: light + heavy ----

light_model = ChatOpenAI(
    model="gpt-4.1-mini",   # cheap/fast
    temperature=0.7,
)

heavy_model = ChatOpenAI(
    model="gpt-4.1",        # smart/expensive
    temperature=0.7,
).bind_tools([claims_search_tool])


# ---- 4. Nodes ----

def router(state: ChatState) -> str:
    """
    Decide if the message is trivial (like "thanks!") or needs processing.
    """
    last_message = state["messages"][-1].content.lower().strip()
    
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
    last_content = last_message.content
    
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
        return {"messages": [response]}
    else:
        # Clean non-trip-planning content from the message
        # IMPORTANT: Consider the full conversation context - references to previous
        # messages (like "not redplace" after "redplace" was mentioned) are trip-planning related
        
        # Format conversation history for context
        conversation_context = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in state["messages"][:-1]
        ])
        
        cleaning_prompt = f"""
        You are a filter before a trip-planning assistant.
        You see the full conversation and the last user message (in Hebrew).

        Your job:
        - If the last user message is related to planning or updating a trip 
          (destinations, dates, people coming, budget, rides, packing, logistics, etc.)
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
        cleaned_content = cleaned_response.content.strip()
        
        # Store the cleaned content in the state
        if cleaned_content == "keep":
            cleaned_message = HumanMessage(content=cleaned_content)
            return {"messages": [cleaned_message]}
        else:
            # No trip-planning content left, return empty response
            return {"messages": [AIMessage(content="I didn't find any trip-planning related questions in your message. How can I help you plan your trip?")]}


def check_after_cleaning(state: ChatState) -> str:
    """
    Check if there's still content after cleaning that needs heavy model.
    """
    last_message = state["messages"][-1]

    # If the last message is an AI response (from light model), we're done
    if isinstance(last_message, AIMessage):
        return "end"

    # If there's a cleaned human message, route to heavy model
    if isinstance(last_message, HumanMessage) and last_message.content.strip():
        return "heavy"
    else:
        return "end"


def planner_node(state: ChatState) -> ChatState:
    """Extract constraints from user query and run RAG searches. Returns constraints + tool results."""
    import json
    from langchain_core.messages import ToolMessage

    system_msg = SystemMessage(
        content=(
            """
            You are a structured query extractor for a campsite recommendation system called Trippy.
            Your task is to analyze a user query and extract all relevant constraints in a precise JSON format.

            Rules:
            1. Output ONLY JSON. No text, commentary, or explanations.
            2. Separate constraints into:
                a) semantic_constraints: qualitative queries about campsite attributes
                    (e.g., "quiet", "good for kids", "has hot water").
                    Each item must have a "query" field containing the user-expressed concept.
                b) numeric_constraints: quantitative filters (e.g., price, distance, rating).
                    Each item must have "field", "operator", "value", and optionally "unit".
            3. Include every constraint explicitly mentioned or clearly implied by the user.
            4. Preserve negation (e.g., "not suitable for kids" → query: "not suitable for kids").
            5. Avoid interpretation beyond what is stated; do not assume preferences or motivations.
            6. Keep queries concise; use the same wording as the user where possible.
            7. Return JSON in a single object with exactly two keys:
                "semantic_constraints" and "numeric_constraints".
                If no constraints exist in a category, use an empty array for that key.
            8. All constraints must be in English

            Example output:

            Input:
            "Looking for a quiet campsite, clean, suitable for kids, and costs less than 500 NIS per night."

            Output:
            {
                "semantic_constraints": [
                    {"query": "quiet"},
                    {"query": "clean"},
                    {"query": "suitable for kids"},
                ],
                    "numeric_constraints": [
                        {"field": "price_per_night", "operator": "<=", "value": 500},
                ]
            }
            """.strip().replace("            ", "")
        )
    )

    response = heavy_model.invoke([system_msg] + state["messages"])

    # Parse JSON from response content
    try:
        constraints_json = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON from text
        import re

        json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
        if json_match:
            constraints_json = json.loads(json_match.group())
        else:
            # No valid JSON, return empty constraints
            constraints_json = {"semantic_constraints": [], "numeric_constraints": []}

    # Run RAG searches based on constraints
    tool_messages = []
    for semantic in constraints_json.get("semantic_constraints", []):
        query = semantic.get("query")
        if query:
            result = search_claims(query, limit=5)
            tool_messages.append(ChatMessage(content=result, role="assistant"))

    if constraints_json.get("numeric_constraints"):
        campsites_result = search_campsites(constraints_json["numeric_constraints"])
        tool_messages.append(
            ChatMessage(content=str(campsites_result), role="assistant")
        )

    # Store constraints as a message for the recommender node
    constraints_msg = AIMessage(content=json.dumps(constraints_json))

    # Return constraints extraction + tool results (but NOT final recommendation)
    return {"messages": [constraints_msg] + tool_messages}


def recommender_node(state: ChatState) -> ChatState:
    """Generate final recommendation based on constraints and tool responses."""
    
    system_msg = SystemMessage(
        content=(
            "You are a helpful trip-planning assistant for Trippy. "
            "Based on the user's constraints and the search results provided, "
            "recommend specific campsites that match their preferences. "
            "Only use information from the search results - do not hallucinate or invent details. "
            "If no campsites match, explain why and suggest alternatives. "
            "Respond in the same language as the user's query."
        )
    )
    
    # Get the original user messages and tool results from state
    # The state contains: user messages + constraint extraction message + tool messages
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    tool_messages = [msg for msg in state["messages"] if isinstance(msg, ChatMessage)]
    
    # Generate recommendation using original user query + tool results
    recommendation = heavy_model.invoke([system_msg] + user_messages + tool_messages)
    
    return {"messages": [recommendation]}


# ---- 5. Build the graph ----

builder = StateGraph(ChatState)

builder.add_node("light", light_node)
builder.add_node("planner", planner_node)
builder.add_node("recommender", recommender_node)

# Router: from START decide if trivial or non-trivial
builder.add_conditional_edges(
    START,
    router,
    {
        "trivial": "light",
        "non_trivial": "light",
    },
)

# After light node, check if we need heavy model
builder.add_conditional_edges(
    "light",
    check_after_cleaning,
    {
        "heavy": "planner",
        "end": END,
    },
)

# After planner node, go to recommender
builder.add_edge("planner", "recommender")

# After recommender, go to END
builder.add_edge("recommender", END)

graph = builder.compile()
