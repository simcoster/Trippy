import warnings
import os

# Suppress Pydantic V1 compatibility warning with Python 3.14+
warnings.filterwarnings("ignore", message=".*Pydantic V1.*", category=UserWarning)

from typing_extensions import TypedDict, Annotated  # noqa: E402

from langgraph.graph import StateGraph, START, END  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage  # noqa: E402
from langchain_core.tools import StructuredTool  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
import psycopg  # noqa: E402
from pgvector.psycopg import register_vector  # noqa: E402
from openai import OpenAI  # noqa: E402
from openai.resources.embeddings import create as _emb_create  # noqa: E402

# Load environment variables
load_dotenv()


# ---- 1. State type for LangGraph ----

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---- 2. RAG Tool for Claims Search ----

MODEL = "text-embedding-3-small"  # 1536 dims by default

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
        resp = _emb_create(
            client.embeddings,
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


def heavy_node(state: ChatState) -> ChatState:
    """Use the heavy model to answer trip-planning questions with tool support."""
    from langchain_core.messages import ToolMessage
    
    response = heavy_model.invoke(state["messages"])
    
    # Check if the model wants to use tools
    if hasattr(response, "tool_calls") and response.tool_calls:
        # Execute tool calls and add results to the conversation
        tool_messages = []
        for tool_call in response.tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id")
            
            if tool_name == "search_claims":
                result = search_claims(**tool_args)
                tool_messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call_id,
                        name=tool_name
                    )
                )
        
        # Invoke model again with tool results to get final answer
        final_response = heavy_model.invoke(state["messages"] + [response] + tool_messages)
        return {"messages": [response] + tool_messages + [final_response]}
    
    return {"messages": [response]}


# ---- 5. Build the graph ----

builder = StateGraph(ChatState)

builder.add_node("light", light_node)
builder.add_node("heavy", heavy_node)

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
        "heavy": "heavy",
        "end": END,
    },
)

# After heavy model, go to END
builder.add_edge("heavy", END)

graph = builder.compile()
