import warnings

# Suppress Pydantic V1 compatibility warning with Python 3.14+
warnings.filterwarnings("ignore", message=".*Pydantic V1.*", category=UserWarning)

from typing_extensions import TypedDict, Annotated  # noqa: E402

from langgraph.graph import StateGraph, START, END  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage  # noqa: E402


# ---- 1. State type for LangGraph ----

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---- 2. Two models: light + heavy ----

light_model = ChatOpenAI(
    model="gpt-4.1-mini",   # cheap/fast
    temperature=0.7,
)

heavy_model = ChatOpenAI(
    model="gpt-4.1",        # smart/expensive
    temperature=0.7,
)


# ---- 3. Nodes ----

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

Latest message: {last_content}"""
        
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
    """Use the heavy model to answer trip-planning questions."""
    response = heavy_model.invoke(state["messages"])
    return {"messages": [response]}


# ---- 4. Build the graph ----

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
