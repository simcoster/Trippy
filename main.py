import warnings
import logging
import os

# Suppress Pydantic V1 compatibility warning with Python 3.14+
warnings.filterwarnings("ignore", message=".*Pydantic V1.*", category=UserWarning)

from dotenv import load_dotenv  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
import httpx  # noqa: E402
from langchain_core.messages import HumanMessage, AIMessage  # noqa: E402
from graph import graph, ChatState  # noqa: E402

# Load environment variables from .env file
load_dotenv()

# Set up logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load secrets from environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
NGROK_URL = os.getenv("NGROK_URL", "")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN environment variable is not set. Please check your .env file.")
TELEGRAM_API_BASE = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

app = FastAPI()

# Store conversation history per chat_id
# In production, use a proper database or Redis
conversations: dict[int, list] = {}


@app.get("/")
async def root():
    return {"status": "ok", "message": "FastAPI + Telegram bot running"}


@app.post("/webhook")
async def telegram_webhook(request: Request):
    """
    This endpoint gets called by Telegram whenever someone sends a message to your bot.
    """
    chat_id = None
    reply_text = ""
    
    try:
        update = await request.json()
        logger.info(f"Received webhook update: {update}")

        # Basic safety: only handle normal messages with text
        message = update.get("message")
        if not message:
            logger.debug("No message in update")
            return {"ok": True}

        chat_id = message["chat"]["id"]
        text = message.get("text", "")
        logger.info(f"Chat ID: {chat_id}, Text: {text}")
        if not text:
            logger.debug("No text in message")
            return {"ok": True}

        # Get or initialize conversation history for this chat
        if chat_id not in conversations:
            conversations[chat_id] = []
            logger.info(f"Initialized new conversation for chat_id: {chat_id}")

        # Add user message to conversation history
        user_message = HumanMessage(content=text)
        conversations[chat_id].append(user_message)
        logger.debug(f"Conversation history length: {len(conversations[chat_id])}")

        # Create state with conversation history
        state: ChatState = {"messages": conversations[chat_id]}

        # Run the graph
        logger.info("Invoking LangGraph...")
        result = graph.invoke(state)
        logger.info("LangGraph completed successfully")
        
        # Get the last AI message from the result
        # The graph returns messages with new AI responses appended
        all_messages = result["messages"]
        logger.debug(f"Total messages after graph: {len(all_messages)}")
        
        # Find the most recent AI message (should be the last one added)
        ai_messages = [msg for msg in all_messages if isinstance(msg, AIMessage)]
        logger.debug(f"Found {len(ai_messages)} AI messages")
        
        if ai_messages:
            # Get the most recent AI message
            reply_text = ai_messages[-1].content
            logger.info(f"AI response: {reply_text[:100]}...")
        else:
            # Fallback: get last message if it has content
            last_message = all_messages[-1] if all_messages else None
            if last_message and hasattr(last_message, 'content'):
                reply_text = last_message.content
                logger.warning("Using fallback: last message content")
            else:
                reply_text = "I'm here to help you plan your trip!"
                logger.warning("Using default fallback message")
        
        # Update conversation history with the full result (graph manages message state)
        conversations[chat_id] = all_messages

    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}", exc_info=True)
        reply_text = f"Sorry, I encountered an error: {str(e)}"
        # Try to get chat_id from update if available
        try:
            if 'update' in locals():
                chat_id = update.get("message", {}).get("chat", {}).get("id")
        except Exception:
            pass

    # Send reply back to Telegram
    if chat_id:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{TELEGRAM_API_BASE}/sendMessage",
                    json={"chat_id": chat_id, "text": reply_text},
                )
                logger.info(f"Sent reply to Telegram, status: {response.status_code}")
        except Exception as e:
            logger.error(f"Error sending message to Telegram: {str(e)}", exc_info=True)

    # Telegram expects a 200 OK quickly
    return {"ok": True}