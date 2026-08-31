"""Normalize LangChain message content to plain text."""

from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage


def message_text(content) -> str:
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


def is_keep_decision(text: str) -> bool:
    """True if the cleaner decided to keep the message for the planner."""
    t = message_text(text).strip().lower().strip("\"'`")
    if not t:
        return False
    first_line = t.splitlines()[0].strip()
    first_word = first_line.split()[0].strip(".,!?;:") if first_line else ""
    if first_word == "keep":
        return True
    if first_word == "drop":
        return False
    return t == "keep" or t.startswith("keep ")


def latest_user_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            text = message_text(msg.content).strip()
            if text:
                return text
    return ""
