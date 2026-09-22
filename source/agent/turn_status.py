"""Live phase line for a Streamlit turn. No listener means a no-op."""

from __future__ import annotations

from collections.abc import Callable

SEARCHING = "Searching"
RANKING = "Ranking"

_listener: Callable[[str], None] | None = None


def set_turn_status(listener: Callable[[str], None] | None) -> None:
    global _listener
    _listener = listener


def report_turn_status(text: str) -> None:
    listener = _listener
    if listener is None:
        return
    try:
        listener(text)
    except Exception:
        return


def found_candidates_line(count: int) -> str:
    noun = "candidate" if count == 1 else "candidates"
    return f"Found {count} {noun}, filtering"
