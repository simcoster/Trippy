"""Turn a recommender stream into a JSON object, including a bad `\\escape`."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.utils.json import parse_partial_json


def json_object_prefix(raw: str) -> str:
    start = (raw or "").find("{")
    if start < 0:
        return ""
    return raw[start:]


def drop_invalid_json_escapes(text: str) -> str:
    """Keep only JSON string escapes; turn `\\pitch` into `pitch`."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "\\":
            out.append(text[i])
            i += 1
            continue
        nxt = text[i + 1] if i + 1 < n else ""
        if nxt in '"\\/bfnrt':
            out.append(text[i : i + 2])
            i += 2
            continue
        hex_digits = "0123456789abcdefABCDEF"
        if (
            nxt == "u"
            and i + 5 < n
            and all(ch in hex_digits for ch in text[i + 2 : i + 6])
        ):
            out.append(text[i : i + 6])
            i += 6
            continue
        i += 1
    return "".join(out)


def parse_stream_json(blob: str) -> dict[str, Any] | None:
    """parse_partial_json re-raises on a finished-but-illegal `\\escape`."""
    for candidate in (blob, drop_invalid_json_escapes(blob)):
        try:
            parsed = parse_partial_json(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
