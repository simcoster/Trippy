"""Nebius chat extract + embedding clients for amenity enrichment.

Also provides the LangGraph agent chat model (same Nebius Qwen instruct endpoint).
"""

from __future__ import annotations

import json
import os
import re
import ssl
from dataclasses import dataclass
from typing import Any

import httpx
from langchain_openai import ChatOpenAI
from openai import OpenAI

from .schemas import AccommodationExtract

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
# Shared instruct model for amenity extract + most agent nodes
QWEN_INSTRUCT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
QWEN_INSTRUCT_INPUT_USD_PER_MTOK = 0.20
QWEN_INSTRUCT_OUTPUT_USD_PER_MTOK = 0.60
# Agent planner / query-constraint extract — keep 30B for now (easy to bump later)
QWEN_INSTRUCT_30B_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    if hasattr(ssl, "VERIFY_X509_STRICT"):
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return ctx


def make_nebius_openai_client() -> OpenAI:
    api_key = os.environ.get("NEBIUS_API_KEY") or os.environ.get("NEBULA_API_KEY")
    if not api_key:
        raise RuntimeError("NEBIUS_API_KEY (or NEBULA_API_KEY) is required")
    return OpenAI(
        base_url=NEBIUS_BASE_URL,
        api_key=api_key,
        http_client=httpx.Client(verify=_ssl_context(), timeout=120.0),
    )


@dataclass
class LlmUsage:
    """Accumulated Nebius chat + embedding token usage for amenity enrichment."""

    chat_prompt_tokens: int = 0
    chat_completion_tokens: int = 0
    embed_prompt_tokens: int = 0
    chat_calls: int = 0
    embed_calls: int = 0

    def add_chat(self, usage: Any | None) -> None:
        if usage is None:
            return
        self.chat_calls += 1
        self.chat_prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.chat_completion_tokens += int(
            getattr(usage, "completion_tokens", 0) or 0
        )

    def add_embed(self, usage: Any | None) -> None:
        if usage is None:
            return
        self.embed_calls += 1
        prompt = getattr(usage, "prompt_tokens", None)
        if prompt is None:
            prompt = getattr(usage, "total_tokens", 0)
        self.embed_prompt_tokens += int(prompt or 0)

    def merge(self, other: LlmUsage) -> None:
        self.chat_prompt_tokens += other.chat_prompt_tokens
        self.chat_completion_tokens += other.chat_completion_tokens
        self.embed_prompt_tokens += other.embed_prompt_tokens
        self.chat_calls += other.chat_calls
        self.embed_calls += other.embed_calls

    @property
    def input_tokens(self) -> int:
        return self.chat_prompt_tokens + self.embed_prompt_tokens

    @property
    def output_tokens(self) -> int:
        return self.chat_completion_tokens

    @property
    def cost_usd(self) -> float:
        return (
            self.chat_prompt_tokens * ExtractorLLMClient.INPUT_USD_PER_MTOK
            + self.chat_completion_tokens * ExtractorLLMClient.OUTPUT_USD_PER_MTOK
            + self.embed_prompt_tokens * EmbeddingLLMClient.INPUT_USD_PER_MTOK
        ) / 1_000_000

    def summary(self, *, prefix: str = "    ") -> str:
        return (
            f"{prefix}LLM usage: "
            f"in={self.input_tokens} "
            f"(chat_prompt={self.chat_prompt_tokens}, "
            f"embed={self.embed_prompt_tokens}), "
            f"out={self.output_tokens}, "
            f"calls={self.chat_calls} chat / {self.embed_calls} embed, "
            f"cost≈${self.cost_usd:.6f}"
        )


def _parse_json_payload(raw: str) -> dict[str, Any]:
    text = _FENCE_RE.sub("", (raw or "").strip()).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object, got {type(data).__name__}")
    return data


class AgentChatClient:
    """Nebius chat client for the Trippy LangGraph agent (Streamlit / Telegram)."""

    MODEL = QWEN_INSTRUCT_MODEL
    INPUT_USD_PER_MTOK = QWEN_INSTRUCT_INPUT_USD_PER_MTOK
    OUTPUT_USD_PER_MTOK = QWEN_INSTRUCT_OUTPUT_USD_PER_MTOK
    TEMPERATURE = 0.7

    def __init__(
        self,
        client: OpenAI | None = None,
        *,
        temperature: float | None = None,
        model: str | None = None,
    ) -> None:
        self._client = client
        self.model = model or self.MODEL
        self.temperature = self.TEMPERATURE if temperature is None else temperature

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = make_nebius_openai_client()
        return self._client

    def as_langchain(self) -> ChatOpenAI:
        """LangChain `ChatOpenAI` bound to Nebius + Qwen instruct."""
        api_key = os.environ.get("NEBIUS_API_KEY") or os.environ.get("NEBULA_API_KEY")
        if not api_key:
            raise RuntimeError("NEBIUS_API_KEY (or NEBULA_API_KEY) is required")
        return ChatOpenAI(
            model=self.model,
            api_key=api_key,
            base_url=NEBIUS_BASE_URL,
            temperature=self.temperature,
            http_client=httpx.Client(verify=_ssl_context(), timeout=120.0),
        )


def make_agent_chat_model(
    *,
    temperature: float = 0.7,
    model: str | None = None,
) -> ChatOpenAI:
    """Factory for a LangChain chat model on Nebius Qwen instruct."""
    return AgentChatClient(temperature=temperature, model=model).as_langchain()


class ExtractorLLMClient:
    """Nebius chat model for Hebrew tooltip → structured accommodation JSON."""

    MODEL = QWEN_INSTRUCT_MODEL
    INPUT_USD_PER_MTOK = QWEN_INSTRUCT_INPUT_USD_PER_MTOK
    OUTPUT_USD_PER_MTOK = QWEN_INSTRUCT_OUTPUT_USD_PER_MTOK
    TEMPERATURE = 0
    SYSTEM_PROMPT = """You are a precise JSON extraction engine.
Extract accommodation details from Hebrew raw text into structured JSON.
You are given the accommodation type name and the tooltip text — use both.

Rules:
- Output valid JSON only, without markdown wrappers.
- Count exact beds (e.g. double_bed, bunk_bed, single_bed). For pitches/parking spots with no beds, use 0.
- room_count: number of connected rooms/units in this listing. Default 1.
  Example: "שתי חושות מחוברות עם דלת מקשרת שבכל חדר: ..." → room_count: 2
- Convert Hebrew amenities to standardized snake_case English terms.
- Infer accommodation_category from the type name (and text if needed). Allowed values:
  room, cabin, tent, trailer_parking, tent_pitch, bungalow, dorm, other
- Always include accommodation_category as the first item in "amenities" (so it is searchable).
- Name amenities in context of the category. Examples for trailer_parking / tent_pitch:
  - חיבור חשמל / נקודת חשמל → electric_hookup (not electricity / power_outlet)
  - חיבור מים → water_hookup
  - ביוב / ניקוז → sewage_hookup
- Named places (LLM must expand — do not rely on a fixed place list):
  Whenever a specific place, landmark, or region is named, ALSO add its type(s)
  as separate amenity strings so generic queries can match.
  Rule: named place → keep the specific label AND add the geographic / feature type.
  Examples (illustrative only; apply the same idea to any place you recognize):
  - כינרת / Kineret → ["near the Kineret", "near a lake", "near a body of water"]
  - נגב תחתון / lower Negev → ["near lower Negev", "near a desert"]
  - ים המלח / Dead Sea → ["near the Dead Sea", "near a body of water"]
  - מכתש רמון → ["near Ramon crater", "near a crater", "near a desert"]
  Use English snake_case or short phrases consistently (e.g. near_a_desert / "near a desert").
- Only add amenities to "not_included" if explicitly stated as not included or that guests should bring their own (e.g. "bring your own towels"). Do not infer not_included from absence alone.
- Extract check_in_time / check_out_time when stated (HH:MM 24h, e.g. "15:00"). Use null if unknown.
- Extract policy_rules only when explicitly stated. Use null for unknown keys. Typical keys:
  - min_nights (int): minimum stay any night
  - max_nights (int), max_weekend_nights (int)
  - min_weekend_nights (int), min_holiday_nights (int)
  - pets_allowed (bool)
  Example: "מותנה במינימום 2 לילות בסופי שבוע ובחגים"
  → {"min_weekend_nights": 2, "min_holiday_nights": 2}
  Omit policy_rules entirely (or use {}) if nothing policy-related is stated.

Schema:
{
  "accommodation_category": str,
  "double_bed": int,
  "single_bed": int,
  "room_count": int,
  "max_people": int | null,
  "check_in_time": "HH:MM" | null,
  "check_out_time": "HH:MM" | null,
  "policy_rules": {
    "min_nights": int | null,
    "max_nights": int | null,
    "max_weekend_nights": int | null,
    "min_weekend_nights": int | null,
    "min_holiday_nights": int | null,
    "pets_allowed": bool | null
  } | null,
  "amenities": list[str],
  "not_included": list[str]
}
"""

    def __init__(
        self,
        client: OpenAI | None = None,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.client = client or make_nebius_openai_client()
        self.model = model or self.MODEL
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT

    def extract(
        self,
        raw_text: str,
        *,
        type_name: str,
        usage: LlmUsage | None = None,
    ) -> dict[str, Any]:
        user_content = (
            f"Accommodation type: {type_name}\n\nTooltip:\n{raw_text}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=self.TEMPERATURE,
        )
        if usage is not None:
            usage.add_chat(response.usage)
        content = response.choices[0].message.content or ""
        data = _parse_json_payload(content)
        return AccommodationExtract.model_validate(data).as_details_dict()


# Same extract prompt without named-place expansion (used by the dedicated place node path).
EXTRACT_WITHOUT_PLACE_EXPANSION_PROMPT = re.sub(
    r"- Named places \(LLM must expand.*?(?=\n- Only add amenities to \"not_included\")",
    "",
    ExtractorLLMClient.SYSTEM_PROMPT,
    count=1,
    flags=re.DOTALL,
)


class PlaceEnrichmentLLMClient:
    """Second-pass: add geographic types for places named in the tooltip."""

    MODEL = QWEN_INSTRUCT_MODEL
    INPUT_USD_PER_MTOK = QWEN_INSTRUCT_INPUT_USD_PER_MTOK
    OUTPUT_USD_PER_MTOK = QWEN_INSTRUCT_OUTPUT_USD_PER_MTOK
    TEMPERATURE = 0
    SYSTEM_PROMPT = """You enrich campsite amenities with geographic types.

You are given:
1. The original Hebrew tooltip (source of truth)
2. Amenities already extracted from that tooltip

Task:
1. Keep every amenity already listed (do not drop or rename any).
2. For EVERY specific place / landmark / region named in the tooltip,
   add a dedicated snake_case place label if it is missing
   (e.g. near_ramon_crater, near_the_kineret, near_eilat).
   The place name itself must appear in an amenity string.
3. Then add that place's geographic TYPE labels
   (near_a_lake, near_a_desert, near_the_sea, near_a_beach,
   near_a_crater, near_a_coral_reef, near_the_red_sea,
   near_a_body_of_water, …).

Rules:
- Output valid JSON only: {"amenities": ["..."]}
- ONLY use places that appear in THIS tooltip. Do not invent other
  Israeli sites (no Dead Sea / Kineret / Negev / Ramon / Eilat unless
  this tooltip names them).
- Do not add types that contradict the tooltip.
- If there are no named places, return the amenities unchanged.

Do not copy example places from any training pattern. Apply the rule to
whatever THIS tooltip actually names.
"""

    def __init__(
        self,
        client: OpenAI | None = None,
        *,
        model: str | None = None,
    ) -> None:
        self.client = client or make_nebius_openai_client()
        self.model = model or self.MODEL

    def enrich(
        self,
        raw_text: str,
        amenities: list[str],
        *,
        usage: LlmUsage | None = None,
    ) -> list[str]:
        user_content = (
            f"Tooltip:\n{raw_text}\n\n"
            f"Extracted amenities:\n{json.dumps(amenities, ensure_ascii=False)}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=self.TEMPERATURE,
        )
        if usage is not None:
            usage.add_chat(response.usage)
        content = response.choices[0].message.content or ""
        data = _parse_json_payload(content)
        raw_list = data.get("amenities", amenities)
        if not isinstance(raw_list, list):
            return list(amenities)
        return [str(a).strip() for a in raw_list if str(a).strip()]


class EmbeddingLLMClient:
    """Nebius embedding model for amenity name vectors."""

    MODEL = "Qwen/Qwen3-Embedding-8B"
    DIMENSIONS = 1536  # HNSW index limit is 2000; Qwen3-Embedding supports MRL dims
    INPUT_USD_PER_MTOK = 0.01

    def __init__(self, client: OpenAI | None = None) -> None:
        self.client = client or make_nebius_openai_client()

    def embed(
        self,
        texts: list[str],
        *,
        usage: LlmUsage | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        resp = self.client.embeddings.create(
            model=self.MODEL,
            input=texts,
            dimensions=self.DIMENSIONS,
        )
        if usage is not None:
            usage.add_embed(resp.usage)
        by_index = {item.index: item.embedding for item in resp.data}
        return [by_index[i] for i in range(len(texts))]


class ClaimsEmbeddingLLMClient(EmbeddingLLMClient):
    """Nebius embedding model for claims RAG (same Qwen3-Embedding-8B + 1536 dims)."""


def amenity_llm_clients() -> tuple[ExtractorLLMClient, EmbeddingLLMClient]:
    """Shared OpenAI/httpx client for extract + embed in one scrape run."""
    shared = make_nebius_openai_client()
    return ExtractorLLMClient(shared), EmbeddingLLMClient(shared)
