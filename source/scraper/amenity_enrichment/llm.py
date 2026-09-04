"""Nebius chat extract + embedding clients for amenity enrichment.

Also provides the LangGraph agent chat model (same Nebius Qwen instruct endpoint).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from langchain_openai import ChatOpenAI
from openai import OpenAI

from source.scraper.tls import ssl_context

from .schemas import AccommodationExtract

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
# Shared instruct model for amenity extract + most agent nodes
QWEN_INSTRUCT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
QWEN_INSTRUCT_INPUT_USD_PER_MTOK = 0.20
QWEN_INSTRUCT_OUTPUT_USD_PER_MTOK = 0.60
# Agent planner / query-constraint extract — keep 30B for now (easy to bump later)
QWEN_INSTRUCT_30B_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
QWEN_INSTRUCT_30B_INPUT_USD_PER_MTOK = 0.10
QWEN_INSTRUCT_30B_OUTPUT_USD_PER_MTOK = 0.30


def chat_usd_per_mtok(model: str | None) -> tuple[float, float]:
    """Nebius Token Factory in/out USD per 1M tokens for an instruct model."""
    name = model or ""
    if "30B" in name:
        return (
            QWEN_INSTRUCT_30B_INPUT_USD_PER_MTOK,
            QWEN_INSTRUCT_30B_OUTPUT_USD_PER_MTOK,
        )
    return (QWEN_INSTRUCT_INPUT_USD_PER_MTOK, QWEN_INSTRUCT_OUTPUT_USD_PER_MTOK)


EMBED_INPUT_USD_PER_MTOK = 0.01


def embed_usd_per_mtok(model: str | None) -> float:
    """Nebius embedding USD per 1M input tokens. One embedding model is in use."""
    return EMBED_INPUT_USD_PER_MTOK


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def make_nebius_openai_client() -> OpenAI:
    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        raise RuntimeError("NEBIUS_API_KEY is required")
    return OpenAI(
        base_url=NEBIUS_BASE_URL,
        api_key=api_key,
        http_client=httpx.Client(verify=ssl_context(), timeout=120.0),
    )


@dataclass
class UsageBucket:
    """Calls and tokens for one (role, model) pair -- e.g. ("judge", the 235B).

    Priced per model, so a judge call on the 235B and a classify call on the
    30B are no longer charged at one flat rate.
    """

    role: str
    model: str
    kind: str  # "chat" or "embed"
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        if self.kind == "embed":
            return self.prompt_tokens * embed_usd_per_mtok(self.model) / 1_000_000
        usd_in, usd_out = chat_usd_per_mtok(self.model)
        return (self.prompt_tokens * usd_in + self.completion_tokens * usd_out) / 1_000_000

    def add(self, other: UsageBucket) -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "model": self.model,
            "kind": self.kind,
            "calls": self.calls,
            "input_tokens": self.prompt_tokens,
            "output_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class LlmUsage:
    """Accumulated Nebius token usage, in total and per (role, model).

    The flat counters are what existing callers read. `buckets` is what the
    cost report is built from: every `add_chat` / `add_embed` names the role
    that made the call (extractor, judge, classify, embed, ...) and the model
    it ran on, and cost is priced per model from the rate table above.
    """

    chat_prompt_tokens: int = 0
    chat_completion_tokens: int = 0
    embed_prompt_tokens: int = 0
    chat_calls: int = 0
    embed_calls: int = 0
    buckets: dict[tuple[str, str], UsageBucket] = field(default_factory=dict)

    def _bucket(self, role: str, model: str | None, kind: str) -> UsageBucket:
        key = (role, model or "unknown")
        bucket = self.buckets.get(key)
        if bucket is None:
            bucket = self.buckets[key] = UsageBucket(role=role, model=key[1], kind=kind)
        return bucket

    def add_chat(
        self, usage: Any | None, *, role: str = "chat", model: str | None = None
    ) -> None:
        if usage is None:
            return
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        self.chat_calls += 1
        self.chat_prompt_tokens += prompt
        self.chat_completion_tokens += completion
        bucket = self._bucket(role, model, "chat")
        bucket.calls += 1
        bucket.prompt_tokens += prompt
        bucket.completion_tokens += completion

    def add_embed(
        self, usage: Any | None, *, role: str = "embed", model: str | None = None
    ) -> None:
        if usage is None:
            return
        prompt = getattr(usage, "prompt_tokens", None)
        if prompt is None:
            prompt = getattr(usage, "total_tokens", 0)
        prompt = int(prompt or 0)
        self.embed_calls += 1
        self.embed_prompt_tokens += prompt
        bucket = self._bucket(role, model, "embed")
        bucket.calls += 1
        bucket.prompt_tokens += prompt

    def merge(self, other: LlmUsage) -> None:
        self.chat_prompt_tokens += other.chat_prompt_tokens
        self.chat_completion_tokens += other.chat_completion_tokens
        self.embed_prompt_tokens += other.embed_prompt_tokens
        self.chat_calls += other.chat_calls
        self.embed_calls += other.embed_calls
        for bucket in other.buckets.values():
            self._bucket(bucket.role, bucket.model, bucket.kind).add(bucket)

    @property
    def input_tokens(self) -> int:
        return self.chat_prompt_tokens + self.embed_prompt_tokens

    @property
    def output_tokens(self) -> int:
        return self.chat_completion_tokens

    @property
    def cost_usd(self) -> float:
        if self.buckets:
            return sum(b.cost_usd for b in self.buckets.values())
        # Counters set without going through add_*: price at the default rates.
        usd_in, usd_out = chat_usd_per_mtok(None)
        return (
            self.chat_prompt_tokens * usd_in
            + self.chat_completion_tokens * usd_out
            + self.embed_prompt_tokens * embed_usd_per_mtok(None)
        ) / 1_000_000

    def by_role(self) -> list[UsageBucket]:
        """Buckets, most expensive first."""
        return sorted(self.buckets.values(), key=lambda b: (-b.cost_usd, b.role))

    def summary(self, *, prefix: str = "    ") -> str:
        head = (
            f"{prefix}LLM usage: "
            f"in={self.input_tokens} "
            f"(chat_prompt={self.chat_prompt_tokens}, "
            f"embed={self.embed_prompt_tokens}), "
            f"out={self.output_tokens}, "
            f"calls={self.chat_calls} chat / {self.embed_calls} embed, "
            f"cost≈${self.cost_usd:.6f}"
        )
        rows = self.by_role()
        if not rows:
            return head
        width = max(len(b.role) for b in rows)
        lines = [head]
        for b in rows:
            lines.append(
                f"{prefix}  {b.role:<{width}}  {_short_model(b.model):<30} "
                f"calls={b.calls:<5} in={b.prompt_tokens:<8} "
                f"out={b.completion_tokens:<7} ${b.cost_usd:.4f}"
            )
        return "\n".join(lines)

    def report(self, kind: str) -> dict[str, Any]:
        """One run's cost, JSON-serialisable, for the scrape cost log."""
        return {
            "kind": kind,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "calls": self.chat_calls + self.embed_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "by_role": [b.as_dict() for b in self.by_role()],
        }


def _short_model(model: str) -> str:
    return model.rsplit("/", 1)[-1]


COST_LOG_ENV = "SCRAPE_COST_LOG"
DEFAULT_COST_LOG = Path("reports") / "scrape_costs.jsonl"


def record_scrape_cost(
    kind: str, usage: LlmUsage, *, path: Path | None = None
) -> Path | None:
    """Append this run's cost report as one JSON line; return the file written.

    `kind` is the scrape (`scrape-rules`, `scrape-reviews`, ...). Nothing is
    written when no LLM call was made. The file is `SCRAPE_COST_LOG` if set,
    else `reports/scrape_costs.jsonl` (git-ignored).
    """
    if not (usage.chat_calls or usage.embed_calls):
        return None
    target = path or Path(os.environ.get(COST_LOG_ENV) or DEFAULT_COST_LOG)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(usage.report(kind), ensure_ascii=False) + "\n")
    return target


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
        api_key = os.environ.get("NEBIUS_API_KEY")
        if not api_key:
            raise RuntimeError("NEBIUS_API_KEY is required")
        return ChatOpenAI(
            model=self.model,
            api_key=api_key,
            base_url=NEBIUS_BASE_URL,
            temperature=self.temperature,
            http_client=httpx.Client(verify=ssl_context(), timeout=120.0),
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
            usage.add_chat(response.usage, role="amenity_extract", model=self.model)
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
            usage.add_chat(response.usage, role="place_enrich", model=self.model)
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
    INPUT_USD_PER_MTOK = EMBED_INPUT_USD_PER_MTOK

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
            usage.add_embed(resp.usage, role="embed", model=self.MODEL)
        by_index = {item.index: item.embedding for item in resp.data}
        return [by_index[i] for i in range(len(texts))]


class ClaimsEmbeddingLLMClient(EmbeddingLLMClient):
    """Nebius embedding model for claims RAG (same Qwen3-Embedding-8B + 1536 dims)."""


def amenity_llm_clients() -> tuple[ExtractorLLMClient, EmbeddingLLMClient]:
    """Shared OpenAI/httpx client for extract + embed in one scrape run."""
    shared = make_nebius_openai_client()
    return ExtractorLLMClient(shared), EmbeddingLLMClient(shared)
