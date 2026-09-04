"""Match a booking-engine lodging name to an info-site rate-card name."""

from __future__ import annotations

from amenity_enrichment.llm import (
    LlmUsage,
    QWEN_INSTRUCT_30B_MODEL,
    _parse_json_payload,
    make_nebius_openai_client,
)
from openai import OpenAI

SYSTEM_PROMPT = """You match a Hebrew INPA booking lodging name to one parks.org.il rate-card name.

Output valid JSON only, no markdown:
{"name": string | null}

Rules:
- "name" must be copied exactly from the provided rate-card names list, or null.
- Pick the same lodging product (paraphrase, extra place suffix, and עמדה/עמדת count as the same).
- If none of the rate-card names is that product, return null.
- Never invent a name that is not in the list.
"""


class InfoWebsiteNameMatcher:
    """Qwen 30B: closest info-site lodging name for a booking name."""

    MODEL = QWEN_INSTRUCT_30B_MODEL
    TEMPERATURE = 0

    def __init__(
        self,
        client: OpenAI | None = None,
        *,
        model: str | None = None,
    ) -> None:
        self.client = client or make_nebius_openai_client()
        self.model = model or self.MODEL

    def pick_name(
        self,
        booking_name: str,
        listing_names: list[str],
        *,
        usage: LlmUsage | None = None,
    ) -> str | None:
        if not listing_names:
            return None
        numbered = "\n".join(
            f"{i}. {name}" for i, name in enumerate(listing_names, start=1)
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Booking name: {booking_name}\n"
                        f"Rate-card names:\n{numbered}"
                    ),
                },
            ],
            temperature=self.TEMPERATURE,
        )
        if usage is not None:
            usage.add_chat(response.usage, role="listing_match", model=self.model)
        content = response.choices[0].message.content or ""
        data = _parse_json_payload(content)
        picked = data.get("name")
        if picked is None:
            return None
        text = str(picked).strip()
        if text not in listing_names:
            return None
        return text


def match_info_website_name(
    booking_name: str,
    listings: list[tuple[int, str]],
    *,
    matcher: InfoWebsiteNameMatcher | None = None,
    usage: LlmUsage | None = None,
) -> int | None:
    """Exact listing name, else 30B pick. listings are (id, name)."""
    needle = (booking_name or "").strip()
    if not needle:
        return None
    exact = [row_id for row_id, name in listings if name == needle]
    if exact:
        return exact[0]
    if matcher is None or not listings:
        return None
    names = [name for _, name in listings]
    picked = matcher.pick_name(needle, names, usage=usage)
    if picked is None:
        return None
    for row_id, name in listings:
        if name == picked:
            return row_id
    return None
