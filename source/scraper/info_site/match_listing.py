"""Match a booking-engine lodging name to an info-site rate-card name."""

from __future__ import annotations

from openai import OpenAI

from source.scraper.amenity_enrichment.llm import (
    QWEN_INSTRUCT_30B_MODEL,
    LlmUsage,
    _parse_json_payload,
    make_nebius_openai_client,
)

SYSTEM_PROMPT = """You match one Hebrew lodging name to the campsite's list of lodging products.

Output valid JSON only, no markdown:
{"name": string | null}

Rules:
- "name" must be copied EXACTLY from the candidate list, or null. Never invent one,
  never repair a typo, never return a name that is not on the list.
- Match the same lodging PRODUCT. These do not change the product:
    a unit or room number      בונגלו עם מזגן מספר 42  =  בונגלו עם מזגן
    singular vs plural         עמדות חניה  =  עמדת חניה
    a rate word                לינה ב-, אמצע שבוע, סופי שבוע וחגים
    a place suffix             ... - חניון צפוני
- These DO change the product, and must never be matched to each other:
    accessible vs not          חושה מונגשת  ≠  חושה
    single vs double           חושה כפולה   ≠  חושה
    equipped vs not            חדר צוות מאובזר  ≠  חדר צוות
    a different structure      בונגלו  ≠  חושה  ≠  אוהל  ≠  קרוואן
- When no candidate is that product, return null. A wrong match is worse than
  none: it attaches prices or vacancies to something the guest will not get.

Examples:
  Name: בונגלו עם מזגן מספר 42
  Candidates: 1. בונגלו עם מזגן  2. בונגלו מונגש עם מזגן
  -> {"name": "בונגלו עם מזגן"}
     (the number identifies one unit; "מונגש" would be a different product)

  Name: לינה בחושה כפולה סופי שבוע וחגים
  Candidates: 1. חושה  2. חושה כפולה  3. חושה עם מזגן שירותים ומקלחת
  -> {"name": "חושה כפולה"}
     (the rate words drop; "כפולה" is part of the product and must be kept)

  Name: חדר צוות עץ
  Candidates: 1. חושה  2. בונגלו עם מזגן  3. עמדת חניה לקרוואן פרטי
  -> {"name": null}
     (no candidate is this product, so nothing is picked)
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
        system_prompt: str | None = None,
        role: str = "listing_match",
    ) -> None:
        self.client = client or make_nebius_openai_client()
        self.model = model or self.MODEL
        # The task is the same either way -- pick one name from a list or say
        # null, never invent -- so the class serves both the rate-card names
        # and the scraped accommodation types; only the framing and the cost
        # role differ.
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.role = role

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
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    # Deliberately neutral: this class matches booking names to
                    # rate-card names, rate-card labels to lodging products, and
                    # booking names to accommodation types. Naming one side
                    # "Rate-card" would be wrong for two of the three.
                    "content": f"Name: {booking_name}\nCandidates:\n{numbered}",
                },
            ],
            temperature=self.TEMPERATURE,
        )
        if usage is not None:
            usage.add_chat(response.usage, role=self.role, model=self.model)
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
