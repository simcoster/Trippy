"""Match a booking-engine lodging name to an info-site rate-card name."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from openai import OpenAI

from source.scraper.amenity_enrichment.llm import (
    QWEN_INSTRUCT_MODEL,
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
- ALWAYS pick the closest candidate. Never return null: the rate card and the
  lodging panel are two descriptions of one campsite, so a price belongs to
  something even when the wording is far apart. Say how sure you are in
  `confidence` instead -- that is how you report a poor match, not by refusing.
- confidence: 1.0 the same product beyond doubt; around 0.5 when the words差 but
  nothing else fits better; below 0.3 when you are picking the least bad of a
  bad set. Judge only this pair, not how tidy the list is.

Output shape: {"name": <exact candidate>, "confidence": <0..1>}

Examples:
  Name: בונגלו עם מזגן מספר 42
  Candidates: 1. בונגלו עם מזגן  2. בונגלו מונגש עם מזגן
  -> {"name": "בונגלו עם מזגן", "confidence": 1.0}
     (the number identifies one unit; "מונגש" would be a different product)

  Name: לינה בחושה כפולה סופי שבוע וחגים
  Candidates: 1. חושה  2. חושה כפולה  3. חושה עם מזגן שירותים ומקלחת
  -> {"name": "חושה כפולה", "confidence": 1.0}
     (the rate words drop; "כפולה" is part of the product and must be kept)

  Name: חדר צוות קטן אמצע שבוע
  Candidates: 1. חדר צוות קטן אלון ורקפת  2. חדר גדול נשר ויחמור
  -> {"name": "חדר צוות קטן אלון ורקפת", "confidence": 0.9}
     (the panel names the rooms and the rate card does not; "קטן" still decides
      it, and the room names are no more part of the product than a number is)

  Name: השכרת אוהל קמפינג זוגי כולל מזרנים (עד 2 לנים)
  Candidates: 1. השכרת אוהל קמפינג משפחתי  2. השכרת אוהל קמפינג זוגי
  -> {"name": "השכרת אוהל קמפינג זוגי", "confidence": 1.0}
     (what the rate card adds -- the mattresses, the occupancy -- describes the
      same product; "זוגי" against "משפחתי" is what decides it)

  Name: חדר צוות עץ
  Candidates: 1. חושה  2. בונגלו עם מזגן  3. עמדת חניה לקרוואן פרטי
  -> {"name": "בונגלו עם מזגן", "confidence": 0.15}
     (nothing here is a wooden staff room; pick the nearest structure and let
      the low confidence say the list is missing it)
"""


def strip_brackets(text: str) -> str:
    """`חדר צוות קטן אמצע שבוע (חדרים 3 ו- 4)` -> the same without the brackets.

    They carry no meaning here -- a rate card parenthesises the room numbers,
    the occupancy, whatever it likes -- and on the 30B the answer turned on
    their *order*: `(...)` was wrong 6 times in 7 where `)...(` was right 6 in
    6, on bytes that were otherwise identical. Removing them is measured as
    inert on the 235B (same picks, same confidences, experiments.md §20), so
    this takes away a lever nothing should have been pulling.

    Only the name being matched is stripped, never the candidates: the answer
    has to come back as a string that is on the list.
    """
    return re.sub(r"\s+", " ", text.replace("(", " ").replace(")", " ")).strip()


@dataclass
class MatchCall:
    """One model call, kept whole so a bad match can be read rather than guessed.

    A wrong pick is either the prompt's fault or the model's, and a run report
    cannot say which unless it shows what was actually sent. `flagged` is set by
    the caller, which is the only side that knows whether the answer came back
    uncertain or was forced.
    """

    booking_name: str
    system: str
    user: str
    reply: str
    picked: str | None
    confidence: float | None
    # The rescue pass may name several products for one rate. `picked` stays the
    # first so every reader keeps working; this is the whole answer.
    picked_names: list[str] = field(default_factory=list)
    # "pick", "rescue" or "collision" -- which prompt asked the question. The
    # report reads it rather than guessing from the shape of the answer.
    kind: str = "pick"
    # Filled in by the run, which is the only side that knows: one matcher
    # serves every site, so the calls arrive in one undifferentiated list.
    site: str = ""


MULTI_MATCH_PROMPT = """A first pass could not confidently match this Hebrew rate-card label to one lodging product. Some labels price MORE THAN ONE product.

Output valid JSON only, no markdown:
{"names": [string, ...], "confidence": number}

Rules:
- Every name copied EXACTLY from the candidate list. Never invent one.
- Return every product the label prices. One name is a fine answer -- most
  labels do price exactly one -- and the list is never empty: if nothing fits,
  return the closest single candidate and say so in `confidence`.
- A label that names several rooms is only several products when the candidate
  list holds them separately. If one candidate already covers all of them, that
  candidate alone is the answer.
- confidence: how sure you are of the whole SET you returned.

Examples:
  Name: חדר צוות גדול אמצע שבוע חדרים 5 ו-6
  Candidates: 1. לינת שטח באוהלים פרטיים  2. חדר צוות מאובזר כפול חדר מספר 1-2
              3. חדרי צוות חדרים 3-4  4. חדר צוות מאובזר ומונגש חדר מספר 5
              5. חדר צוות מאובזר חדר מספר 6
  -> {"names": ["חדר צוות מאובזר ומונגש חדר מספר 5", "חדר צוות מאובזר חדר מספר 6"], "confidence": 0.9}
     (rooms 5 and 6 are two separate candidates, so the rate prices both)

  Name: חדר צוות קטן אמצע שבוע חדרים 3 ו- 4
  Candidates: 1. חדר צוות מאובזר כפול חדר מספר 1-2  2. חדרי צוות חדרים 3-4
              3. חדר צוות מאובזר חדר מספר 6
  -> {"names": ["חדרי צוות חדרים 3-4"], "confidence": 1.0}
     (one candidate already covers both rooms; do not split it)

  Name: מתחם pitch עד 4 לנים
  Candidates: 1. לינת שטח באוהלים פרטיים  2. מאהל גדול קבוע
  -> {"names": ["לינת שטח באוהלים פרטיים"], "confidence": 0.3}
     (nothing here is a pitch; the closest single candidate, said quietly)
"""


COLLISION_PROMPT = """Two rate-card labels from one campsite were each matched to the SAME lodging product. They cannot both be right: the rate card prices them on separate lines, so they are separate products.

Pick one candidate for each label. The two picks MUST be different.

Output valid JSON only, no markdown:
{"first": string, "second": string, "confidence": number}

Rules:
- "first" is the product for label A, "second" for label B. Both copied EXACTLY
  from the candidate list, and they must not be equal.
- What separates two products is usually one small word -- כפול, מונגש, קטן,
  גדול, a room number -- while everything around it is shared. Weigh that word
  above how much of the rest overlaps.
- The catalog may word a product quite differently from the rate card: the two
  are separate descriptions of one campsite, so `מתחם כפול בתוך מבנה החאן` and
  `מאהל ... כפול` can be the same thing. Match the product, not the phrasing.
- One of the two may well keep the product they were both matched to. The other
  takes the candidate that the distinguishing word points at.
- confidence: how sure you are of the PAIR, not of either half.
"""


class InfoWebsiteNameMatcher:
    """Qwen 235B: closest info-site lodging name for a booking name.

    On the 30B this was wrong 6 times in 7 on one Khan Be'erot label, always the
    same wrong pick at the same 0.40 -- and right 6 times in 6 on the identical
    string with its brackets swapped, which is not a distinction the answer
    should turn on. The 235B was right 12 times out of 12 across both forms at
    1.00 (experiments.md §20).
    """

    MODEL = QWEN_INSTRUCT_MODEL
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
        # Every call, in order. The run report prints the flagged ones in full.
        self.calls: list[MatchCall] = []

    def pick_name(
        self,
        booking_name: str,
        listing_names: list[str],
        *,
        usage: LlmUsage | None = None,
    ) -> tuple[str | None, float | None]:
        """The closest candidate and how sure the model is, or (None, None).

        A null comes back only when the prompt allows one -- the accommodation
        matcher's does, because a booking unit the lodging panel never listed is
        a real thing and inventing a match for it loses the catalog. The listing
        prompt forbids it: a price always belongs to something.
        """
        if not listing_names:
            return None, None
        booking_name = strip_brackets(booking_name)
        numbered = "\n".join(
            f"{i}. {name}" for i, name in enumerate(listing_names, start=1)
        )
        # Deliberately neutral: this class matches booking names to rate-card
        # names, rate-card labels to lodging products, and booking names to
        # accommodation types. Naming one side "Rate-card" would be wrong for
        # two of the three.
        user_message = f"Name: {booking_name}\nCandidates:\n{numbered}"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=self.TEMPERATURE,
        )
        if usage is not None:
            usage.add_chat(response.usage, role=self.role, model=self.model)
        content = response.choices[0].message.content or ""
        data = _parse_json_payload(content)
        picked = data.get("name")
        confidence = data.get("confidence")
        try:
            confidence = None if confidence is None else float(confidence)
        except (TypeError, ValueError):
            confidence = None
        text = None if picked is None else str(picked).strip()
        # Never a name that is not on the list, however plausible it reads.
        if text is not None and text not in listing_names:
            text = None
        self.calls.append(
            MatchCall(
                booking_name=booking_name,
                system=self.system_prompt,
                user=user_message,
                reply=content,
                picked=text,
                confidence=confidence,
                picked_names=[] if text is None else [text],
                kind="pick",
            )
        )
        return text, confidence


    def pick_names(
        self,
        booking_name: str,
        listing_names: list[str],
        *,
        usage: LlmUsage | None = None,
    ) -> tuple[list[str], float | None]:
        """Every candidate this label prices, for a label the first pass doubted.

        A rate card sometimes prices two products on one line -- Khan Be'erot's
        `חדרים 5 ו-6` is rooms 5 and 6, which the panel lists separately -- and a
        single pick has to be wrong about one of them. Asked only after a poor
        first answer, so the common case still costs one call.

        Names not on the candidate list are dropped rather than repaired, the
        same rule the single pick follows.
        """
        if not listing_names:
            return [], None
        booking_name = strip_brackets(booking_name)
        numbered = "\n".join(
            f"{i}. {name}" for i, name in enumerate(listing_names, start=1)
        )
        user_message = f"Name: {booking_name}\nCandidates:\n{numbered}"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": MULTI_MATCH_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=self.TEMPERATURE,
        )
        if usage is not None:
            usage.add_chat(response.usage, role=f"{self.role}_rescue", model=self.model)
        content = response.choices[0].message.content or ""
        data = _parse_json_payload(content)
        raw = data.get("names")
        names: list[str] = []
        if isinstance(raw, list):
            for item in raw:
                text = str(item).strip()
                if text in listing_names and text not in names:
                    names.append(text)
        confidence = data.get("confidence")
        try:
            confidence = None if confidence is None else float(confidence)
        except (TypeError, ValueError):
            confidence = None
        self.calls.append(
            MatchCall(
                booking_name=booking_name,
                system=MULTI_MATCH_PROMPT,
                user=user_message,
                reply=content,
                picked=names[0] if names else None,
                confidence=confidence,
                picked_names=names,
                kind="rescue",
            )
        )
        return names, confidence


    def pick_pair(
        self,
        label_a: str,
        label_b: str,
        collided_on: str,
        listing_names: list[str],
        *,
        usage: LlmUsage | None = None,
    ) -> tuple[str | None, str | None, float | None]:
        """One product each for two labels that landed on the same one.

        `list_prices` is unique on (product, guest type, rate period, class), so
        two rate lines resolving to one product do not both survive -- the
        second silently overwrites the first. That the clash happened is a fact
        the code establishes on its own, without a confidence to go by, which is
        what makes this worth asking: Tel Arad's two Canaanite structures were
        matched onto one listing at 1.00 and 0.80, and nothing flagged it.

        Returns (None, None, confidence) unless both picks are on the list and
        differ from each other -- an answer that fails either test leaves the
        rows exactly as they were.
        """
        if not listing_names:
            return None, None, None
        numbered = "\n".join(
            f"{i}. {name}" for i, name in enumerate(listing_names, start=1)
        )
        user_message = (
            f"Label A: {strip_brackets(label_a)}\n"
            f"Label B: {strip_brackets(label_b)}\n"
            f"Both were matched to: {collided_on}\n"
            f"Candidates:\n{numbered}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": COLLISION_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=self.TEMPERATURE,
        )
        if usage is not None:
            usage.add_chat(
                response.usage, role=f"{self.role}_collision", model=self.model
            )
        content = response.choices[0].message.content or ""
        data = _parse_json_payload(content)
        first = str(data.get("first") or "").strip() or None
        second = str(data.get("second") or "").strip() or None
        confidence = data.get("confidence")
        try:
            confidence = None if confidence is None else float(confidence)
        except (TypeError, ValueError):
            confidence = None
        usable = (
            first in listing_names and second in listing_names and first != second
        )
        if not usable:
            first = second = None
        self.calls.append(
            MatchCall(
                booking_name=f"{label_a}  ||  {label_b}",
                system=COLLISION_PROMPT,
                user=user_message,
                reply=content,
                picked=first,
                confidence=confidence,
                picked_names=[] if first is None else [first, second],
                kind="collision",
            )
        )
        return first, second, confidence


def match_info_website_name(
    booking_name: str,
    listings: list[tuple[int, str]],
    *,
    full_label: str | None = None,
    matcher: InfoWebsiteNameMatcher | None = None,
    usage: LlmUsage | None = None,
) -> tuple[int | None, float | None]:
    """The listing row a name belongs to, and how sure that is.

    Exact name first, which is certain and costs nothing; otherwise one 235B
    pick. `listings` are (id, name). Confidence is None for an exact hit --
    there is nothing to be unsure about -- and the model's own number otherwise.

    The two stages read different strings on purpose. Only the normalised name
    can equal a catalog entry, so that is what the exact test uses; but the
    normalisation is what drops `(חדרים 3 ו- 4)`, and those numbers are the only
    thing separating four staff rooms at Khan Be'erot. `full_label` is the
    unnormalised rate-card label, and it is what the model is shown when the
    free path misses (experiments.md §19).
    """
    needle = (booking_name or "").strip()
    if not needle:
        return None, None
    exact = [row_id for row_id, name in listings if name == needle]
    if exact:
        return exact[0], None
    if matcher is None or not listings:
        return None, None
    names = [name for _, name in listings]
    asked = (full_label or "").strip() or needle
    picked, confidence = matcher.pick_name(asked, names, usage=usage)
    if picked is None:
        return None, confidence
    for row_id, name in listings:
        if name == picked:
            return row_id, confidence
    return None, confidence


def rescue_info_website_names(
    full_label: str,
    listings: list[tuple[int, str]],
    *,
    matcher: InfoWebsiteNameMatcher,
    usage: LlmUsage | None = None,
) -> tuple[list[int], float | None]:
    """The listing rows a doubted label prices, possibly more than one.

    Only called when the first pass came back below `UNCERTAIN_BELOW` or refused
    outright, so a confident single match never pays for it.
    """
    names = [name for _, name in listings]
    picked, confidence = matcher.pick_names(full_label, names, usage=usage)
    by_name = {name: row_id for row_id, name in listings}
    return [by_name[name] for name in picked if name in by_name], confidence
