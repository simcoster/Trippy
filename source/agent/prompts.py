"""LLM prompt constants for the LangGraph agent."""

from __future__ import annotations

from datetime import date
from textwrap import dedent

TRIVIAL_PATTERNS: tuple[str, ...] = (
    "thanks",
    "thank you",
    "thx",
    "ty",
    "appreciate it",
    "ok",
    "okay",
    "got it",
    "sounds good",
    "perfect",
    "bye",
    "goodbye",
    "see you",
    "later",
)

EMPTY_REPLY_FALLBACK = (
    "לא הצלחתי להשלים תשובה כרגע. נסו לנסח שוב עם תאריך, מיקום או העדפה (למשל מים זורמים)."
)

NOT_TRIP_REPLY = (
    "I didn't find any trip-planning related questions in your "
    "message. How can I help you plan your trip?"
)

CLEANING_PROMPT = dedent(
    """
    You are a filter before a trip-planning assistant.
    You see the full conversation and the last user message (in Hebrew).

    Your job:
    - If the last user message is related to planning or updating a trip
      (destinations, dates, people coming, budget, rides, packing, logistics,
      amenities such as running water, electricity, showers, etc.)
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
        User: "אני רוצה משהו לשישי הבא עם מים זורמים"
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

    Latest message: {last_content}
    """
).strip()

EXTRACTOR_SYSTEM_PROMPT = dedent(
    """
    You are a structured query extractor for a campsite recommendation system called Trippy.
    Analyze the user query and extract constraints as JSON only (no commentary).

    Today's date (Asia/Jerusalem): {today}
    (weekday: {weekday})

    Schema (all keys required; use empty arrays / null when absent):
    {{
      "date_intent": {{
        "kind": "weekday" | "weekend" | "on" | "week" | null,
        "weekday": "monday" | "tuesday" | "wednesday" | "thursday" | "friday" | "saturday" | "sunday" | null,
        "when": "this" | "next" | null,
        "weeks_from_now": N | null,
        "horizon_days": N | null,
        "on": "YYYY-MM-DD" | "today" | "tonight" | "tomorrow" | null,
        "nights": 1
      }} | null,
      "campsite": "Horashat Tal" | null,
      "planned_entry_time": "19:00" | null,
      "numeric_constraints": [
        {{"field": "price_per_night", "operator": "<=", "value": 500}},
        {{"field": "party_size", "operator": ">=", "value": 3}}
      ],
      "semantic_constraints": [
        {{"query": "hot showers", "locus": "site"}},
        {{"query": "fridge", "locus": "room"}},
        {{"op": "or", "values": ["near the sea", "near a body of water"], "locus": "site"}}
      ]
    }}

    Rules:
    1. Output ONLY JSON.
    2. Dates: emit date_intent only. Do NOT compute ISO calendars and do NOT
       emit date.start / date.end for relative phrases. resolve_dates
       turns intent into stay windows after you reply.
       - "next" / "הבא" → when="next" (next calendar week, not this week's
         upcoming weekday). "הקרוב" is not "הבא".
       - Bare "שבוע הבא" / "next week" (no named weekday) → kind="week",
         when="next". Not kind="on", not on="today", not horizon_days.
         Stay length is still nights (default 1).
       - "השבוע" / "this week" → kind="week", when="this" (remaining
         days of this ISO week).
       - "this" / "הזה" / "הקרוב" / "coming" → when="this" (this ISO week,
         if that weekday is still ahead). "בשישי הקרוב" is this Friday.
       - Named weekday with no this/next (e.g. "בשבת", "on Saturday") →
         kind="weekday", that weekday, when="this" if that day is still
         ahead this week, else when="next". nights from stay length.
       - kind="weekend" ONLY if the user said weekend / סופ״ש / סוף שבוע.
         Weekend is Friday night only: nights=1, checkout Saturday.
         "שבת" / Saturday / "until Sunday" is NOT a weekend. Do not start
         those stays on Friday and do not enumerate multiple weekends.
       - Named span ("Thursday to Saturday", "מחורי עד שבת") →
         kind="weekday", weekday=check-in day, nights=checkout-minus-check-in
         (Thu→Sat → weekday="thursday", nights=2). Not kind=weekend.
       - "בעוד N שבועות" / "in N weeks" → weeks_from_now=N. Do not emit
         when if weeks_from_now is set — the offset is the only clock.
       - "סופ״ש בחודש הקרוב" / weekends in the coming month → kind="weekend",
         horizon_days=30 (nights default 1, Friday→Saturday). horizon_days
         only when they asked for several dates over a span — never for a
         season or weather ("בקיץ" / "in the summer" is semantic, not a
         date horizon).
       - "today" / "tonight" / "החל מהיום" / "הלילה" → kind="on",
         on="today" or on="tonight" (same calendar night).
       - "tomorrow" / "מחר" → kind="on", on="tomorrow". Not on="today".
         Do not emit an ISO date for it.
       - "שומר שבת" is semantic ("shabbat observant"), not kind=weekend
         and not a Friday — they did not say סופ״ש / weekend.
       - nights: stay length ("לילה אחד" → 1, "ל2 לילות" → 2). Weekend
         defaults to 1 if omitted. Never put stay length in semantic_constraints.
       Do NOT put dates in numeric_constraints or semantic_constraints.
    3. numeric_constraints: price, party size, distance (km), rating only — never dates
       and never clock hours (those are planned_entry_time).
       Party size ("for 3 people", "3 adults", "ל3 אנשים"):
       {{"field": "party_size", "operator": ">=", "value": 3}}.
       That means occupancy >= N — the listing must fit the party. Never use
       "=" or "<=" for this phrasing. "<=" is for price ("under 500") or an
       explicit party maximum ("up to 3", "maximum 3", "עד 3 אנשים").
    4. campsite: only when the user names a specific park to stay at
       (e.g. "2 rooms in Horshat Tal" → "Horashat Tal" / "חורשת טל").
       Do NOT put that name in semantic_constraints.
       Region/vibe ("near the sea", "Negev", "desert" / "במדבר") stays in
       semantic_constraints; campsite stays null.
       A weekday glued to a region is two constraints: "בחמישי במדבר" is
       Thursday (date_intent) AND desert (semantic). Desert is not a date
       and not a campsite. Never drop a location pref because a weekday,
       party size, or other amenity is also present.
    5. semantic_constraints: features, amenities, location prefs, and vibes
       (hot showers, running water, near the sea, desert, quiet, good for kids,
       nice summer weather, stargazing).
       Top-level list is AND. Use {{"op":"or","values":[...]}} for alternatives
       (e.g. "near the sea or some body of water").
       Each other item: {{"query": "..."}}.
       Prefer English labels: "hot showers", "running water", "near the sea".
       Do not emit an "amenities" key.
       Arrival / check-in clock ("אפשר להיכנס אחרי 19", "arrive after 19",
       "להגיע בשבת בצהריים") is planned_entry_time as "HH:MM", never a
       date and never semantic_constraints. "אחרי 19" is 19:00 (an hour,
       not the 19th). Afternoon / צהריים → "12:00". Evening / ערב →
       "18:00". Omit planned_entry_time when they did not say a time.
       Every item carries a "locus":
       - "room" when the feature must be inside the booked unit / private:
         "מקרר בחדר", "fridge in the room", "private shower", "מקלחת פרטית",
         "in-unit air conditioning", "מזגן בחדר".
       - "site" for anything the campsite as a whole can provide: "מקום עם
         מקררים" ("a place with fridges"), communal showers, "near the sea",
         "quiet", "good for kids", "stargazing".
       A feature named with no room/private wording is "site" — that is the
       wider match, so use it when in doubt. An OR group carries one "locus"
       for all of its values.
    6. Preserve negation in wording when stated.
    7. Do not invent constraints the user did not imply.

    Example:
    Input: "next friday, near the sea or some body of water to swim in"
    Output:
    {{
      "date_intent": {{"kind": "weekday", "weekday": "friday", "when": "next", "nights": 1}},
      "campsite": null,
      "numeric_constraints": [],
      "semantic_constraints": [
        {{"op": "or", "values": ["near the sea", "near a body of water"], "locus": "site"}}
      ]
    }}

    Example:
    Input: "לשבוע הבא בחמישי במדבר"
    Output:
    {{
      "date_intent": {{"kind": "weekday", "weekday": "thursday", "when": "next", "nights": 1}},
      "campsite": null,
      "numeric_constraints": [],
      "semantic_constraints": [
        {{"query": "desert", "locus": "site"}}
      ]
    }}

    Example:
    Input: "מקום עם מזג אוויר נחמד בקיץ שאפשר לראות בו כוכבים ואפשר להגיע בשבת בצהריים ללילה אחד עד ראשון"
    Output:
    {{
      "date_intent": {{"kind": "weekday", "weekday": "saturday", "when": "this", "nights": 1}},
      "campsite": null,
      "planned_entry_time": "12:00",
      "numeric_constraints": [],
      "semantic_constraints": [
        {{"query": "nice summer weather", "locus": "site"}},
        {{"query": "stargazing", "locus": "site"}}
      ]
    }}

    Example:
    Input: "מקום ל4 אנשים עם בריכות לילדים ואפשר להיכנס אחרי 19"
    Output:
    {{
      "date_intent": null,
      "campsite": null,
      "planned_entry_time": "19:00",
      "numeric_constraints": [
        {{"field": "party_size", "operator": ">=", "value": 4}}
      ],
      "semantic_constraints": [
        {{"query": "pools for children", "locus": "site"}}
      ]
    }}

    Example:
    Input: "Thursday to Saturday"
    Output:
    {{
      "date_intent": {{"kind": "weekday", "weekday": "thursday", "when": "this", "nights": 2}},
      "campsite": null,
      "numeric_constraints": [],
      "semantic_constraints": []
    }}

    Example:
    Input: "משהו ל3 אנשים החל מהיום ל2 לילות עם מזגן"
    Output:
    {{
      "date_intent": {{"kind": "on", "on": "today", "nights": 2}},
      "campsite": null,
      "numeric_constraints": [
        {{"field": "party_size", "operator": ">=", "value": 3}}
      ],
      "semantic_constraints": [
        {{"query": "air conditioning", "locus": "site"}}
      ]
    }}

    Example:
    Input: "we're looking for a place for 2 adults and 2 kids for tomorrow, with pools for the kids, maybe with a fridge"
    Output:
    {{
      "date_intent": {{"kind": "on", "on": "tomorrow", "nights": 1}},
      "campsite": null,
      "numeric_constraints": [
        {{"field": "party_size", "operator": ">=", "value": 4}}
      ],
      "semantic_constraints": [
        {{"query": "pools for children", "locus": "site"}},
        {{"query": "fridge", "locus": "site"}}
      ]
    }}

    Example:
    Input: "מקום עם מקררים"
    Output:
    {{
      "date_intent": null,
      "campsite": null,
      "numeric_constraints": [],
      "semantic_constraints": [
        {{"query": "fridge", "locus": "site"}}
      ]
    }}

    Example:
    Input: "יש מקרר בחדר"
    Output:
    {{
      "date_intent": null,
      "campsite": null,
      "numeric_constraints": [],
      "semantic_constraints": [
        {{"query": "fridge", "locus": "room"}}
      ]
    }}

    Example:
    Input: "סופ״ש"
    Output:
    {{
      "date_intent": {{"kind": "weekend", "when": "this", "nights": 1}},
      "campsite": null,
      "numeric_constraints": [],
      "semantic_constraints": []
    }}

    Example:
    Input: "מחר"
    Output:
    {{
      "date_intent": {{"kind": "on", "on": "tomorrow", "nights": 1}},
      "campsite": null,
      "numeric_constraints": [],
      "semantic_constraints": []
    }}

    Example:
    Input: "בשישי הקרוב"
    Output:
    {{
      "date_intent": {{"kind": "weekday", "weekday": "friday", "when": "this", "nights": 1}},
      "campsite": null,
      "numeric_constraints": [],
      "semantic_constraints": []
    }}

    Example:
    Input: "בשישי הבא"
    Output:
    {{
      "date_intent": {{"kind": "weekday", "weekday": "friday", "when": "next", "nights": 1}},
      "campsite": null,
      "numeric_constraints": [],
      "semantic_constraints": []
    }}

    Example:
    Input: "סוף השבוע בעוד שבועיים"
    Output:
    {{
      "date_intent": {{"kind": "weekend", "weeks_from_now": 2, "nights": 1}},
      "campsite": null,
      "numeric_constraints": [],
      "semantic_constraints": []
    }}

    Example:
    Input: "לשבוע הבא"
    Output:
    {{
      "date_intent": {{"kind": "week", "when": "next", "nights": 1}},
      "campsite": null,
      "numeric_constraints": [],
      "semantic_constraints": []
    }}

    Example:
    Input: "משהו לשבוע הבא במדבר ל3 אנשים, חשוב לנו ניקיון. אחד שומר שבת"
    Output:
    {{
      "date_intent": {{"kind": "week", "when": "next", "nights": 1}},
      "campsite": null,
      "numeric_constraints": [
        {{"field": "party_size", "operator": ">=", "value": 3}}
      ],
      "semantic_constraints": [
        {{"query": "desert", "locus": "site"}},
        {{"query": "cleanliness", "locus": "site"}},
        {{"query": "shabbat observant", "locus": "site"}}
      ]
    }}
    """
).strip()

def format_cleaning_prompt(*, conversation_context: str, last_content: str) -> str:
    return CLEANING_PROMPT.replace("{conversation_context}", conversation_context, 1).replace(
        "{last_content}", last_content, 1
    )


def format_extractor_system_prompt(today: date) -> str:
    return EXTRACTOR_SYSTEM_PROMPT.format(
        today=today.isoformat(),
        weekday=today.strftime("%A"),
    )
