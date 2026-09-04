"""Subject-name hygiene: snake_case, and never negatively phrased.

A subject name states a predicate positively — `dogs_allowed`,
`towels_included`, `dogs_must_wear_a_muzzle`. Negation lives in
`campsite_rules.polarity`, not in the name, for two reasons: it keeps one
vector per real-world subject (so "are dogs allowed?" retrieves the row either
way), and it keeps near-identical opposites like `dogs_allowed` /
`dogs_not_allowed` out of the same embedding space, where they would be
almost indistinguishable.

Pure module: no I/O, no LLM. The extractor prompt states the rule too; this is
the backstop.
"""

from __future__ import annotations

import re

_WS_RE = re.compile(r"[\s\-]+")
_REPEAT_UNDERSCORE_RE = re.compile(r"_{2,}")
_ILLEGAL_RE = re.compile(r"[^a-z0-9_]+")

# Suffixes: `<subject>_not_allowed` → `<subject>_allowed`, polarity False.
_NEGATIVE_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("_not_allowed", "_allowed"),
    ("_not_permitted", "_permitted"),
    ("_not_included", "_included"),
    ("_not_provided", "_provided"),
    ("_not_available", "_available"),
    ("_disallowed", "_allowed"),
    ("_forbidden", "_allowed"),
    ("_prohibited", "_allowed"),
    ("_banned", "_allowed"),
    ("_unavailable", "_available"),
)

# Prefixes: `no_pets` → `pets_allowed`, polarity False.
_NEGATIVE_PREFIXES: tuple[str, ...] = ("no_", "not_", "never_", "without_")

# Present anywhere and not rewritable into a clean positive.
_UNSALVAGEABLE: tuple[str, ...] = (
    "_not_",
    "cant_",
    "_cant",
    "cannot_",
    "_cannot",
    "_without_",
    "_no_",
    "_never_",
)


def normalize_alias(term: str) -> str:
    """Fold a raw surface form into the alias key: lowercase snake_case."""
    text = _WS_RE.sub("_", (term or "").strip().casefold())
    text = _ILLEGAL_RE.sub("_", text)
    text = _REPEAT_UNDERSCORE_RE.sub("_", text)
    return text.strip("_")


def to_positive_subject(name: str) -> tuple[str | None, bool | None]:
    """Rewrite a negatively-phrased subject name into a positive one.

    Returns `(positive_name, polarity)`:
      - already positive        → `(name, None)`   — caller keeps its own polarity
      - negation stripped       → `(name, False)`  — caller must apply False
      - negation unrewritable   → `(None, False)`  — caller drops the statement

    A `None` polarity means "this name said nothing about polarity", not
    "polarity is unknown"; the caller keeps whatever the extractor gave it.

        dogs_not_allowed        -> ("dogs_allowed", False)
        no_pets                 -> ("pets_allowed", False)
        towels_not_included     -> ("towels_included", False)
        dogs_must_wear_a_muzzle -> ("dogs_must_wear_a_muzzle", None)
        cant_be_without_muzzle  -> (None, False)
    """
    text = normalize_alias(name)
    if not text:
        return None, None

    for suffix, positive in _NEGATIVE_SUFFIXES:
        if text.endswith(suffix):
            stem = text[: -len(suffix)]
            if not stem:
                return None, False
            return _settle(stem + positive), False

    for prefix in _NEGATIVE_PREFIXES:
        if text.startswith(prefix):
            stem = text[len(prefix) :]
            if not stem or stem in {"allowed", "permitted", "included", "provided"}:
                # "not_allowed" with no subject — nothing to name.
                return None, False
            # `no_pets` states a permission, so name the permission.
            if not stem.endswith(("_allowed", "_permitted", "_included", "_provided")):
                stem = f"{stem}_allowed"
            return _settle(stem), False

    if any(token in f"_{text}_" for token in _UNSALVAGEABLE):
        return None, False

    return text, None


def _settle(text: str) -> str | None:
    """Re-check a rewritten name; a leftover negation means give up."""
    cleaned = normalize_alias(text)
    if not cleaned:
        return None
    if any(token in f"_{cleaned}_" for token in _UNSALVAGEABLE):
        return None
    return cleaned


# Whether two names state the same predicate (a permission vs a time vs a
# price) is the judge LLM's call, made with both names and their contexts in
# view; see ADJUDICATE_SYSTEM_PROMPT. A suffix list used to pre-empt it here and
# fragmented the vocabulary wherever a suffix was missing from the list
# (`_until` vs `_end_time`, `_applies` vs `_required`). Do not reintroduce one.

# Two names that pick opposite members of one of these pairs state opposite
# facts and are never one subject. A live ingest merged all four mattress-window
# bounds into `mattress_pickup_start_time` and `child_max_age` into
# `child_min_age` — the sameness judge sees near-identical strings and says yes,
# so antonyms are decided here instead of asked about. False positives only
# over-split, which is the safe direction (see docs/design.md).
ANTONYM_PAIRS: tuple[tuple[str, str], ...] = (
    ("min", "max"),
    ("minimum", "maximum"),
    ("start", "end"),
    ("first", "last"),
    ("early", "late"),
    ("earliest", "latest"),
    ("in", "out"),
    ("entry", "exit"),
    ("pickup", "return"),
    ("arrival", "departure"),
    ("open", "close"),
    ("before", "after"),
)


def opposed(left: str, right: str) -> bool:
    """True when the two names take opposite sides of a known antonym pair."""
    a = set(normalize_alias(left).split("_"))
    b = set(normalize_alias(right).split("_"))
    return any(
        (x in a and y in b) or (y in a and x in b) for x, y in ANTONYM_PAIRS
    )
