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


# The trailing token of a subject name is its predicate: `barbecue_allowed` and
# `barbecue_equipment_included` are two facts about one noun, not one subject.
# Embeddings put them almost on top of each other and the adjudicator LLM merges
# them if asked, so the suffix comparison is done in code instead.
# For an amenity, "is it provided" is what polarity records, so these suffixes
# name no predicate of their own: `towels_included` and `towels` are one
# subject, and `electric_hookup_included` is not a second `electric_hookup`.
PROVISION_SUFFIXES: tuple[str, ...] = ("included", "provided", "available")

# For a rule these all ask the same question — may I? — so `late_checkout_allowed`
# and `late_check_out_available` are one subject. Note `available` appears in both
# lists: on an amenity it means "supplied", on a rule it means "permitted", which
# is why the comparison needs to know the category.
PERMISSION_SUFFIXES: tuple[str, ...] = ("allowed", "permitted", "available")
PERMISSION = "permission"

# subject_vectors.category values, kept as plain ints so this module stays free
# of database imports.
CATEGORY_RULE = 2

PREDICATE_SUFFIXES: tuple[str, ...] = (
    "allowed",
    "permitted",
    "required",
    "time",
    "fee",
    "price",
    "age",
    "count",
    "nights",
    "days",
    "limit",
    "deposit",
)


def predicate_suffix(name: str, *, category: int | None = None) -> str | None:
    """The predicate a subject name ends in, or None for a bare noun.

    On a rule, every way of asking "may I?" collapses to `PERMISSION`. On
    anything else a trailing provision word is stripped first, so
    `towels_included` reports the same (absent) predicate as `towels`.
    """
    text = normalize_alias(name)
    if category == CATEGORY_RULE:
        if _ends_with_any(text, PERMISSION_SUFFIXES):
            return PERMISSION
    else:
        text = _strip_suffix(text, PROVISION_SUFFIXES)
    for suffix in PREDICATE_SUFFIXES:
        if text == suffix or text.endswith(f"_{suffix}"):
            return suffix
    return None


def _ends_with_any(text: str, suffixes: tuple[str, ...]) -> bool:
    return any(text == s or text.endswith(f"_{s}") for s in suffixes)


def _strip_suffix(text: str, suffixes: tuple[str, ...]) -> str:
    for suffix in suffixes:
        if text.endswith(f"_{suffix}"):
            return text[: -len(suffix) - 1]
    return text


def same_predicate(left: str, right: str, *, category: int | None = None) -> bool:
    """Can these two names possibly be one subject?

    Only compares predicates, so it says nothing about the nouns —
    `check_in_time` and `check_out_time` both pass and are told apart later.
    """
    return predicate_suffix(left, category=category) == predicate_suffix(
        right, category=category
    )
