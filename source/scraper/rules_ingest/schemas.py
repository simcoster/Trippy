"""Pydantic payloads for the rule extractor."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, field_validator

from db.models import QualifierUnit, SubjectCategory

# The names the extractor writes; anything else falls back to NONE.
UNIT_BY_NAME: dict[str, QualifierUnit] = {
    "none": QualifierUnit.NONE,
    "count": QualifierUnit.COUNT,
    "hour_of_day": QualifierUnit.HOUR_OF_DAY,
    "nights": QualifierUnit.NIGHTS,
    "days": QualifierUnit.DAYS,
    "years": QualifierUnit.YEARS,
    "ils": QualifierUnit.ILS,
    "meters": QualifierUnit.METERS,
    "percent": QualifierUnit.PERCENT,
}


# The predicates the extractor prompt allows a rule name to end in, quoted from
# it: "predicate (rules only) is the LAST part of the name and is one of exactly
# ... Never coin another predicate."
#
# This list is in code, which `llm-decides-semantics` normally forbids, and the
# `PREDICATE_SUFFIXES` guard it names as its BAD example looked like this. The
# difference, and the reason Omri agreed to it: that guard decided which
# candidate pairs the judge was allowed to *compare*, which is a judgement about
# meaning. This decides only whether the model kept to an output contract the
# prompt states exhaustively -- the same kind of check `_coerce_unit` and
# `_coerce_category` already make on the other fields of this reply. It never
# rejects a statement; it clears a category the model has already contradicted,
# so the resolver searches on the evidence instead of on a broken label.
RULE_PREDICATES = (
    "_allowed",
    "_required",
)


def miscategorised_rule(subject: str, category: int | None) -> bool:
    """A `boolean_rule` whose name coins a predicate the prompt does not allow.

    Measured over 18 sites: 19 of 431 rows, two shapes. `tent_setup` is a
    perfectly good *amenity* name mislabelled a rule, and `room_assignment`
    asserts nothing at all and disagreed with itself across three sites.
    """
    if category != int(SubjectCategory.BOOLEAN_RULE):
        return False
    return not subject.casefold().endswith(RULE_PREDICATES)


class RuleStatement(BaseModel):
    """One extracted fact about a campsite."""

    subject: str
    # The extractor reads the sentence, so it knows whether it is stating a
    # provision, a permission or a number far better than a classifier shown
    # one word. Amenities, boolean rules and numeric rules are searched apart,
    # so a permission is never a merge candidate for a deadline on the same
    # topic. None when the model omitted it: search every category rather than
    # assert a category that may be wrong.
    category: int | None = None
    polarity: bool | None = None
    qualifier: Decimal | None = None
    qualifier_unit: int = int(QualifierUnit.NONE)
    evidence_span: str | None = None
    confidence: float | None = None

    @field_validator("subject")
    @classmethod
    def _require_subject(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("subject must not be empty")
        return text

    @field_validator("category", mode="before")
    @classmethod
    def _coerce_category(cls, value: object) -> int | None:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            text = value.strip().casefold()
            if text.startswith("amenit"):
                return int(SubjectCategory.AMENITY)
            if text.startswith("bool"):
                return int(SubjectCategory.BOOLEAN_RULE)
            if text.startswith("num"):
                return int(SubjectCategory.NUMERIC_RULE)
            if not text.isdigit():
                # A bare "rule" no longer says which kind; None searches everything.
                return None
        try:
            number = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return number if number in {int(c) for c in SubjectCategory} else None

    @field_validator("qualifier", mode="before")
    @classmethod
    def _coerce_qualifier(cls, value: object) -> object:
        if value is None or value == "":
            return None
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            return None

    @field_validator("qualifier_unit", mode="before")
    @classmethod
    def _coerce_unit(cls, value: object) -> int:
        if isinstance(value, str):
            return int(UNIT_BY_NAME.get(value.strip().casefold(), QualifierUnit.NONE))
        try:
            number = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return int(QualifierUnit.NONE)
        units = {int(u) for u in QualifierUnit}
        return number if number in units else int(QualifierUnit.NONE)


class RuleExtract(BaseModel):
    """The extractor's whole reply for one section."""

    statements: list[RuleStatement] = []
