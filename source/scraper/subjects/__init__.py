"""Canonical subject vocabulary: naming rules, alias matching, resolution."""

from source.scraper.subjects.llm import SubjectAdjudicatorLLMClient
from source.scraper.subjects.naming import (
    normalize_alias,
    predicate_suffix,
    same_predicate,
    to_positive_subject,
)
from source.scraper.subjects.resolve import (
    MATCH_MAX_DISTANCE,
    NEAREST_K,
    Candidate,
    ResolutionTrace,
    SubjectRef,
    SubjectStore,
    ensure_table_name,
    format_trace,
    resolve_subject,
)

__all__ = [
    "SubjectAdjudicatorLLMClient",
    "SubjectRef",
    "SubjectStore",
    "ensure_table_name",
    "Candidate",
    "ResolutionTrace",
    "format_trace",
    "resolve_subject",
    "normalize_alias",
    "predicate_suffix",
    "same_predicate",
    "to_positive_subject",
    "NEAREST_K",
    "MATCH_MAX_DISTANCE",
]
