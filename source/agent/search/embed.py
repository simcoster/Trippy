"""Embed planner retrieve queries."""

from __future__ import annotations

from collections.abc import Iterable

from dotenv import load_dotenv
from langsmith import traceable

from source.agent.timing import stage
from source.scraper.amenity_enrichment.llm import ClaimsEmbeddingLLMClient

load_dotenv()

_claims_embedder = ClaimsEmbeddingLLMClient()


def _format_vec(embedding: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"


def _query_vec_literal(query: str) -> str:
    with stage("embed"):
        return _format_vec(_claims_embedder.embed([query])[0])


# Planner tests replace this name with a fake vector. Production batches.
_QUERY_VEC_LITERAL = _query_vec_literal


@traceable(name="embed_queries", run_type="tool")
def _query_vec_literals(queries: Iterable[str]) -> dict[str, str]:
    """One embeddings request for every distinct phrase."""
    unique = list(dict.fromkeys(queries))
    if not unique:
        return {}
    if _query_vec_literal is not _QUERY_VEC_LITERAL:
        return {query: _query_vec_literal(query) for query in unique}
    with stage("embed"):
        vectors = _claims_embedder.embed(unique)
    return {
        query: _format_vec(vector)
        for query, vector in zip(unique, vectors, strict=True)
    }
