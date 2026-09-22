"""Embed planner retrieve queries."""

from __future__ import annotations

import contextvars
import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from langsmith import traceable
from pydantic import BaseModel, Field

from source.agent.timing import stage
from source.agent.tracing import tracing_env_on
from source.scraper.amenity_enrichment.llm import ClaimsEmbeddingLLMClient

load_dotenv()

_claims_embedder = ClaimsEmbeddingLLMClient()
QUERY_EMBED_CONCURRENCY = 5


def _query_vec_literal(query: str) -> str:
    with stage("embed"):
        embedding = _claims_embedder.embed([query])[0]
        return "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"


class _EmbedQueryArgs(BaseModel):
    query: str = Field(description="Amenity or place phrase to embed.")


def _run_embed_query_tool(query: str) -> str:
    return _query_vec_literal(query)


embed_query_tool = StructuredTool.from_function(
    func=_run_embed_query_tool,
    name="embed_query",
    description="Embed one planner retrieve query for pgvector search.",
    args_schema=_EmbedQueryArgs,
)


def _use_embed_tool() -> bool:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return tracing_env_on()


def _invoke_embed_query_tool(query: str) -> str:
    return embed_query_tool.invoke({"query": query})


@traceable(name="embed_queries", run_type="tool")
def _query_vec_literals(queries: Iterable[str]) -> dict[str, str]:
    """Embed distinct query statements, up to QUERY_EMBED_CONCURRENCY at a time."""
    unique = list(dict.fromkeys(queries))
    if not unique:
        return {}
    worker = (
        _invoke_embed_query_tool if _use_embed_tool() else _query_vec_literal
    )
    workers = min(QUERY_EMBED_CONCURRENCY, len(unique))
    if workers == 1:
        query = unique[0]
        return {query: worker(query)}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        if worker is _invoke_embed_query_tool:
            futs = [
                pool.submit(contextvars.copy_context().run, worker, query)
                for query in unique
            ]
            literals = [fut.result() for fut in futs]
        else:
            literals = list(pool.map(worker, unique))
    return dict(zip(unique, literals, strict=True))
