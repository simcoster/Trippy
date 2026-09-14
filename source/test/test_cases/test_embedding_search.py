"""Nearest-neighbour claim search against a live embedding (was in test_webhook)."""

import os

import psycopg
import pytest
from pgvector.psycopg import register_vector

from source.scraper.amenity_enrichment.llm import ClaimsEmbeddingLLMClient


@pytest.mark.llm
def test_embedding_search_fit_for_kids():
    # Prepare
    prompt = "fit for stargazing"
    db_url = os.environ.get("DATABASE_URL")
    assert db_url, "DATABASE_URL is required"
    assert os.environ.get("NEBIUS_API_KEY"), "NEBIUS_API_KEY is required"

    embedding = ClaimsEmbeddingLLMClient().embed([prompt])[0]
    vec_literal = "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"

    # Connect to DB
    with psycopg.connect(db_url) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            # Nearest neighbour by negative inner product. The vector goes in as
            # a parameter: a pgvector literal is a *string* — '[1,2,3]'::vector —
            # so interpolating the bare brackets into the SQL is a syntax error
            # at the '['. Every query in source/agent/search.py passes it as a
            # parameter for exactly this reason.
            cur.execute(
                """
                SELECT campsite_id, claim, embedding <#> %(vec)s::vector AS distance
                FROM claims
                WHERE embedding IS NOT NULL
                ORDER BY embedding <#> %(vec)s::vector
                LIMIT 1
                """,
                {"vec": vec_literal},
            )
            row = cur.fetchone()
            assert row is not None, "No row found"
            claim_id, claim_text, distance = row
            print(f"Closest claim: {claim_text} (id: {claim_id}, distance: {distance})")
            assert isinstance(claim_text, str)
            # `<#>` is the NEGATIVE inner product, so a close match is very
            # negative and only an orthogonal one approaches 0.
            assert distance <= 0
