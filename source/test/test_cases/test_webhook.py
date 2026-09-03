import json
import os
from pathlib import Path

import psycopg
import pytest
from pgvector.psycopg import register_vector

from main import telegram_webhook
from source.scraper.amenity_enrichment.llm import ClaimsEmbeddingLLMClient

# Anchored to this file, not to the working directory: the payloads sit beside
# it, and a bare "test_cases/..." only opened when pytest was run from
# source/test.
FIXTURES = Path(__file__).resolve().parent


# Minimal fake Request to pass to the handler
class FakeRequest:
    def __init__(self, payload: dict):
        self._payload = payload

    async def json(self):
        return self._payload


@pytest.fixture
def fake_request_factory():
    """Factory fixture to create FakeRequest instances."""
    def _create(payload: dict):
        return FakeRequest(payload)
    return _create


@pytest.mark.asyncio
async def test_webhook_with_trip_planning_message(fake_request_factory):
    """Test webhook with a trip-planning related message."""
    payload = json.loads((FIXTURES / "update_yes_trip.json").read_text(encoding="utf-8"))

    req = fake_request_factory(payload)
    result = await telegram_webhook(req)

    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_webhook_with_non_trip_message(fake_request_factory):
    """Test webhook with a non-trip-planning message."""
    payload = json.loads((FIXTURES / "update_non_trip.json").read_text(encoding="utf-8"))

    req = fake_request_factory(payload)
    result = await telegram_webhook(req)

    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_webhook_with_trivial_message(fake_request_factory):
    """Test webhook with a trivial message like 'thanks'."""
    # Create a simple trivial message payload
    payload = {
        "update_id": 123456789,
        "message": {
            "message_id": 1,
            "from": {
                "id": 123456789,
                "is_bot": False,
                "first_name": "Test",
                "username": "testuser",
            },
            "chat": {
                "id": 123456789,
                "type": "private",
            },
            "date": 1234567890,
            "text": "thanks!",
        },
    }

    req = fake_request_factory(payload)
    result = await telegram_webhook(req)

    assert result == {"ok": True}

@pytest.mark.asyncio
async def test_webhook_with_planning_message(fake_request_factory):
    """Test webhook with a trivial message like 'thanks'."""
    # Create a simple trivial message payload
    payload = {
        "update_id": 123456789,
        "message": {
            "message_id": 1,
            "from": {
                "id": 123456789,
                "is_bot": False,
                "first_name": "Test",
                "username": "testuser",
            },
            "chat": {
                "id": 123456789,
                "type": "private",
            },
            "date": 1234567890,
            "text": "בואו נלך לאתר קמפינג שקט בדרום אבל עד 150 שקל",
        },
    }

    req = fake_request_factory(payload)
    result = await telegram_webhook(req)

    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_webhook_with_no_message(fake_request_factory):
    """Test webhook with update that has no message."""
    payload = {
        "update_id": 123456789,
    }

    req = fake_request_factory(payload)
    result = await telegram_webhook(req)

    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_webhook_with_no_text(fake_request_factory):
    """Test webhook with message that has no text."""
    payload = {
        "update_id": 123456789,
        "message": {
            "message_id": 1,
            "from": {
                "id": 123456789,
                "is_bot": False,
                "first_name": "Test",
            },
            "chat": {
                "id": 123456789,
                "type": "private",
            },
            "date": 1234567890,
        },
    }

    req = fake_request_factory(payload)
    result = await telegram_webhook(req)

    assert result == {"ok": True}

MODEL = ClaimsEmbeddingLLMClient.MODEL


@pytest.mark.llm
@pytest.mark.asyncio
async def test_embedding_search_fit_for_kids():
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

