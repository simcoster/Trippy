import json
import pytest
from main import telegram_webhook
import psycopg
import os
from openai import OpenAI
from pgvector.psycopg import register_vector
from openai.resources.embeddings import create as _emb_create

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
    with open("test_cases/update_yes_trip.json", "r", encoding="utf-8") as f:
        payload = json.load(f)

    req = fake_request_factory(payload)
    result = await telegram_webhook(req)

    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_webhook_with_non_trip_message(fake_request_factory):
    """Test webhook with a non-trip-planning message."""
    with open("test_cases/update_non_trip.json", "r", encoding="utf-8") as f:
        payload = json.load(f)

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

MODEL = "text-embedding-3-small"  # 1536 dims by default :contentReference[oaicite:4]{index=4}

@pytest.mark.asyncio
async def test_embedding_search_fit_for_kids():
    # Prepare
    prompt = "fit for stargazing"
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    db_url = os.environ.get("DATABASE_URL")
    assert openai_api_key, "OPENAI_API_KEY is required"
    assert db_url, "DATABASE_URL is required"
    client = OpenAI(api_key=openai_api_key)

    resp = _emb_create(
        client.embeddings,
        model=MODEL,
        input=prompt,
        encoding_format="float",
    )

    # Get embedding from OpenAI
    embedding = resp.data[0].embedding
    vec_literal = "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"

    # Connect to DB
    with psycopg.connect(db_url) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            # Find the closest embedding (cosine distance or L2, assuming pgvector <-> operator)
            cur.execute(
                """
                SELECT id, claim_en, embedding <#> %s::vector AS distance
                FROM claims
                ORDER BY embedding <#> %s::vector
                LIMIT 1
                """,
                (vec_literal, vec_literal)
            )
            row = cur.fetchone()
            assert row is not None, "No row found"
            claim_id, claim_text, distance = row
            print(f"Closest claim: {claim_text} (id: {claim_id}, distance: {distance})")
            # Optionally add more thorough checks here
            assert isinstance(claim_text, str)
            assert distance >= 0

