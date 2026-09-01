"""Split Google reviews into claims and upsert into Postgres.

One review per splitter call (Qwen3-235B). Drops claims with confidence < 0.5.
Does not store aspect or locus. Method takes a dict of reviews; fetching from
Places is a separate step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from source.scraper.amenity_enrichment.llm import (
    QWEN_INSTRUCT_MODEL,
    ClaimsEmbeddingLLMClient,
    LlmUsage,
    _parse_json_payload,
    make_nebius_openai_client,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

SCRAPER_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRAPER_DIR / "config.json"

MIN_CONFIDENCE = 0.5
SPLIT_MAX_TOKENS = 2500
DEFAULT_SOURCE = "google"

SPLIT_SYSTEM = """You split one campsite Google review into atomic claims for RAG.

Output valid JSON only, no markdown.

Rules:
- One review can yield many claims only when they are different site facts
  (broken stall latches vs no hot water vs dirty toilets).
- Do not split on every sentence or newline. One visitor incident = one claim.
- Keep evidence_span as the exact original substring (Hebrew or English).
- text_en is a close translation of that span, written as a standalone sentence.
  Use the rest of THIS review only to resolve pronouns / missing feature
  (e.g. "there was no water" + the review is about נחלים → name the streams).
  Do not invent details that are not in the review.
- The JSON may include a "place" field (the campsite name). Use it only as
  context. Do NOT put the park / site name in text_en (not Hebrew, not English,
  not a transliteration like "Horshet Tal"). Site identity is stored elsewhere.
  Bad: "Toilet paper in the bathroom stalls at Horshet Tal is below standard."
  Good: "Toilet paper in the bathroom stalls is below standard."
- Do NOT generalize or swap in broader categories.
  שירותים / תאי שירותים = toilets / bathroom stalls (not "facilities")
  מקלחות / תאים = showers / shower stalls (not "cabins")
  סוגרים = latches/locks (not just "doors")
  נחל / תעלה = stream / channel (not generic "water")
  בריכה = pool
- Skip generic overall judgments. They are not claims. Omit them
  (do not emit them with low confidence).
  Drop: "the place is excellent", "great place", "in short a disappointment",
  "shameful", "unacceptable", "audacious", star-rating restatements.
  Contrast "great except X" / "excellent in every way except X": keep ONLY X.
  Bad: "The place is excellent in every way except for the restrooms and showers."
  Good: "The restrooms and showers are a problem."
  (evidence_span should be the except-clause, e.g. "חוץ מעניין השירותים והמקלחות")
- Do not wrap a fact in rant or price-sentiment.
  Bad: "It is unacceptable that we paid full price and 80 percent of the channels and streams are dry and closed."
  Good: "80 percent of the channels and streams at the site are dry and closed."
  Drop "we paid full price" / "unacceptable" unless the review states a distinct price fact
  (e.g. a specific fee) separate from the feature.
- Direct experience vs speculation:
  Keep: "I talked to management and they were rude." (witnessed)
  Drop: "management is responsible", "they should stop saving money and open the streams",
  "the park ought to…", blame without a witnessed interaction.
- Dedup within ONE review: if the same fact is repeated (streams closed / dry / still renovation),
  emit ONE claim. Put extra details on that one claim (e.g. "dry and closed for over a year,
  said to be under renovation") — do not emit a second stream-water row for the duration.
  A different fact stays separate (crowding at two remaining spots is not the same as
  "streams are dry").
- One visitor incident = one claim. Supporting beats of the same visit are NOT extra claims:
  turning the cars around, the cashier repeating the rule, asking again and being refused.
  Fold those into the one site fact (or drop them).
  Bad (4 claims): refused with a leashed dog; asked to leash at the gate; 4 cars turned around;
  cashier said it is the rule in all parks.
  Good (1 claim): They were not allowed in with a dog even though it was leashed.
- Keep specific feature claims (shade, hot water, pets, pools, streams, booking, BBQ,
  bungalow rental, mattress rental).
  "Excellent for camping" may stay if camping is the feature; bare "great" must not.
  Skip personal asides that are not about the site ("a dog is part of the family").
- polarity: positive | negative | neutral
- Opposite sentiments are separate claims (do not merge "hot showers" with "no hot water").
- Skip empty reviews. Do not invent facts.
- confidence: 0-1 how sure this row is an atomic site fact with a faithful translation.
  Omit anything you would score below 0.5.

Schema:
{
  "claims": [
    {
      "text_en": str,
      "polarity": "positive" | "negative" | "neutral",
      "evidence_span": str,
      "confidence": number
    }
  ]
}
"""

UPSERT_REVIEW_SQL = """
INSERT INTO reviews (
    campsite_id, source, author, rating, text, published_at, review_uid
) VALUES (
    %(campsite_id)s, %(source)s, %(author)s, %(rating)s, %(text)s,
    %(published_at)s, %(review_uid)s
)
ON CONFLICT (review_uid) DO UPDATE
SET author = EXCLUDED.author,
    rating = EXCLUDED.rating,
    text = EXCLUDED.text,
    published_at = EXCLUDED.published_at
RETURNING id;
"""

DELETE_CLAIMS_FOR_REVIEW_SQL = """
DELETE FROM claims WHERE review_id = %(review_id)s
"""

INSERT_CLAIM_SQL = """
INSERT INTO claims (
    review_id, campsite_id, claim, evidence_span,
    polarity, confidence, embedding
) VALUES (
    %(review_id)s, %(campsite_id)s, %(claim)s, %(evidence_span)s,
    %(polarity)s, %(confidence)s, %(embedding)s
)
"""


def database_url(config: dict | None = None) -> str:
    url = os.environ.get("DATABASE_URL")
    if not url and config:
        url = config.get("database_url")
    if not url:
        url = "postgresql://trippy:trippy@localhost:5432/trippy"
    return url.replace("@db:", "@localhost:")


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def log(msg: str) -> None:
    print(msg, flush=True)


def _sha256(*parts: object) -> str:
    raw = "\0".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def review_uid(
    campsite_id: int,
    source: str,
    author: str | None,
    published_at: datetime | None,
    text: str,
) -> str:
    published = published_at.isoformat() if published_at is not None else ""
    return _sha256(campsite_id, source, author or "", published, text)


def _parse_published_at(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    text = str(value).strip()
    if not text:
        return None
    if len(text) == 10 and text[4] == "-":
        text = f"{text}T00:00:00+00:00"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def normalize_review_dict(raw: dict, *, source: str = DEFAULT_SOURCE) -> dict:
    text = str(raw.get("text") or raw.get("full_review") or "")
    rating = raw.get("rating", raw.get("stars"))
    try:
        rating_i = int(rating) if rating is not None and str(rating).strip() else None
    except (TypeError, ValueError):
        rating_i = None
    published = (
        raw.get("published_at")
        or raw.get("published_utc")
        or raw.get("date")
        or raw.get("time")
    )
    author = raw.get("author") or raw.get("author_name") or raw.get("review_author")
    author_s = str(author).strip() if author else None
    return {
        "source": str(raw.get("source") or source),
        "author": author_s or None,
        "rating": rating_i,
        "text": text,
        "published_at": _parse_published_at(published),
    }


def reviews_from_dict(payload: dict, *, source: str = DEFAULT_SOURCE) -> list[dict]:
    """Accept `{reviews: [...]}` or a mapping of keys → review dicts."""
    if isinstance(payload.get("reviews"), list):
        raw_list = payload["reviews"]
    else:
        raw_list = [
            value
            for value in payload.values()
            if isinstance(value, dict)
            and ("text" in value or "rating" in value or "author" in value)
        ]
    return [normalize_review_dict(item, source=source) for item in raw_list]


def _confidence(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def filter_claims(raw_claims: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for item in raw_claims:
        if not isinstance(item, dict):
            continue
        text_en = str(item.get("text_en") or "").strip()
        if not text_en:
            continue
        conf = _confidence(item.get("confidence"))
        if conf is None or conf < MIN_CONFIDENCE:
            continue
        polarity = str(item.get("polarity") or "").strip().lower()
        if polarity not in {"positive", "negative", "neutral"}:
            polarity = None
        evidence = str(item.get("evidence_span") or "").strip() or None
        kept.append(
            {
                "text_en": text_en,
                "polarity": polarity,
                "evidence_span": evidence,
                "confidence": conf,
            }
        )
    return kept


def split_one_review(
    client: Any,
    review: dict,
    *,
    place: str,
    model: str = QWEN_INSTRUCT_MODEL,
    usage: LlmUsage | None = None,
) -> list[dict]:
    text = (review.get("text") or "").strip()
    if not text:
        return []
    user = json.dumps(
        {
            "place": place,
            "review": {
                "text": text,
                "rating": review.get("rating"),
                "date": (
                    review["published_at"].date().isoformat()
                    if isinstance(review.get("published_at"), datetime)
                    else None
                ),
            },
        },
        ensure_ascii=False,
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=SPLIT_MAX_TOKENS,
        messages=[
            {"role": "system", "content": SPLIT_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    if usage is not None:
        usage.add_chat(response.usage)
    parsed = _parse_json_payload((response.choices[0].message.content or "").strip())
    claims = parsed.get("claims")
    if not isinstance(claims, list):
        wrapped = parsed.get("reviews")
        if isinstance(wrapped, list) and wrapped:
            claims = wrapped[0].get("claims") if isinstance(wrapped[0], dict) else []
        else:
            claims = []
    return filter_claims(claims if isinstance(claims, list) else [])


def upsert_review(cur, *, campsite_id: int, review: dict) -> int:
    uid = review_uid(
        campsite_id,
        review["source"],
        review.get("author"),
        review.get("published_at"),
        review.get("text") or "",
    )
    cur.execute(
        UPSERT_REVIEW_SQL,
        {
            "campsite_id": campsite_id,
            "source": review["source"],
            "author": review.get("author"),
            "rating": review.get("rating"),
            "text": review.get("text") or "",
            "published_at": review.get("published_at"),
            "review_uid": uid,
        },
    )
    row = cur.fetchone()
    return int(row[0])


def replace_claims(
    cur,
    *,
    review_id: int,
    campsite_id: int,
    claims: list[dict],
    embeddings: list[list[float]],
) -> int:
    cur.execute(DELETE_CLAIMS_FOR_REVIEW_SQL, {"review_id": review_id})
    for claim, vector in zip(claims, embeddings, strict=True):
        cur.execute(
            INSERT_CLAIM_SQL,
            {
                "review_id": review_id,
                "campsite_id": campsite_id,
                "claim": claim["text_en"],
                "evidence_span": claim.get("evidence_span"),
                "polarity": claim.get("polarity"),
                "confidence": claim.get("confidence"),
                "embedding": vector,
            },
        )
    return len(claims)


def populate_reviews_and_claims(
    campsite_id: int,
    reviews: dict,
    *,
    conn=None,
    place: str | None = None,
    source: str = DEFAULT_SOURCE,
    chat_client: Any | None = None,
    embedder: ClaimsEmbeddingLLMClient | None = None,
    usage: LlmUsage | None = None,
) -> dict:
    """Upsert reviews from `reviews` and split+embed claims. One review per LLM call."""
    place_name = place or str(reviews.get("name") or reviews.get("place") or "")
    items = reviews_from_dict(reviews, source=source)
    own_conn = conn is None
    if own_conn:
        config = load_config() if CONFIG_PATH.exists() else {}
        conn = psycopg.connect(database_url(config))
    register_vector(conn)
    chat = chat_client or make_nebius_openai_client()
    embed_client = embedder or ClaimsEmbeddingLLMClient(chat)
    llm_usage = usage if usage is not None else LlmUsage()

    review_rows: list[tuple[int, dict, list[dict]]] = []
    total = len(items)
    try:
        with conn.cursor() as cur:
            for i, review in enumerate(items, 1):
                review_id = upsert_review(
                    cur, campsite_id=campsite_id, review=review
                )
                text = (review.get("text") or "").strip()
                if not text:
                    log(f"  review {i}/{total} id={review_id}: empty text, stored only")
                    review_rows.append((review_id, review, []))
                    continue
                log(f"  splitting review {i}/{total} id={review_id}")
                try:
                    claims = split_one_review(
                        chat, review, place=place_name, usage=llm_usage
                    )
                except (ValueError, json.JSONDecodeError) as exc:
                    log(f"    split/parse failed: {exc}")
                    claims = []
                log(f"    {len(claims)} claim(s) kept (conf>={MIN_CONFIDENCE})")
                review_rows.append((review_id, review, claims))

        all_claims = [c for _, _, claims in review_rows for c in claims]
        vectors: list[list[float]] = []
        if all_claims:
            log(f"  embedding {len(all_claims)} claim(s)")
            vectors = embed_client.embed(
                [c["text_en"] for c in all_claims], usage=llm_usage
            )

        offset = 0
        n_claims = 0
        with conn.cursor() as cur:
            for review_id, _review, claims in review_rows:
                chunk = vectors[offset : offset + len(claims)]
                offset += len(claims)
                n_claims += replace_claims(
                    cur,
                    review_id=review_id,
                    campsite_id=campsite_id,
                    claims=claims,
                    embeddings=chunk,
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        if own_conn:
            conn.close()

    result = {
        "campsite_id": campsite_id,
        "reviews": len(items),
        "claims": n_claims,
    }
    log(
        f"Done campsite_id={campsite_id}: "
        f"{result['reviews']} review(s), {result['claims']} claim(s)."
    )
    if llm_usage.chat_calls or llm_usage.embed_calls:
        log(llm_usage.summary(prefix=""))
    return result


def lookup_campsite_id(conn, name: str) -> tuple[int, str]:
    needle = f"%{name.strip()}%"
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, name FROM campsites
            WHERE name ILIKE %s
            ORDER BY id
            """,
            (needle,),
        )
        rows = cur.fetchall()
    if not rows:
        raise SystemExit(f"No campsite matching {name!r}")
    if len(rows) > 1:
        listing = ", ".join(f"{r[0]}:{r[1]}" for r in rows)
        log(f"Multiple matches ({listing}); using id={rows[0][0]}")
    return int(rows[0][0]), str(rows[0][1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a dict of reviews into claims and store them."
    )
    parser.add_argument(
        "--reviews",
        required=True,
        help="JSON file: {name, reviews: [{author, rating, text, published_utc}, ...]}",
    )
    parser.add_argument("--campsite-id", type=int, default=None)
    parser.add_argument(
        "--name",
        default=None,
        help="Campsite name substring if --campsite-id is omitted",
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    args = parser.parse_args()

    payload = json.loads(Path(args.reviews).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("reviews JSON must be an object")

    config = load_config() if CONFIG_PATH.exists() else {}
    with psycopg.connect(database_url(config)) as conn:
        campsite_id = args.campsite_id
        place = str(payload.get("name") or payload.get("place") or "")
        if campsite_id is None:
            lookup = args.name or place
            if not lookup:
                raise SystemExit("Pass --campsite-id or --name (or name in the JSON)")
            campsite_id, db_name = lookup_campsite_id(conn, lookup)
            place = place or db_name
            log(f"Campsite {campsite_id}: {db_name}")
        populate_reviews_and_claims(
            campsite_id,
            payload,
            conn=conn,
            place=place,
            source=args.source,
        )


if __name__ == "__main__":
    main()
