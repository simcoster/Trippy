import json
import os
import hashlib
import re
from typing import Any, Dict, Iterable, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from openai.resources.embeddings import create as _emb_create

import psycopg
from pgvector.psycopg import register_vector

load_dotenv()

MODEL = "text-embedding-3-small"  # 1536 dims by default :contentReference[oaicite:4]{index=4}

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
DB_URL = os.environ["DATABASE_URL"]


def stable_uid(*parts: str) -> str:
    """Create a stable ID so reruns UPSERT cleanly."""
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def iter_claim_rows(data: Any) -> Iterable[Dict[str, Any]]:
    """
    Flexible extractor:
    - If your JSON is already a list of claim dicts, yield them.
    - If your JSON is a list of reviews each containing "claims": [...], flatten them.
    Adjust here if your structure differs.
    """
    if isinstance(data, list):
        # case A: list of claims
        if data and isinstance(data[0], dict) and ("claim_he" in data[0] or "claim_en" in data[0] or "claim" in data[0]):
            for c in data:
                yield c
            return

        # case B: list of reviews with nested claims
        for review in data:
            if not isinstance(review, dict):
                continue
            claims = review.get("claims") or []
            for c in claims:
                row = dict(c)
                # inherit review-level metadata if present
                for k in ["location_name", "source", "review_author", "review_date"]:
                    if k not in row and k in review:
                        row[k] = review[k]
                yield row
        return

    raise ValueError("Unsupported JSON structure. Expected list at top level.")


EMBEDDING_STRINGS_TO_REMOVE = [
    "(per this review)",
    "(per this reviewer)",  
    "(לפי המבקר)",      
    "(לפי המבקרת)",      
    "(לפי החוויה של המבקרים בביקורת)"
]

def clean_for_embedding(text: str) -> str:
    cleaned = text.strip()
    for s in EMBEDDING_STRINGS_TO_REMOVE:
        cleaned = cleaned.replace(s, "")
    return cleaned

def pick_text_for_embedding(row: Dict[str, Any]) -> str:
    """
    Returns (text, lang).
    Prefer original Hebrew claim if present; fallback to English; fallback to generic 'claim'.
    """
    claim =""
    if row.get("claim_he"):
        claim = row["claim_he"]
        lang = "he"
    if row.get("claim_en"):
        claim =  row["claim_en"]
        lang = "en"
    return clean_for_embedding(claim), lang 


def embed_texts(texts: List[str], batch_size: int = 128) -> List[List[float]]:
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        # Work around a bug in the installed OpenAI SDK where the Embeddings
        # resource instance is missing a bound .create() method by directly
        # calling the generated create(...) function and passing the resource.
        resp = _emb_create(
            client.embeddings,
            model=MODEL,
            input=chunk,
            encoding_format="float",
        )
        # response.data is in the same order as inputs
        embeddings.extend([d.embedding for d in resp.data])
    return embeddings


def upsert_claims(conn: psycopg.Connection, rows: List[Dict[str, Any]], vectors: List[List[float]]) -> None:
    sql = """
    INSERT INTO claims (
      campsite_id, source, review_author, review_date, lang,
      claim_he, claim_en, evidence_span, polarity, severity, confidence,
      claim_uid, embedding
    )
    VALUES (
      %(campsite_id)s, %(source)s, %(review_author)s, %(review_date)s, %(lang)s,
      %(claim_he)s, %(claim_en)s, %(evidence_span)s, %(polarity)s, %(severity)s, %(confidence)s,
      %(claim_uid)s, %(embedding)s
    )
    ON CONFLICT (claim_uid) DO UPDATE SET
      campsite_id   = EXCLUDED.campsite_id,
      source        = EXCLUDED.source,
      review_author = EXCLUDED.review_author,
      review_date   = EXCLUDED.review_date,
      lang          = EXCLUDED.lang,
      claim_he      = EXCLUDED.claim_he,
      claim_en      = EXCLUDED.claim_en,
      evidence_span = EXCLUDED.evidence_span,
      polarity      = EXCLUDED.polarity,
      severity      = EXCLUDED.severity,
      confidence    = EXCLUDED.confidence,
      embedding     = EXCLUDED.embedding
    ;
    """

    payloads = []
    for row, vec in zip(rows, vectors):
        text, lang = pick_text_for_embedding(row)

        location_name = str(row.get("location_name") or "unknown")
        source = str(row.get("source") or "unknown")
        author = (row.get("review_author") or row.get("author") or None)
        rdate = (row.get("review_date") or row.get("date") or None)

        # stable uid: campsite + source + author + embedded text
        uid = stable_uid(location_name, source, str(author or ""), text)

        payloads.append({
            "campsite_id": location_name,
            "source": source,
            "review_author": author,
            "review_date": rdate,
            "lang": row.get("lang") or lang,
            "claim_he": row.get("claim_he"),
            "claim_en": row.get("claim_en"),
            "evidence_span": row.get("evidence_span"),
            "polarity": row.get("polarity"),
            "severity": row.get("severity"),
            "confidence": row.get("confidence"),
            "claim_uid": uid,
            "embedding": vec,
        })

    with conn.cursor() as cur:
        cur.executemany(sql, payloads)
    conn.commit()


def main() -> None:
    import csv

    # Read the CSV file and extract all claims from the "claims" JSON column
    with open(r"temp_data\RAG - 6 campsites - Sheet1.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        claims_data = []
        for row in reader:
            claims_json = row.get("claims")
            if claims_json:
                try:
                    # Excel/Sheets export sometimes inserts non‑breaking spaces (U+00A0)
                    # and other weird whitespace that the JSON parser does NOT treat
                    # as valid whitespace, which leads to:
                    # json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes
                    #
                    # Normalize the string before json.loads:
                    fixed = claims_json.replace("\xa0", " ")
                    fixed = re.sub(r"\s+", " ", fixed).strip()
                    claims = json.loads(fixed)
                    if isinstance(claims, list):
                        claims_data.extend(claims)
                    elif isinstance(claims, dict):
                        claims_data.append(claims)
                except Exception as e:
                    print(f"Error parsing claims JSON: {e}")

    rows = [r for r in iter_claim_rows(claims_data)]

    # filter empty-text claims
    embed_inputs: List[str] = []
    kept_rows: List[Dict[str, Any]] = []
    for r in rows:
        text, _ = pick_text_for_embedding(r)
        if text:
            embed_inputs.append(text)
            kept_rows.append(r)

    print(f"Loaded {len(rows)} claim-ish rows; embedding {len(kept_rows)} with text.")

    vectors = embed_texts(embed_inputs)

    with psycopg.connect(DB_URL) as conn:
        register_vector(conn)
        upsert_claims(conn, kept_rows, vectors)

    print("Done: upserted claims with embeddings.")


if __name__ == "__main__":
    main()
