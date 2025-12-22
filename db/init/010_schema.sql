CREATE TABLE IF NOT EXISTS claims (
  id BIGSERIAL PRIMARY KEY,
  campsite_id TEXT NOT NULL,
  source TEXT NOT NULL,
  review_author TEXT NULL,
  review_date TEXT NULL,
  lang TEXT NULL,
  claim_he TEXT NULL,
  claim_en TEXT NULL,
  evidence_span TEXT NULL,
  polarity TEXT NULL,
  severity INT NULL,
  confidence REAL NULL,
  claim_uid TEXT UNIQUE NOT NULL,     -- stable id for upsert
  embedding vector(1536)              -- text-embedding-3-small default dim :contentReference[oaicite:2]{index=2}
);

CREATE TABLE IF NOT EXISTS campsites (
  id BIGSERIAL PRIMARY KEY,
  campsite_id TEXT NOT NULL,
  region TEXT NOT NULL,
  price REAL NOT NULL,
  ride_time_from_tlv REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS claim_campsite_idx ON claims(campsite_id);
CREATE INDEX IF NOT EXISTS claim_embedding_idx ON claims USING hnsw (embedding vector_cosine_ops);