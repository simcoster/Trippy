"""Create missing availability tables and stamp Alembic without nuking campsites."""

import psycopg

DDL = """
CREATE TABLE IF NOT EXISTS accommodation_types (
  id BIGSERIAL PRIMARY KEY,
  name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS availability (
  id BIGSERIAL PRIMARY KEY,
  site_id BIGINT NOT NULL REFERENCES campsites(id) ON DELETE CASCADE,
  start_date DATE NOT NULL,
  end_date DATE NOT NULL,
  accommodation_type_id BIGINT NOT NULL REFERENCES accommodation_types(id) ON DELETE RESTRICT,
  price DOUBLE PRECISION NOT NULL,
  adults_no INTEGER NOT NULL,
  scraped_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT availability_unique_slot UNIQUE (
    site_id, start_date, end_date, accommodation_type_id, adults_no
  )
);

CREATE INDEX IF NOT EXISTS availability_site_dates_idx
  ON availability (site_id, start_date, end_date);

CREATE TABLE IF NOT EXISTS alembic_version (
  version_num VARCHAR(32) NOT NULL PRIMARY KEY
);

DELETE FROM alembic_version;
INSERT INTO alembic_version (version_num) VALUES ('001_initial');
"""

conn = psycopg.connect("postgresql://trippy:trippy@localhost:5432/trippy")
conn.execute(DDL)
conn.commit()
cur = conn.execute(
    "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY 1"
)
print("tables:", [r[0] for r in cur.fetchall()])
print("alembic:", conn.execute("SELECT version_num FROM alembic_version").fetchone()[0])
conn.close()
print("OK")
