"""Add room_count and clear availability / accommodation_types."""

import psycopg

SQL = """
ALTER TABLE availability
ADD COLUMN IF NOT EXISTS room_count INTEGER NOT NULL DEFAULT 1;

TRUNCATE TABLE availability RESTART IDENTITY;
TRUNCATE TABLE accommodation_types RESTART IDENTITY CASCADE;
"""

conn = psycopg.connect("postgresql://trippy:trippy@localhost:5432/trippy")
conn.execute(SQL)
conn.commit()

cur = conn.execute(
    """
    SELECT
      (SELECT COUNT(*) FROM availability) AS availability_rows,
      (SELECT COUNT(*) FROM accommodation_types) AS accommodation_rows,
      (SELECT column_name FROM information_schema.columns
       WHERE table_name='availability' AND column_name='room_count') AS room_count_col
    """
)
print(cur.fetchone())
conn.close()
print("OK")
