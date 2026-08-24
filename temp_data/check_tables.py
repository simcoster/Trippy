import psycopg

conn = psycopg.connect("postgresql://trippy:trippy@localhost:5432/trippy")
cur = conn.cursor()
cur.execute(
    "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY 1"
)
print("tables:", [r[0] for r in cur.fetchall()])
cur.execute(
    "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
    "WHERE table_name='alembic_version')"
)
print("alembic_version exists:", cur.fetchone()[0])
cur.execute("SELECT COUNT(*) FROM campsites")
print("campsites rows:", cur.fetchone()[0])
conn.close()
