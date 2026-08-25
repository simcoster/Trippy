# Database

Schema is defined in `db/models.py` and migrated with **Alembic**.

## Local nuke-and-pave

While iterating on schema, prefer regenerating from the single initial migration instead of stacking tiny revisions:

```bash
docker compose down -v
docker compose up -d db
# wait until healthy
alembic upgrade head
```

Or against an already-running DB:

```bash
alembic downgrade base
alembic upgrade head
```

After model changes locally: update `db/models.py`, rewrite/replace `alembic/versions/001_initial.py` (or `alembic revision --autogenerate`), then nuke-and-pave.

## Apply migrations

```bash
alembic upgrade head
```

`DATABASE_URL` (if set) overrides `alembic.ini`. Host runs should use `localhost`, not the Docker hostname `db`.
