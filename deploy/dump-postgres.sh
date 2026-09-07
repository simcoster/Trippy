#!/bin/bash
# Dump Postgres to the data disk (survives VM stop). Optionally copy to Object Storage.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
COMPOSE=(docker compose -f docker-compose.yml -f docker-compose.prod.yml)

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DIR=/var/lib/trippy/backups
mkdir -p "$DIR"
DUMP="$DIR/trippy-${STAMP}.sql.gz"

"${COMPOSE[@]}" exec -T db pg_dump -U trippy -d trippy | gzip > "$DUMP"
echo "wrote $DUMP"

# Keep the last 14 dumps on disk.
ls -1t "$DIR"/trippy-*.sql.gz 2>/dev/null | tail -n +15 | xargs -r rm --

if [[ -n "${TRIPPY_S3_BUCKET:-}" ]]; then
  ENDPOINT="${AWS_ENDPOINT_URL:-https://storage.eu-north1.nebius.cloud}"
  aws s3 cp "$DUMP" "s3://${TRIPPY_S3_BUCKET}/postgres/$(basename "$DUMP")" \
    --endpoint-url "$ENDPOINT"
  echo "uploaded s3://${TRIPPY_S3_BUCKET}/postgres/$(basename "$DUMP")"
fi
