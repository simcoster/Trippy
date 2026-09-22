#!/bin/sh
# Restore public schema from a local custom-format dump or s3://bucket/key.
# Leaves experiments (and extensions) alone. Target DB must already have
# vector / pg_trgm from db/init.
set -eu

self=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
case "$self" in
  */scripts/cloud) ROOT="${TRIPPY_ROOT:-$(CDPATH= cd -- "$self/../.." && pwd)}" ;;
  *) ROOT="${TRIPPY_ROOT:-/opt/trippy}" ;;
esac

if [ -n "${TRIPPY_BACKUP_DIR:-}" ]; then
  BACKUP_DIR="$TRIPPY_BACKUP_DIR"
elif [ -d /var/lib/trippy/backups ] && [ -w /var/lib/trippy/backups ]; then
  BACKUP_DIR=/var/lib/trippy/backups
elif [ -n "${HOME:-}" ]; then
  BACKUP_DIR="${HOME}/.trippy-backups"
else
  BACKUP_DIR="${ROOT}/backups"
fi

if [ -n "${TRIPPY_COMPOSE_FILE:-}" ]; then
  COMPOSE_FILE="$TRIPPY_COMPOSE_FILE"
elif [ -d /var/lib/trippy/backups ]; then
  COMPOSE_FILE=docker-compose.prod.yml
else
  COMPOSE_FILE=docker-compose.yml
fi

COMPOSE="docker compose -f ${ROOT}/${COMPOSE_FILE}"
if [ -f "${ROOT}/.env" ]; then
  COMPOSE="${COMPOSE} --env-file ${ROOT}/.env"
  set -a
  # shellcheck disable=SC1091
  . "${ROOT}/.env"
  set +a
fi

src="${1:-}"
if [ -z "$src" ]; then
  echo "usage: restore.sh <file.dump|s3://bucket/key|latest>" >&2
  exit 2
fi

if [ "$src" = "latest" ]; then
  BACKUP_S3_BUCKET=$(printf '%s' "${BACKUP_S3_BUCKET:-}" | tr -d '\r')
  AWS_ENDPOINT_URL=$(printf '%s' "${AWS_ENDPOINT_URL:-}" | tr -d '\r')
  : "${BACKUP_S3_BUCKET:?BACKUP_S3_BUCKET is required to find the latest dump}"
  : "${AWS_ENDPOINT_URL:?AWS_ENDPOINT_URL is required to find the latest dump}"
  listing=$(docker run --rm \
    --env-file "${ROOT}/.env" \
    amazon/aws-cli \
    --endpoint-url "${AWS_ENDPOINT_URL}" \
    s3 ls "s3://${BACKUP_S3_BUCKET}/postgres/")
  name=$(printf '%s\n' "$listing" | awk '{print $NF}' | grep -E '^trippy-.*\.dump$' | sort | tail -n 1)
  if [ -z "$name" ]; then
    echo "restore.sh: no trippy-*.dump under s3://${BACKUP_S3_BUCKET}/postgres/" >&2
    exit 1
  fi
  src="s3://${BACKUP_S3_BUCKET}/postgres/${name}"
  echo "latest dump ${src}"
fi

cleanup_tmp=""
if [ "${src#s3://}" != "$src" ]; then
  : "${AWS_ENDPOINT_URL:?AWS_ENDPOINT_URL is required to download}"
  mkdir -p "${BACKUP_DIR}"
  tmp="${BACKUP_DIR}/restore-download.dump"
  docker run --rm \
    --env-file "${ROOT}/.env" \
    -v "${BACKUP_DIR}:/data" \
    amazon/aws-cli \
    --endpoint-url "${AWS_ENDPOINT_URL}" \
    s3 cp "$src" /data/restore-download.dump
  src="$tmp"
  cleanup_tmp="$tmp"
fi

if [ ! -f "$src" ]; then
  echo "restore.sh: not a file: ${src}" >&2
  exit 2
fi

${COMPOSE} up -d --wait db
# experiments copies public sequences (nextval defaults). --clean cannot drop
# those sequences while the schema is there. It is a disposable copy.
${COMPOSE} exec -T db psql -U trippy -d trippy -v ON_ERROR_STOP=1 \
  -c "DROP SCHEMA IF EXISTS experiments CASCADE"
${COMPOSE} cp "$src" db:/tmp/trippy-restore.dump
${COMPOSE} exec -T db pg_restore -U trippy -d trippy --clean --if-exists \
  --exit-on-error -n public /tmp/trippy-restore.dump
${COMPOSE} exec -T db rm -f /tmp/trippy-restore.dump
if [ -n "$cleanup_tmp" ]; then
  rm -f "$cleanup_tmp"
fi
echo "restored public schema from ${src}"
