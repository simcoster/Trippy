#!/bin/sh
# Dump Postgres to a local directory and optionally copy to Object Storage.
set -eu

ROOT="${TRIPPY_ROOT:-/opt/trippy}"
BACKUP_DIR="${TRIPPY_BACKUP_DIR:-/var/lib/trippy/backups}"
KEEP_LOCAL="${TRIPPY_BACKUP_KEEP_LOCAL:-7}"
COMPOSE="docker compose -f ${ROOT}/docker-compose.prod.yml --env-file ${ROOT}/.env"

if [ -f "${ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "${ROOT}/.env"
  set +a
fi

mkdir -p "${BACKUP_DIR}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
dump="${BACKUP_DIR}/trippy-${stamp}.dump"

${COMPOSE} exec -T db pg_dump -U trippy -Fc trippy > "${dump}"
echo "wrote ${dump}"

find "${BACKUP_DIR}" -type f -name 'trippy-*.dump' -mtime "+${KEEP_LOCAL}" -delete

if [ -n "${BACKUP_S3_BUCKET:-}" ]; then
  aws --endpoint-url "${AWS_ENDPOINT_URL:?AWS_ENDPOINT_URL is required to upload}" \
    s3 cp "${dump}" "s3://${BACKUP_S3_BUCKET}/postgres/$(basename "${dump}")"
fi
