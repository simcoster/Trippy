#!/bin/sh
# Dump public schema to a local directory and optionally copy to Object Storage.
# pg_dump writes inside the db container; docker compose cp brings the file out
# (PowerShell `>` would corrupt the custom-format dump).
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

KEEP_LOCAL="${TRIPPY_BACKUP_KEEP_LOCAL:-7}"
COMPOSE="docker compose -f ${ROOT}/${COMPOSE_FILE}"
if [ -f "${ROOT}/.env" ]; then
  COMPOSE="${COMPOSE} --env-file ${ROOT}/.env"
  set -a
  # shellcheck disable=SC1091
  . "${ROOT}/.env"
  set +a
fi

mkdir -p "${BACKUP_DIR}"
if [ ! -w "${BACKUP_DIR}" ]; then
  echo "backup.sh: cannot write ${BACKUP_DIR} (gh-actions is not root; use \$HOME/.trippy-backups)" >&2
  exit 1
fi
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
name="trippy-${stamp}.dump"
remote="/tmp/${name}"
dump="${BACKUP_DIR}/${name}"

${COMPOSE} exec -T db pg_dump -U trippy -n public -Fc -f "${remote}" trippy
${COMPOSE} cp "db:${remote}" "${dump}"
${COMPOSE} exec -T db rm -f "${remote}"

size=$(wc -c < "${dump}" | tr -d ' ')
echo "wrote ${dump} (${size} bytes, public schema)"

find "${BACKUP_DIR}" -type f -name 'trippy-*.dump' -mtime "+${KEEP_LOCAL}" -delete

if [ -n "${BACKUP_S3_BUCKET:-}" ]; then
  : "${AWS_ENDPOINT_URL:?AWS_ENDPOINT_URL is required to upload}"
  docker run --rm \
    --env-file "${ROOT}/.env" \
    -v "${dump}:/data/dump:ro" \
    amazon/aws-cli \
    --endpoint-url "${AWS_ENDPOINT_URL}" \
    s3 cp /data/dump "s3://${BACKUP_S3_BUCKET}/postgres/${name}"
  echo "uploaded s3://${BACKUP_S3_BUCKET}/postgres/${name}"
fi
