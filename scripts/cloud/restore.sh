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
elif [ -d /var/lib/trippy/backups ]; then
  BACKUP_DIR=/var/lib/trippy/backups
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
  echo "usage: restore.sh <file.dump|s3://bucket/key>" >&2
  exit 2
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

${COMPOSE} cp "$src" db:/tmp/trippy-restore.dump
${COMPOSE} exec -T db pg_restore -U trippy -d trippy --clean --if-exists \
  --exit-on-error -n public /tmp/trippy-restore.dump
${COMPOSE} exec -T db rm -f /tmp/trippy-restore.dump
if [ -n "$cleanup_tmp" ]; then
  rm -f "$cleanup_tmp"
fi
echo "restored public schema from ${1}"
