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

# Comments and Windows CRLF in .env leave a trailing CR on values.
BACKUP_S3_BUCKET=$(printf '%s' "${BACKUP_S3_BUCKET:-}" | tr -d '\r')
AWS_ENDPOINT_URL=$(printf '%s' "${AWS_ENDPOINT_URL:-}" | tr -d '\r')
AWS_ACCESS_KEY_ID=$(printf '%s' "${AWS_ACCESS_KEY_ID:-}" | tr -d '\r')
AWS_SECRET_ACCESS_KEY=$(printf '%s' "${AWS_SECRET_ACCESS_KEY:-}" | tr -d '\r')
AWS_DEFAULT_REGION=$(printf '%s' "${AWS_DEFAULT_REGION:-}" | tr -d '\r')

if [ -z "${BACKUP_S3_BUCKET}" ]; then
  if [ "${TRIPPY_BACKUP_REQUIRE_S3:-}" = "1" ]; then
    echo "backup.sh: BACKUP_S3_BUCKET is unset in ${ROOT}/.env — object-store upload is required" >&2
    exit 1
  fi
  echo "BACKUP_S3_BUCKET unset; dump stayed on disk only"
  exit 0
fi

: "${AWS_ENDPOINT_URL:?AWS_ENDPOINT_URL is required to upload}"
: "${AWS_ACCESS_KEY_ID:?AWS_ACCESS_KEY_ID is required to upload}"
: "${AWS_SECRET_ACCESS_KEY:?AWS_SECRET_ACCESS_KEY is required to upload}"

s3_uri="s3://${BACKUP_S3_BUCKET}/postgres/${name}"
docker run --rm \
  -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY \
  -e AWS_DEFAULT_REGION \
  -e AWS_ENDPOINT_URL \
  -v "${dump}:/data/dump:ro" \
  amazon/aws-cli \
  --endpoint-url "${AWS_ENDPOINT_URL}" \
  s3 cp /data/dump "${s3_uri}"
echo "uploaded ${s3_uri}"

report="${BACKUP_REPORT_PATH:-${BACKUP_DIR}/last.md}"
mkdir -p "$(dirname "${report}")"
{
  echo "Backup was written to \`${s3_uri}\`."
  echo
  echo "- local: \`${dump}\`"
  echo "- size: ${size} bytes"
} > "${report}"
