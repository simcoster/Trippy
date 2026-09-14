#!/bin/sh
# First-boot on the Nebius VM. Run as root from /opt/trippy after clone + .env.
set -eu

ROOT="${TRIPPY_ROOT:-/opt/trippy}"
BACKUP_DIR="${TRIPPY_BACKUP_DIR:-/var/lib/trippy/backups}"

if [ "$(id -u)" -ne 0 ]; then
  echo "run as root" >&2
  exit 1
fi
if [ ! -f "${ROOT}/.env" ]; then
  echo "missing ${ROOT}/.env — copy .env.example and fill secrets" >&2
  exit 1
fi
if [ ! -f "${ROOT}/docker-compose.prod.yml" ]; then
  echo "missing ${ROOT}/docker-compose.prod.yml" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl git

if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
fi
if id ubuntu >/dev/null 2>&1; then
  usermod -aG docker ubuntu
fi

mkdir -p "${BACKUP_DIR}"
chmod 700 "${BACKUP_DIR}"

install -m 0755 "${ROOT}/scripts/cloud/backup.sh" /usr/local/sbin/trippy-backup
cat >/etc/cron.d/trippy-backup <<EOF
# Daily 01:15 UTC dump + optional object-store upload.
15 1 * * * root /usr/local/sbin/trippy-backup >> /var/log/trippy-backup.log 2>&1
EOF

cd "${ROOT}"
docker compose -f docker-compose.prod.yml --env-file .env build
docker compose -f docker-compose.prod.yml --env-file .env up -d
docker compose -f docker-compose.prod.yml --env-file .env --profile scrape run --rm scrape migrate

echo "compose is up. Register a GitHub self-hosted runner labeled 'trippy' (see docs/cloud.md)."
echo "Restore a laptop dump with:"
echo "  docker compose -f docker-compose.prod.yml --env-file .env exec -T db pg_restore -U trippy -d trippy --clean --if-exists < dump.dump"
