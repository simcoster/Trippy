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
chgrp docker "${BACKUP_DIR}"
chmod 770 "${BACKUP_DIR}"

install -m 0755 "${ROOT}/scripts/cloud/backup.sh" /usr/local/sbin/trippy-backup
install -m 0755 "${ROOT}/scripts/cloud/restore.sh" /usr/local/sbin/trippy-restore
# GitHub Actions dumps once a day at 14:00 IDT (backup.yml). No VM cron.
rm -f /etc/cron.d/trippy-backup

cd "${ROOT}"
docker compose -f docker-compose.prod.yml --env-file .env build
docker compose -f docker-compose.prod.yml --env-file .env up -d db price-sandbox
docker compose -f docker-compose.prod.yml --env-file .env --profile scrape run --rm scrape migrate
docker compose -f docker-compose.prod.yml --env-file .env --profile load run --rm price-sandbox-loader
docker compose -f docker-compose.prod.yml --env-file .env up -d

echo "compose is up. GitHub Actions SSHs as gh-actions (see docs/cloud.md)."
echo "Dump public schema: just backup   (or /usr/local/sbin/trippy-backup)"
echo "Restore: just restore backups/trippy-….dump"
