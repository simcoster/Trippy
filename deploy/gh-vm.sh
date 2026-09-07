#!/bin/bash
# GitHub Actions helper. Expects NEBIUS_* secrets/vars in the environment.
set -euo pipefail

action="${1:?usage: gh-vm.sh start|stop|snapshot}"

install_nebius() {
  if command -v nebius >/dev/null 2>&1; then
    return
  fi
  curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
  export PATH="${HOME}/.nebius/bin:${HOME}/.local/bin:/usr/local/bin:${PATH}"
  hash -r
  command -v nebius
}

auth_nebius() {
  : "${NEBIUS_CREDENTIALS:?}"
  : "${SA_ID:?}"
  : "${PROJECT_ID:?}"
  mkdir -p "${HOME}/.nebius"
  printf '%s' "$NEBIUS_CREDENTIALS" > "${HOME}/.nebius/${SA_ID}-credentials.json"
  chmod 600 "${HOME}/.nebius/${SA_ID}-credentials.json"
  nebius profile create \
    --endpoint api.nebius.cloud \
    --service-account-file "${HOME}/.nebius/${SA_ID}-credentials.json" \
    --parent-id "$PROJECT_ID" \
    --profile github
}

write_ssh_key() {
  : "${SSH_PRIVATE_KEY:?}"
  mkdir -p "${HOME}/.ssh"
  printf '%s\n' "$SSH_PRIVATE_KEY" > "${HOME}/.ssh/id_ed25519"
  chmod 600 "${HOME}/.ssh/id_ed25519"
}

ssh_vm() {
  : "${VM_HOST:?}"
  ssh -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes \
    -i "${HOME}/.ssh/id_ed25519" \
    "ubuntu@${VM_HOST}" "$@"
}

wait_ssh() {
  local i
  for i in $(seq 1 60); do
    if ssh_vm "true"; then
      return 0
    fi
    sleep 5
  done
  echo "SSH to ${VM_HOST} did not become ready" >&2
  return 1
}

compose() {
  ssh_vm "cd /opt/trippy && docker compose -f docker-compose.yml -f docker-compose.prod.yml $*"
}

case "$action" in
  start)
    install_nebius
    auth_nebius
    write_ssh_key
    : "${INSTANCE_ID:?}"
    nebius compute instance start --id "$INSTANCE_ID"
    wait_ssh
    compose "up -d"
    ;;
  stop)
    install_nebius
    auth_nebius
    write_ssh_key
    : "${INSTANCE_ID:?}"
    ssh_vm "cd /opt/trippy && bash deploy/dump-postgres.sh"
    compose "stop"
    nebius compute instance stop --id "$INSTANCE_ID"
    ;;
  snapshot)
    install_nebius
    auth_nebius
    : "${DATA_DISK_ID:?}"
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    nebius compute disk-snapshot create \
      --name "trippy-pgdata-${stamp}" \
      --source-disk-id "$DATA_DISK_ID"
    ;;
  *)
    echo "unknown action: $action" >&2
    exit 2
    ;;
esac
