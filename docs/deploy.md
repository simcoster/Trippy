# Host Trippy on a Nebius VM

Why these choices: [deploy_design.md](deploy_design.md). This file is the runbook; it does not create anything by itself.

Region: **eu-north1**. Size: **cpu-e2 / 2vcpu-8gb**. Image: **ubuntu24.04-driverless**.

Stop the VM only from the Nebius console, CLI, or the GitHub Action. `shutdown` / `halt` inside Linux looks like a crash: Compute reboots the box and **keeps charging**.

Never use a **local SSD** for Postgres. It is wiped when the VM stops.

## What you will have

- Boot disk (~40 GiB network SSD) — OS. Replaceable.
- Data disk (~40 GiB **standalone** network SSD, deletion protection) — Postgres. Survives stop, and you can attach it to a new VM.
- Static public IP — still yours after a night stop. A dynamic IP is released after one hour stopped.
- Streamlit on `127.0.0.1:8501`. You reach it with an SSH tunnel. It is not on the public internet.
- Three Compose services, **one image**: `db`, `api` (Streamlit/LangGraph), `jobs` (scrape/clear HTTP on port 8080, not published). Streamlit sets `JOBS_URL=http://jobs:8080`. Do not open 8080 on the security group.

## Console (do this)

### 1. Network

In **Virtual Networks**, note the default network and subnet IDs for `eu-north1`.

Create a **security group** on that network. Add:

- Ingress TCP **22** from your home/office `/32` only.
- Egress **any** to `0.0.0.0/0` (apt, Docker Hub, Token Factory, INPA, GitHub).

Do not open 5432, 8501, or 8080. The default group allows everything; assign **this** group to the VM so the default is not what applies.

### 2. Disks

**Storage → Disks → Create**

1. Boot disk, ~40 GiB, type **Network SSD**, source image family `ubuntu24.04-driverless`. Name e.g. `trippy-boot`.
2. Empty data disk, ~40 GiB, **Network SSD**. Name e.g. `trippy-pgdata`. Enable **forbid deletion**. This is the Postgres disk.

### 3. Virtual machine

**Compute → Create instance**

- Platform `cpu-e2`, preset `2vcpu-8gb`.
- Boot disk: the disk from step 2.
- Additional disk: `trippy-pgdata`, attach read-write, **device id** `trippy-pgdata` (cloud-init looks for `/dev/disk/by-id/virtio-trippy-pgdata`).
- Public IP: **static** (not dynamic).
- Subnet: the one you noted. Security group: the one you created.
- Cloud-init: paste [deploy/cloud-init.yaml](../deploy/cloud-init.yaml) after replacing `REPLACE_WITH_YOUR_SSH_PUBLIC_KEY`.

Create and wait until it is running. Copy the public IPv4.

### 4. First boot

```bash
ssh ubuntu@<static-ip>
```

```bash
git clone <this-repo-url> /opt/trippy
cd /opt/trippy
cp deploy/env.example .env
# edit .env: POSTGRES_PASSWORD, DATABASE_URL (same password), NEBIUS_API_KEY, GOOGLE_API_KEY
chmod 600 .env
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec jobs \
  /app/.venv/bin/alembic upgrade head
```

On your laptop:

```bash
ssh -L 8501:127.0.0.1:8501 ubuntu@<static-ip>
```

Open http://127.0.0.1:8501 — chat plus a **Jobs** sidebar for scrape/clear.

### 5. Object Storage (dumps)

**Storage → Object Storage → Create bucket** (regional, private). Optional: S3 access key for a service account, then set `TRIPPY_S3_BUCKET` and AWS keys in `.env` so `deploy/dump-postgres.sh` also uploads. Dumps always land on the data disk at `/var/lib/trippy/backups` even without a bucket.

### 6. GitHub start/stop

Create a **service account** in IAM, put it in a group that can start/stop this VM and snapshot disks (not tenant-wide admin if you can avoid it). Generate an authorized-key credentials file:

```bash
export SA_ID=$(nebius iam service-account get-by-name --name <name> --format jsonpath='{.metadata.id}')
nebius iam auth-public-key generate \
  --service-account-id "$SA_ID" \
  --output "./${SA_ID}-credentials.json"
```

Repo **secrets**:

| Secret | Value |
|--------|--------|
| `NEBIUS_CREDENTIALS` | entire credentials JSON |
| `NEBIUS_SA_ID` | service account id |
| `VM_SSH_PRIVATE_KEY` | private key that matches the cloud-init public key |

Repo **variables**:

| Variable | Value |
|----------|--------|
| `NEBIUS_PROJECT_ID` | project id |
| `NEBIUS_INSTANCE_ID` | VM id |
| `NEBIUS_DATA_DISK_ID` | data disk id (weekly snapshot) |
| `VM_SSH_HOST` | static public IP |

Until `NEBIUS_INSTANCE_ID` is set, [.github/workflows/nebius-vm-schedule.yml](../.github/workflows/nebius-vm-schedule.yml) does nothing. After that:

- **04:00 UTC** (~07:00 Israel summer) — start VM, `docker compose up -d`
- **20:00 UTC** (~23:00 Israel summer) — `deploy/dump-postgres.sh`, `compose stop`, Nebius stop
- **Sunday 03:00 UTC** — snapshot the data disk

You can also run the workflow manually (`start` / `stop` / `snapshot`).

Stop goes through Nebius. The Action SSHs dump + compose stop first so Postgres checkpoints, then calls `nebius compute instance stop`.

## CLI cheat-sheet

IDs from `nebius vpc subnet list`, `nebius compute disk list`, `nebius compute instance list`.

```bash
# standalone data disk
nebius compute disk create \
  --name trippy-pgdata \
  --type NETWORK_SSD \
  --size-gibibytes 40 \
  --forbid-deletion true \
  --block-size-bytes 4096

# boot disk from public image
nebius compute disk create \
  --name trippy-boot \
  --type NETWORK_SSD \
  --size-gibibytes 40 \
  --source-image-family-image-family ubuntu24.04-driverless \
  --block-size-bytes 4096

# static IP (keeps the address while the VM is stopped)
nebius compute instance create \
  --name trippy \
  --resources-platform cpu-e2 \
  --resources-preset 2vcpu-8gb \
  --boot-disk-existing-disk-id <boot_disk_id> \
  --boot-disk-attach-mode READ_WRITE \
  --secondary-disks '[{"existing_disk":{"id":"<data_disk_id>"},"attach_mode":"READ_WRITE","device_id":"trippy-pgdata"}]' \
  --network-interfaces '[{"name":"eth0","ip_address":{},"public_ip_address":{"static":true},"subnet_id":"<subnet_id>"}]' \
  --cloud-init-user-data "$(cat deploy/cloud-init.yaml)"
```

Start / stop:

```bash
nebius compute instance start --id <vm_id>
nebius compute instance stop --id <vm_id>
```

Snapshot (works while stopped):

```bash
nebius compute disk-snapshot create \
  --name trippy-pgdata-$(date -u +%Y%m%d) \
  --source-disk-id <data_disk_id>
```

Security groups: [docs](https://docs.nebius.com/vpc/security-groups/manage). Assign the group on the NIC so 22 is not world-open.

## Cost (list prices, no VAT)

`2vcpu-8gb` is about **$0.05/h** while `RUNNING`. 80 GiB network SSD is about **$0.19/day** even when stopped. Sixteen hours a day is about **$1/day** infra. Token Factory is extra.

## Restore

1. Keep the data disk (or create a new disk from a snapshot).
2. Attach it to a VM with device id `trippy-pgdata`.
3. Cloud-init / `trippy-mount-data.sh` mounts it; compose bind-mounts `/var/lib/trippy/pgdata`.
4. Or: `gunzip -c /var/lib/trippy/backups/trippy-….sql.gz | docker compose … exec -T db psql -U trippy -d trippy`

Do one restore drill before you trust night stops.
