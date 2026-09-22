# Cloud (Nebius VM)

Phase-1 host: one always-on CPU VM, self-hosted Postgres, Streamlit over
Cloudflare Tunnel, ingest as one-shot containers.
Start/stop of the VM is later.

LLM still goes to Token Factory over HTTPS. The VM does not need a GPU.

## Shape

```text
testers ──HTTPS──► Cloudflare Tunnel ──► Streamlit :8501
                                              │ quotes
                                    price-sandbox (quote network)
                                              ▲ POST /load, then exit
                                    price-sandbox-loader (one-shot)
                                              │
GitHub Actions ──SSH──► docker compose run scrape
                                              ▼
                                    Postgres (compose network only)
                                              │
     14:00 IDT  pg_dump -n public ──► ~/.trippy-backups
                                 └──► Object Storage (trippy-backups)
```

Cost, 24/7, `eu-north1` `cpu-d3` `4vcpu-16gb`: about **$80/month**
(compute ~$72 + two 50 GiB network SSD ~$7 + object storage pennies).
Token Factory is a separate bill.

`restart: unless-stopped` on long-running services is the usual Compose
choice: come back after a crash or reboot; stay down after an explicit
`docker compose stop`.

## Scrapes = one image, different command

There is no always-on scraper. `docker-compose.prod.yml` builds `trippy:prod`
once. Streamlit is that image with a Streamlit command. A scrape is the
same image, profile `scrape`, entrypoint [`scripts/cloud/job.sh`](../scripts/cloud/job.sh):

```bash
docker compose -f docker-compose.prod.yml --env-file .env \
  --profile scrape run --rm scrape availability
```

GitHub Actions [`.github/workflows/scrape-availability.yml`](../.github/workflows/scrape-availability.yml),
[`.github/workflows/scrape-reviews.yml`](../.github/workflows/scrape-reviews.yml),
and [`.github/workflows/scrape-prices.yml`](../.github/workflows/scrape-prices.yml)
are GitHub-hosted runners that **SSH into the VM** as `gh-actions` and run
that `compose run`. The SSH steps live in
[`.github/actions/scrape-job`](../.github/actions/scrape-job/action.yml)
(a composite action, so it does not appear in the Actions list).
After the scrape the same SSH session **always** runs
`price-sandbox-loader`. If that service is missing from compose, the
job fails (`no such service`). Do not skip the reload. The scrape itself (INPA HTTP, LLM, Postgres writes)
happens on Nebius, not on GitHub. Daily availability at **08:00 IDT**
(`cron: 0 5 * * *`; 07:00 in winter IST); daily reviews at **09:00 IDT**
(`cron: 0 6 * * *`; 08:00 in winter IST). One dump per day at **14:00 IDT**
([`backup.yml`](../.github/workflows/backup.yml), `cron: 0 11 * * *`;
13:00 in winter IST) — not before a scrape. On-demand: Actions →
**Backup Postgres**. Prices is `workflow_dispatch` only (rate cards
change rarely). Other ingest (claims / info / sites / place-ids) is
`just prod-scrape <job>` on the VM. Two scrapes cannot overlap
(`concurrency: scrape` on each caller; the afternoon dump uses the
same group).

The report is the run’s **Summary** tab (Actions → **Scrape availability**,
**Scrape reviews**, or **Scrape prices** → that run), not a file and not
Streamlit. Availability lists vacancy changes; reviews lists new Google
rows then claims written (visit gate / split / embed); prices lists
stored vs failed compiles, gold/AST lines, and dump names. Function
dumps stay on the VM under `~/.trippy-scrape/<timestamp>/`. The
log still has the per-site scroll. GitHub emails you if the job fails.

## Manual steps

### 1. VM

In [console.nebius.com](https://console.nebius.com), project you already have:

1. Region **`eu-north1`** (Finland). Platform **`cpu-d3`**, preset **`4vcpu-16gb`**.
2. Ubuntu 24.04, 50 GiB boot **network SSD**, Docker later via bootstrap.
3. Optional second 50 GiB disk mounted at `/var/lib/trippy/backups`
   (only if you dump as root; Actions writes `~/.trippy-backups`).
4. SSH key. Security group: **22** reachable from the internet (GitHub-hosted
   runners have no stable IP list worth allowlisting). Key-only, no passwords.
   Do not open 5432 or 8501. Use a **static** public IP — GitHub stores it as
   `TRIPPY_VM_HOST`.
5. Cloudflare Tunnel still carries Streamlit; the static IP is for SSH.

### 2. Object Storage (backups)

1. Create bucket `trippy-backups` (Standard class).
2. Static access keys for that bucket.
3. Endpoint `https://storage.eu-north1.nebius.cloud` (same region as
   the VM). `AWS_DEFAULT_REGION=eu-north1`.
4. Lifecycle: expire prefix `postgres/` after **30 days**.
5. Put `BACKUP_S3_BUCKET`, `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`,
   `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` **uncommented** in the
   VM `.env`. `backup.yml` requires an upload; a dump that stays on
   disk only fails the job. Laptop `just backup` still skips S3 if
   those are unset. The Actions Summary is `Backup was written to
   s3://…` after a successful upload. `AWS_SECRET_ACCESS_KEY` is the
   one-time secret string, not the `accesskey-e00…` resource id.

`pg_dump -n public -Fc` only. `experiments` and `extensions` stay out.
On-disk `trippy` is tens of MB (indexes + a copy in `experiments`); the
object is heap+TOAST for `public`. A 2026-09-17 laptop dump was **4.6 MB**.
Standard storage is **$0.0147/GiB-month** (~$0.002/month for 30 daily
dumps at that size).
Egress **$0.015/GiB** applies when downloading off Nebius. Upload from
the VM to the bucket is not that line item. No per-object fee on the
price list.

Laptop: `just backup` / `just restore backups/trippy-….dump` /
`just restore-latest` (newest `postgres/trippy-*.dump` in
`BACKUP_S3_BUCKET`, into local Postgres). docker
compose cp; do not redirect `pg_dump` in PowerShell). Destructive
`just scrape-info` / `clear-*` dump first unless
`TRIPPY_SCHEMA=experiments`. VM: one GitHub Actions dump at 14:00 IDT
([`backup.yml`](../.github/workflows/backup.yml)), not cron and not
tied to a scrape. [`scripts/cloud/backup.sh`](../scripts/cloud/backup.sh)
writes under `~/.trippy-backups` (7 days; `gh-actions` cannot write
`/var/lib/trippy/backups`) and `s3://…/postgres/`.
Restore: `just restore backups/…`, `just restore s3://trippy-backups/postgres/…`,
or `just restore-latest` for the newest object in that prefix.

### 3. Cloudflare Tunnel

Easy path, named tunnel (URL stays stable across reboots):

1. Free account at [dash.cloudflare.com](https://dash.cloudflare.com).
2. Zero Trust → Networks → Tunnels → Create a tunnel → **Cloudflared**.
3. Name it `trippy`. Copy the **token** into `CLOUDFLARE_TUNNEL_TOKEN`.
4. Public Hostname: a subdomain you own on Cloudflare, or
   `<tunnel-uuid>.cfargotunnel.com`. Type HTTP, URL **`streamlit:8501`**
   (Compose service name, not localhost).
5. Leave Additional application settings empty for now. Cloudflare Access
   (Google login) can wrap this later.

No public 80/443 on the VM. `cloudflared` only makes outbound connections.

### 4. Code and secrets on the VM

```bash
sudo mkdir -p /opt/trippy
sudo git clone git@github.com:<org>/Trippy.git /opt/trippy
cd /opt/trippy
sudo cp .env.example .env
sudo chmod 600 .env
# edit .env: POSTGRES_PASSWORD, NEBIUS_API_KEY, GOOGLE_API_KEY,
#            CLOUDFLARE_TUNNEL_TOKEN, LANGSMITH_API_KEY,
#            BACKUP_S3_BUCKET + AWS_*
sudo sh ./scripts/cloud/bootstrap.sh
```

Bootstrap installs Docker if needed, `trippy-backup` / `trippy-restore`
(no cron — Actions dumps at 14:00 IDT via `backup.yml`), builds the
image, starts `db` + `streamlit` + `cloudflared` + `price-sandbox`,
runs Alembic, then a one-shot loader pushes `site_price_functions`
into the sandbox and exits. Backup dir is `770 root:docker` so
`gh-actions` can write it.

### 5. Restore the laptop database onto the VM

On the laptop (Compose Postgres up):

```text
just backup
```

Copy `backups/trippy-*.dump` to the VM, then:

```bash
cd /opt/trippy
just restore backups/trippy-YYYYMMDDTHHMMSSZ.dump
```

That replaces `public` only (`pg_restore --clean --if-exists -n public`).
`extensions` stays. `experiments` is dropped first: its column defaults
reference `public` sequences, so `--clean` cannot drop those sequences
while the schema is present. Recreate it with `just setup-experiments copy`.
A new cluster needs `db/init` /
Alembic first so `vector` / `pg_trgm` exist.

Do not scrape `public` as a smoke test of the new box.
After the restore, reload compiled quote functions:

```bash
docker compose -f docker-compose.prod.yml --env-file .env \
  --profile load run --rm price-sandbox-loader
```

### 6. GitHub Actions → SSH (`gh-actions`)

Do **not** put the Actions key on your personal account. A Linux user named
`gh-actions` holds a dedicated key so `auth.log` names a job, not you.
Anyone in the `docker` group can take root; this is key isolation, not a
sandbox. Do not give `gh-actions` sudo.

On the laptop (do **not** overwrite your existing id):

```bash
ssh-keygen -t ed25519 -C "github-actions-trippy" -f ./trippy-gha -N ""
```

SSH in with **your** key, then on the VM:

```bash
sudo adduser --disabled-password --gecos "GitHub Actions" gh-actions
sudo usermod -aG docker gh-actions
sudo mkdir -p /home/gh-actions/.ssh
echo '<paste contents of trippy-gha.pub>' | sudo tee /home/gh-actions/.ssh/authorized_keys
sudo chown -R gh-actions:gh-actions /home/gh-actions/.ssh
sudo chmod 700 /home/gh-actions/.ssh
sudo chmod 600 /home/gh-actions/.ssh/authorized_keys

# compose --env-file must be able to read this (600 root/you would fail)
sudo chmod 640 /opt/trippy/.env
sudo chgrp docker /opt/trippy/.env
```

Repo Settings → Secrets and variables → Actions:

| Secret | Value |
|--------|--------|
| `TRIPPY_VM_HOST` | static public IP |
| `TRIPPY_SSH_USER` | `gh-actions` |
| `TRIPPY_SSH_KEY` | entire `trippy-gha` private key, including the `BEGIN` / `END` lines |

Optional repo variable `TRIPPY_ROOT` if the clone is not `/opt/trippy`.

Delete `trippy-gha` from the laptop once the secret is saved. First check:
`ssh -i trippy-gha gh-actions@<ip>`, then Actions → **Scrape availability** →
Run workflow, extra args `--site 2`. The Summary tab is the vacancy
change report; the log is the SSH session. The container runs on the VM.

### 7. Deploy a new commit (later)

```bash
cd /opt/trippy
sudo git pull
sudo docker compose -f docker-compose.prod.yml --env-file .env up -d --build
sudo docker compose -f docker-compose.prod.yml --env-file .env \
  --profile scrape run --rm scrape migrate
sudo docker compose -f docker-compose.prod.yml --env-file .env \
  --profile load run --rm price-sandbox-loader
```

### 8. Day-1 billing check

Compute, disks, and Token Factory are separate line items. Confirm the VM
is the `4vcpu-16gb` you meant, not a GPU preset.

## Restore drill

Once: take today's object-store dump (`just restore s3://trippy-backups/postgres/…`
or a local file) onto a throwaway Compose project, `just streamlit`, one
real query. Do not `--clean` onto live `public` as the first test. That
is the backup existing until you have done it.

## Local vs prod Compose

| | `docker-compose.yml` | `docker-compose.prod.yml` |
|--|----------------------|---------------------------|
| Who | laptop | Nebius VM |
| Bind-mount / `--reload` | yes | no |
| Streamlit | host `just streamlit` | container |
| Price sandbox | `127.0.0.1:8503` | internal `quote` network |
| Load functions | `just load-price-sandbox` | `just prod-load-sandbox` |
| Postgres port | `5432:5432` | unpublished |
