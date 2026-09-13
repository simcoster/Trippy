# Cloud (Nebius VM)

Phase-1 host: one always-on CPU VM, self-hosted Postgres, Streamlit over
Cloudflare Tunnel, ingest as one-shot containers. Telegram is later.
Start/stop of the VM is later.

LLM still goes to Token Factory over HTTPS. The VM does not need a GPU.

## Shape

```text
testers ──HTTPS──► Cloudflare Tunnel ──► Streamlit :8501
                                              │
GitHub Actions ──SSH──► docker compose run scrape
                                              ▼
                                    Postgres (compose network only)
                                              │
                         pg_dump ──► /var/lib/trippy/backups
                                 └──► Object Storage (optional)
```

Cost, 24/7, `me-west1` `cpu-d3` `4vcpu-16gb`: about **$80/month**
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

GitHub Actions [`.github/workflows/scrape.yml`](../.github/workflows/scrape.yml)
is a GitHub-hosted runner that **SSHs into the VM** and runs that
`compose run`. The scrape itself (INPA HTTP, LLM, Postgres writes) happens
on Nebius, not on GitHub. Daily availability at 01:00 UTC; anything else
is `workflow_dispatch`. Two scrapes cannot overlap (`concurrency: scrape`).

## Manual steps

### 1. VM

In [console.nebius.com](https://console.nebius.com), project you already have:

1. Region **`me-west1`** (Israel). Platform **`cpu-d3`**, preset **`4vcpu-16gb`**.
2. Ubuntu 24.04, 50 GiB boot **network SSD**, Docker later via bootstrap.
3. Optional second 50 GiB disk mounted at `/var/lib/trippy/backups`.
4. SSH key. Security group: **22** reachable from the internet (GitHub-hosted
   runners have no stable IP list worth allowlisting). Key-only, no passwords.
   Do not open 5432 or 8501. Use a **static** public IP — GitHub stores it as
   `TRIPPY_VM_HOST`.
5. Cloudflare Tunnel still carries Streamlit; the static IP is for SSH.

### 2. Object Storage (backups)

1. Create bucket `trippy-backups`.
2. Static access keys for that bucket.
3. Endpoint `https://storage.me-west1.nebius.cloud` (adjust if you picked another region).
4. Put `BACKUP_S3_BUCKET`, `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`,
   `AWS_SECRET_ACCESS_KEY` in the VM `.env`. If those are unset, dumps stay on disk only.

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
#            CLOUDFLARE_TUNNEL_TOKEN, optional AWS_* 
sudo sh ./scripts/cloud/bootstrap.sh
```

Bootstrap installs Docker if needed, daily cron for
[`scripts/cloud/backup.sh`](../scripts/cloud/backup.sh), builds the image,
starts `db` + `streamlit` + `cloudflared`, runs Alembic.

### 5. Restore the laptop database

On the laptop (with Compose Postgres up):

```bash
docker compose exec -T db pg_dump -U trippy -Fc trippy > trippy.dump
```

Copy `trippy.dump` to the VM, then:

```bash
cd /opt/trippy
docker compose -f docker-compose.prod.yml --env-file .env exec -T db \
  pg_restore -U trippy -d trippy --clean --if-exists < trippy.dump
```

Do not scrape `public` as a smoke test of the new box.

### 6. GitHub Actions → SSH

Generate a **dedicated** ed25519 key (not your laptop key). Put the public
half in `ubuntu`'s `authorized_keys` on the VM. Repo Settings → Secrets →
Actions:

| Secret | Value |
|--------|--------|
| `TRIPPY_VM_HOST` | static public IP |
| `TRIPPY_SSH_USER` | `ubuntu` |
| `TRIPPY_SSH_KEY` | private key PEM (the `-----BEGIN …` file) |

Optional repo variable `TRIPPY_ROOT` if the clone is not `/opt/trippy`.

Then Actions → **Scrape** → Run workflow. The job logs are the SSH session;
the container runs on the VM.

### 7. Deploy a new commit (later)

```bash
cd /opt/trippy
sudo git pull
sudo docker compose -f docker-compose.prod.yml --env-file .env up -d --build
sudo docker compose -f docker-compose.prod.yml --env-file .env \
  --profile scrape run --rm scrape migrate
```

### 8. Day-1 billing check

Compute, disks, and Token Factory are separate line items. Confirm the VM
is the `4vcpu-16gb` you meant, not a GPU preset.

## Restore drill

Once: take today's object-store dump (or a local file), `pg_restore` onto
this VM or a throwaway disk, `just streamlit` / the public URL, one real
query. That is the backup existing until you have done it.

## Local vs prod Compose

| | `docker-compose.yml` | `docker-compose.prod.yml` |
|--|----------------------|---------------------------|
| Who | laptop | Nebius VM |
| Bind-mount / `--reload` | yes | no |
| Streamlit | host `just streamlit` | container |
| Postgres port | `5432:5432` | unpublished |
| FastAPI / Telegram | `api` service | not started |
