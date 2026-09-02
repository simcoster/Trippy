# Scaling, cloud, and production-ready

Ops and reliability only — not product work (recommender node, group constraints, pin mixing, etc.). Complements `PLAN.md` §7 (which is still a sketch). Catalog is Israel INPA campsites; live channel is Telegram (`main.py`); Streamlit is a local harness.

Written 2026-09-02 from the current repo: one Compose stack, in-memory chats, sync webhook, `just` scrapes, Nebius Qwen (30B / 235B / embed), Postgres + pgvector.

---

## What we have today

One Docker Compose stack: Postgres + a FastAPI container that also starts **ngrok**, runs **uvicorn `--reload`**, and bind-mounts the repo.

Product path:

```text
Telegram → POST /webhook → graph.invoke (sync) → sendMessage
```

The webhook holds the HTTP request until the whole LangGraph turn finishes (light 235B, extractor 30B, planner SQL, recommender 235B). Conversations live in a process dict (`conversations: dict[int, list]` in `main.py`). Restart = amnesia; two replicas = split-brain.

Scrapers are `just` CLIs a human runs. Availability is still capped (`limit_campsites: 2` plus a leftover `and id = 2` filter). Every search opens a fresh `psycopg.connect()`. No CI, no real health check, no job runner, no backups.

Reviews ingest (`populate_reviews_and_claims`): for each campsite, for each review: 30B visit gate → 235B split (one review per call, locked in `docs/claims.md`) → one embed batch for the site. The site is one transaction; an exception rolls all of it back.

Availability scrape: for each site, for each of 14 nights: HTTP to INPA, 0.5s pause, sometimes 30B name-match and 235B amenity enrich. Sequential.

---

## Scale reality

Production risk is not “millions of QPS.” It is Telegram latency, LLM flakiness, scrape freshness, and the stack being a laptop process.

Honest chat load: tens of concurrent chats at a Thursday/Friday peak, not thousands. A live turn is still a handful of LLM calls and a small SQL filter on planner `fits`. Catalog size does not turn the API into a scale problem until concurrent chats pile up — a different axis.

| Growth | What hurts | What does not |
|--------|------------|----------------|
| Thousands of reviews | 235B call count, $, crash-resume, Nebius concurrency | Postgres rows, HNSW, Telegram |
| Hundreds of campsites | INPA HTTP × nights, scrape wall-clock vs freshness SLO | Availability table size, planner SQL |
| Concurrent chats | Webhook timeout, in-memory sessions, Nebius stampede | Postgres QPS |

Users forgive a slightly clumsy rec. They will not forgive “it said Friday was free and it wasn’t.” Availability freshness is product reliability.

---

## Target shape: three process types

Do **not** put API, scrapes, and ngrok in one container. Leave Streamlit off the prod image.

```text
Telegram ──webhook──► API (FastAPI)
                         │  ack 200 immediately
                         │  enqueue turn
                         ▼
                      Worker(s) ── LangGraph ── Nebius
                         │
                         ▼
                   Postgres+pgvector
                         ▲
                         │
           Scheduler ── scrape jobs (availability / prices / reviews)
```

| Process | Role | Scale independently |
|---------|------|---------------------|
| **API** | Telegram webhook, health, maybe later a public HTTP API | On webhook QPS (tiny) |
| **Worker** | Run the graph, send the reply | On concurrent chats × LLM latency |
| **Jobs** | Ingest: sites, prices, availability, reviews/claims | On scrape cadence, not user traffic |

Host: a small always-on API + workers (Fly, Render, Cloud Run + jobs, or a Nebius VM). Managed Postgres (Neon, RDS, Cloud SQL, or Nebius Postgres) with daily backups and PITR.

A **work queue** is what creates ingest/chat scale. Kubernetes is one way to *run* the consumers of that queue. Revisit K8s only when operating several always-on process types is harder than a VM — an ops threshold, not a row-count threshold. See [Kubernetes](#kubernetes) below.

---

## 1. Telegram contract (the reliability bug)

Telegram retries if the webhook does not 200 quickly. A 30B+235B turn plus embeddings can blow past that. Today we hold the request until `graph.invoke` finishes, then send the reply, then return 200.

Production contract:

1. Verify the request (`X-Telegram-Bot-Api-Secret-Token`; the webhook URL is not a secret).
2. Ignore non-text / no `message`.
3. Dedup on `update_id` (Telegram retries).
4. Persist the inbound message, enqueue work, return `{"ok": true}` in tens of milliseconds.
5. Worker runs the graph, then `sendMessage`. Show typing (`sendChatAction`) so the chat does not look dead.
6. Never put `str(e)` in the user reply (we do that now). Log the exception, send a generic Hebrew sorry.

Use a public HTTPS URL and `setWebhook`. Delete ngrok from the Dockerfile and from `scripts/startup.sh`. `--reload` in prod is wrong.

`graph.invoke` is sync inside an async route, so one slow turn blocks the event loop. Workers should `ainvoke` (or run sync graph in a thread pool / separate process). Cap concurrent graph runs (semaphore) so 20 Friday-afternoon chats do not stampede Nebius and Postgres.

---

## 2. State that survives restart and replicas

The in-memory dict means: restart = amnesia, two API replicas = split-brain, no audit trail.

Minimum:

- Table `conversations (chat_id PK, messages jsonb, updated_at)` — `PLAN.md` §4 sketches this.
- Load by `chat_id` at the start of a turn, write-through after.
- Bound history (last N turns or a compacted prefs blob). Unbounded LangChain messages make 235B cost and latency grow without bound.
- Redis later if we need a queue **and** a hot cache. Postgres JSONB is enough at this traffic.

Until session store is shared, we cannot run more than one API/worker replica safely. Horizontal API scale without a queue and a session store just duplicates in-memory dicts and double-replies.

---

## 3. Data plane

Local `trippy/trippy` on Docker with a volume is fine for a laptop. Production needs:

- Managed Postgres 16 + pgvector (HNSW already on claims / amenities / notices).
- Secrets from the host’s secret store, not `.env` on disk and not `config.json` with a localhost URL.
- `alembic upgrade head` as a **release step**, not a `just` we remember to run.
- Automated backups + a restore drill once.
- Connection **pool** in the API/worker (`psycopg_pool` or SQLAlchemy). Search currently connects per query. That will not die at our QPS, but it adds latency and fails messily under a scrape + chat collision.
- Staging DB that is not production. Nuke-and-pave (`docker compose down -v`) must be impossible against prod.
- Kill the `url.replace("@db:", "@localhost:")` hack in scrapers; use one `DATABASE_URL` per environment.
- `/healthz` that checks Postgres (and maybe “can reach Nebius”) vs `/readyz` for the orchestrator. `/` today always says ok.

A read replica is unnecessary until the catalog is 10× or we add heavy analytics. Availability for ~50 sites × 14 nights × a handful of types is tiny. Hundreds of sites is still tens of thousands of rows — planner stage 1 is a date-window `GROUP BY`.

Don’t cache availability in the API until scrape cost or DB CPU shows up. If we cache, TTL must be shorter than scrape cadence or we recommend ghosts. CDN / edge: irrelevant.

---

## 4. Jobs: freshness is reliability

Treat ingest as scheduled, observable jobs — not `just scrape-availability` on a laptop. Do not run these in the API process. A 14-night × N-site crawl will steal CPU, sockets, and Nebius quota from live chats.

| Job | Cadence (start here) | Why |
|-----|----------------------|-----|
| **Availability** | Hourly Thu–Sat IL daytime; every 4–6h otherwise | Booking truth decays fast near weekends |
| **Prices / info-site** | Daily or weekly | Tariffs change rarely |
| **Site discovery + booking IDs** | Weekly | Catalog is almost static |
| **Reviews `newest`** | Weekly | Places cap is 5; LLM-heavy |
| **Reviews `most_relevant` seed** | Once per new site | Not a cron |

How: Cloud Run Jobs + Scheduler, GitHub Actions cron, or a tiny worker with cron. Same Docker image, different command. Lock per job so two overlapping availability scrapes cannot delete-then-insert the same night.

Productionize the scrapers themselves:

- Drop `limit_campsites: 2` and the `id = 2` debug filter; drive limits from env.
- Polite concurrency + backoff against INPA / parks.org.il / Google. Getting banned is an outage.
- Per-site try/except: one HTML change must not abort the other 40 sites.
- Idempotent upserts (already have `availability_unique_slot`).
- Job metrics: sites attempted/ok/fail, rows written, `max(scraped_at)` age, LLM $ spent.
- Alert if availability `scraped_at` is older than 2× the cadence, or if a job exits non-zero.

Reviews/claims are a **queue of reviews**, not a long synchronous CLI. A crash mid-site should resume; reviews are already keyed by `review_uid`.

---

## 5. LLM is the hardest dependency

Every real turn is 2–4 Nebius calls. Ingest is many more. Reliability here beats replica count.

- Timeouts, retries with jitter, and a circuit breaker per model (30B / 235B / embed).
- Hard concurrency limit toward Token Factory (don’t open 50 235B calls because 50 webhooks arrived).
- Fallback: if 235B is down, extractor/recommender on 30B with a user-visible “slower/simpler” path is better than silence.
- Cache embeddings of identical amenity queries (planner stage 2). Catalog embeddings are already in Postgres — don’t re-embed on read.
- Track tokens, latency, error rate, $ per turn. A busy Friday is a cost incident, not a scale incident.
- Fail closed to the user: “try again in a minute,” not a stack trace.

We cannot HA Nebius. We can fail fast and degrade. **If the bottleneck is a third-party QPS cap, more pods only help up to that cap.** For us that cap is Nebius (reviews, chat) and INPA (availability).

---

## 6. Ship / release bar

- **CI:** `pytest -m "not llm"` on every PR. Alembic `upgrade` against a throwaway Postgres. No `NEBIUS_API_KEY` in CI.
- **CD:** build an image **without** ngrok, without `--reload`, without the repo bind-mount, without pytest as a runtime dep (`pytest` is in `[project] dependencies` today). Run migrations, then roll the API.
- **Secrets:** `TELEGRAM_TOKEN`, `NEBIUS_API_KEY`, `GOOGLE_API_KEY`, `DATABASE_URL` only from the platform. Rotate the bot token once ngrok URLs have been public.
- **Staging** with a test bot and a copy of prod data (PII-aware: review authors, Telegram chat ids).
- **Observability:** JSON logs with `chat_id` / `update_id` / `job_name` / latency. Metrics: webhook accept vs worker success, graph node latency, scrape freshness, Nebius errors. Alert on job failure, webhook 5xx, error-rate spike, stale availability. Grafana + Prometheus or whatever the host ships; don’t build a metrics platform first.
- **Security:** secret webhook token; do not log full Telegram updates in prod (we log the whole `update` today); strip PII from traces; keep Streamlit off the internet.

---

## Suggested sequence (reliability first)

1. **Prod image + managed Postgres + secrets + `alembic` on deploy.** Kill ngrok/`--reload`. Health checks. This is “it stays up when the laptop sleeps.”
2. **Webhook ack + queue + no leaked exceptions.** This is “Telegram doesn’t retry-storm us.”
3. **Persist conversations.** This is “restart and two workers don’t break chats.”
4. **Scheduled availability job with freshness alerts.** This is “recommendations aren’t lies.”
5. **LLM timeouts/retries/concurrency/cost metrics.** This is “Friday peak doesn’t melt the budget or hang every chat.”
6. **CI on `not llm` + a staging bot.**
7. **Then** weekly reviews job, prices job, optional Redis if the queue needs it.

Skip until they are earned: Kubernetes, read replicas, multi-region, embedding microservices, “event-driven mesh.”

### What “done” looks like (current catalog / traffic)

- A Telegram message gets a 200 in &lt;1s even if Nebius is slow; the user still gets a reply or a clean failure.
- Restarting the API does not wipe chats; two workers do not fork history.
- Availability for the next 14 days is no staler than the job SLO, and we get paged if it is.
- We can restore yesterday’s Postgres.
- We can ship a commit without SSHing and running `just`.
- We know $ and p95 latency per turn.

That is cloudifying this repo: **split API / worker / jobs, durable state, scheduled ingest, LLM as a flaky dependency.** Not a bigger orchestrator.

---

## Thousands of reviews

Places still only gives 5+5. “Thousands” means a scrape-vendor backfill (all recent reviews) or many sites × history.

Locked path (`docs/claims.md`): gate 30B + split 235B, one review per chat call, ~5–15s of model time per kept review.

- **1,000 reviews sequential** → hours
- **10,000** → a day
- **100,000** → do not run 235B on every row without a budget cap and a skip policy

Cost scales with **kept** reviews (gate skips ads). Embeddings are cheap next to 235B. Parallelism is capped by Token Factory rate limits, not by how many pods we start. Fifty ingest pods against a 10-concurrent 235B quota is fifty pods waiting.

What to change in the app (this is the real work):

1. **Unit of work = one `review_uid` (or `review_id`), not one campsite.** Fetch/store text first (no LLM). Enqueue `pending_gate` → `pending_split` → `pending_embed`.
2. **Commit per review.** Today a mid-site crash undoes every split we already paid for.
3. **Idempotency we almost have:** skip if `skip_reason` is set; skip split if claims already exist for that uid. Make “already done” the default path so a job is safely retryable.
4. **Workers with a global semaphore / queue QoS** toward Nebius (separate pools for 30B gate vs 235B split). Scale on queue depth, not CPU — these workers are idle waiting on HTTP.
5. **Embed in small batches** (20–100 texts), not “all claims for the site after every split.” A 2,000-review site should not hold 10k vectors in RAM until the end.
6. **Backfill vs incremental:** backfill is a finite Job with high parallelism until the queue drains. Weekly `newest` is a trickle (hundreds of sites × 5). Do not use the backfill replica count on the cron.

pgvector: thousands of claims is nothing. Hundreds of thousands of 1536-d HNSW vectors is still normal. Care at **millions** (index RAM, `ef_search`, vacuum). Partitioning `claims` by `campsite_id` is optional later.

K8s mapping if we want it: `CronJob` (fetch → enqueue) + `Deployment` ingest-workers + HPA on queue length. A Cloud Run Job with `--tasks=20` or one VM running 20 worker processes against Redis/SQS does the same thing. The cluster is optional; the queue is not.

---

## Hundreds of campsites

200 sites × 14 nights ≈ **2,800 HTTP calls**. At ~1–2s + 0.5s pause, one process is **1–2 hours**. Fine for a nightly job; **not** fine if we want hourly freshness on a Thursday.

What to change:

1. **Unit of work = `(site_id, start_date)`** (or `site_id` for a 14-night bundle if INPA bans per-night parallelism).
2. **Cap in-flight requests to INPA** (start at 2–4). Throughput is `min(workers, INPA budget)`, not `min(workers, cluster size)`. Getting rate-limited is a worse outage than a 90-minute scrape.
3. **Isolate LLM from HTML.** Name-match and amenity enrich are rare after the first pass (only new room types). Don’t let 235B sit on the scrape hot path; enqueue “new type seen.”
4. **Per-site / per-night errors:** we already `continue` on HTTP failure; keep that, and mark the slot stale instead of leaving yesterday’s rooms as if they were live.
5. **Planner/DB:** 200 × 14 × a few types is tens of thousands of `availability` rows. Stage 1 is still a date-window `GROUP BY`. Hundreds of sites do not require a read replica.

If we later have **thousands** of campsites (other countries, SimpleBooking, …): more scrape shards, same politeness per origin, maybe region-sharded jobs. Still not “the API needs 50 replicas.”

---

## Kubernetes

K8s does not create the scale we need. A work queue does. Copying the current CLI into a CronJob yields the same sequential 10-hour job, now with YAML.

### When it earns its keep

When we already have **several always-on process types with different scaling knobs**:

```text
Deployment  api              (tiny; webhook ack only)
Deployment  agent-worker     (HPA: chat queue; Nebius cap)
Deployment  ingest-worker    (HPA: review queue; separate Nebius cap)
CronJob     scrape-availability
CronJob     scrape-prices / discover
CronJob     reviews-fetch    (enqueue only)
Job         reviews-backfill (completions, parallelism)
Postgres    managed — not Postgres-in-the-cluster unless we enjoy it
Queue       Redis / SQS / Cloud Tasks / NATS
```

Then we get, for real:

- **Isolation:** a scrape burst cannot starve webhook CPU (resource requests + separate deployments). Today they would share one container.
- **Autoscaling on queue depth** for ingest and chat workers (KEDA / HPA + custom metric). CPU-based HPA is the wrong signal for LLM workers.
- **CronJobs + Jobs** with backoff, deadlines, `concurrencyPolicy: Forbid` so two availability crawls don’t overlap.
- **Rollouts:** migrate, then roll API/workers; ingest can keep draining the queue.
- **Limits:** Nebius budget as a cluster-level throttle is still *our* code (semaphore / token bucket). K8s will not infer Token Factory quotas.

What K8s does **not** do:

- Raise Nebius QPM
- Make INPA like being scraped harder
- Replace per-review commits or `review_uid` idempotency
- Fix Telegram webhook timeouts (that’s still ack-then-queue)
- Make one giant `populate_reviews_and_claims()` loop safe just because it runs in a Pod

### When not to start there

For hundreds of sites and a one-time 10k-review backfill, this is enough:

- Managed Postgres
- One small API service
- A queue
- N worker processes (Cloud Run Jobs, Fly machines, or a single VM + systemd/docker)
- A scheduler

Add Kubernetes when **operating those N process types** (secrets, rollouts, cron, HPA, “scrape must not kill chat”) is harder than a VM.

### If we expect that growth anyway

1. **Queue + per-review / per-(site,night) jobs + idempotent retries** — required at thousands of reviews even on one machine.
2. **Separate chat workers from ingest workers** with two Nebius budgets — so a backfill cannot lock 235B away from Telegram.
3. **Scrape parallelism with a hard INPA concurrency limit** — required at hundreds of sites if freshness SLO &lt; crawl time.
4. **K8s (or Cloud Run Jobs + scheduler, which is the same shape)** once (1)–(3) exist and we don’t want to babysit processes.

Hundreds of campsites: think **polite fan-out of HTTP**, not a cluster. Thousands of reviews: think **review-shaped jobs and 235B quota**, not HNSW or K8s. Use Kubernetes when those workers should be first-class services — not as the thing that makes the current `for site: for review:` loop production-scale.
