# Trippy — Product & Engineering Plan

Campsite recommendation agent for Israel (parks.org.il + Google reviews), with RAG over claims/site data, availability/price search, and a Telegram-facing agent.

---

## Current baseline

Already in repo:

- LangGraph agent (`graph.py`) with claims RAG + numeric campsite filters
- Postgres + pgvector schema: `claims`, `campsites`
- FastAPI entry (`main.py`), Docker, crawler stub for parks.org.il listing

This plan extends that into a full ingestion → retrieval → agent → eval → production stack.

---

## 1. Google reviews → claim RAG

### Goal
Ingest **5 most recent** + **5 most relevant** Google reviews per campsite, split into atomic claims, embed, store in `claims`.

### Pipeline (cloud)

```
Campsite list → Google Places / Reviews API
  → raw reviews store
  → claim splitter
  → normalize (HE/EN), polarity, confidence
  → embed (text-embedding-3-small)
  → upsert claims (claim_uid)
```

### Claim splitting
Treat a review as a bag of claims, not one blob.

| Rule | Example |
|------|---------|
| Sentence / period | `"האתר יפה. אבל רועש"` → 2 claims |
| Newline | Multi-line reviews |
| Contrast discourse | `but` / `however` / `אבל` / `עם זאת` / `אך` |
| Soft separators | `;` , ` - ` when they mark independent judgments |

**Do not** over-split on every comma. Prefer: split on strong dividers, then LLM (or rules + LLM fallback) for borderline cases.

Example:

> "site is nice. but loud" →  
> 1. `site is nice`  
> 2. `loud` (negated amenity / noise)

Store `evidence_span` (original substring), `polarity`, `confidence`, `source=google`, author/date when available.

### Open questions
- Google Places API vs scraping (prefer official API; scraping is brittle/ToS risk)
- Dedup: same claim from many reviews → keep multiplicity or collapse with frequency weight?
- Refresh cadence: daily / weekly per campsite?

### Deliverables
- Ingestion job (batch + incremental)
- Extended `claims` usage (already mostly fits)
- Monitoring: reviews fetched, claims/review, embed failures

---

## 2. Vacancies & prices (next ~2 weeks)

### Goal
For each campsite, pull availability and price for a rolling window (start with **14 days**), bucketed by stay type and party size.

### Stay-type buckets
| Bucket | Notes |
|--------|--------|
| `tent` | Standard camping / tent pitch |
| `glamping` | Elevated tent / cabin-tent |
| `room` | Fixed lodging / room / suite if offered |

Confirm once against parks.org.il (and any other sources) what SKUs actually exist; drop unused buckets.

### Party-size buckets (to validate once)
| Bucket | Guests |
|--------|--------|
| `s` | 1–2 |
| `m` | 3–5 |
| `l` | 6–8 |

Sanity-check against real price tables (family packs, per-person vs per-unit). Adjust if sites price only per unit or have odd cutoffs (e.g. 4 / 8).

### Suggested table: `availability`

```text
campsite_id | stay_type | party_bucket | date | available | price | currency | scraped_at | source_url
```

Query pattern for the agent: “free Fri–Sat in 2 weeks, tent, 4 people, under ₪X”.

### Open questions
- Exact source of vacancy/price (parks.org.il booking pages? API?)
- Timezone / night vs calendar-day semantics
- How often to refresh (hourly vs nightly; peak weekends)

---

## 3. Site master data (+ description RAG)

### Goal
Canonical campsite profile + searchable description embeddings.

### Table: `campsites` (extend current)

| Field | Purpose |
|-------|---------|
| `campsite_id` | Stable ID (slug / parks ID) |
| `name_he` / `name_en` | Display |
| `location` | Lat/lng + region |
| `url` | Official page |
| `description_he` / `description_en` | Long text |
| `amenities` | Structured JSON if available |
| `ride_time_from_tlv` | Keep existing numeric filter |
| `price` | Deprecate as single field once `availability` exists (or keep as “from” price) |

### Description RAG
Option A: embed full description on `campsites`  
Option B (preferred): chunk descriptions into `site_chunks` with embeddings (same pattern as claims)

Agent uses:

- **claims RAG** → experiential / review-derived attributes  
- **site RAG** → official facts, location, amenities wording  

---

## 4. Conversation history

### Goal
Persist Telegram (and Streamlit) sessions as structured state the agent can resume, not only raw message logs.

### Table: `conversations`

| Field | Type | Notes |
|-------|------|-------|
| `conversation_id` | TEXT PK | Telegram chat id or UUID |
| `channel` | TEXT | `telegram` / `streamlit` |
| `updated_at` | TIMESTAMPTZ | |
| `messages` | JSONB | Optional full transcript |
| `state` | JSONB | Structured prefs (below) |

### Proposed `state` JSON shape

```json
{
  "hard_constraints": [
    {"field": "party_size", "op": "=", "value": 4, "strength": 1.0},
    {"field": "stay_type", "op": "=", "value": "tent", "strength": 1.0},
    {"field": "date_range", "op": "within", "value": ["2026-09-12", "2026-09-13"], "strength": 1.0}
  ],
  "soft_preferences": [
    {"query": "quiet at night", "strength": 0.8, "lang": "en"},
    {"query": "good for kids", "strength": 0.6, "lang": "en"},
    {"query": "not crowded", "strength": 0.4, "lang": "en"}
  ],
  "budget": {"op": "<=", "value": 500, "currency": "ILS", "strength": 0.9},
  "region_bias": [{"region": "negev", "strength": 0.5}],
  "exclusions": [{"campsite_id": "...", "reason": "already visited"}],
  "last_recommendations": ["id1", "id2"]
}
```

**Strength** ∈ `[0, 1]`:

- `1.0` = hard / must  
- `0.5–0.9` = important soft  
- `<0.5` = nice-to-have  

Update rules: merge on each user turn (LLM structured extract → merge with decay or explicit override). Hard constraints replace; soft prefs upsert by normalized query key.

### Open questions
- Keep full message history vs state-only + last N turns for LLM context
- PII / retention policy for Telegram

---

## 5. Agent (LangGraph) + Streamlit test UI

### Agent (extend `graph.py`)
Rough graph:

```text
START
  → light router / cleaner (existing)
  → state merge (update conversation JSON)
  → planner (semantic + numeric + availability constraints)
  → tools in parallel:
       search_claims | search_site_chunks | search_availability | search_campsites
  → rank / explain
  → recommender reply
END
```

Tools must be grounded: no invented prices or amenities.

### Streamlit (dev harness)
- Chat UI calling the same graph
- Sidebar: show extracted `state` JSON, tool traces, top retrieved claims
- “Reset conversation” / load fixture sessions
- Not for production users — Telegram later

---

## 6. Testing — LLM-as-judge + CI

### Eval set
JSON fixtures:

```json
{
  "query": "שקט, אוהל, עד 400 ש\"ח, סופ״ש הקרוב",
  "must_include_campsite_ids": ["..."],
  "must_exclude_campsite_ids": ["..."],
  "notes": "Fits quiet + tent + budget"
}
```

### LLM-as-judge
Judge rubric (pass/fail + short reason):

1. Recommended sites satisfy hard constraints (dates, party, budget when known)
2. Soft prefs reflected in evidence (claims / site text cited)
3. No hallucinated facts
4. Language matches user

### CI
- Unit: claim splitter, state merge, SQL builders
- Integration: graph on fixtures with mocked tools or seed DB
- Nightly / on-PR: LLM judge on a small golden set (cost-gated)
- Fail PR if judge score &lt; threshold or hard-constraint violations

---

## 7. Productionize (Nebius)

### Deployment options (decide in spike)

| Option | Pros | Cons |
|--------|------|------|
| **Serverless containers** (API + workers) | Simple, pay-per-use | Cold starts; long crawls awkward |
| **Always-on small API + cron/workers** | Stable Telegram latency | Cost when idle |
| **K8s** | Scaling, standard ops | Heavier than needed early |

**Recommendation:** start with **API service + scheduled ingestion jobs** on Nebius; revisit K8s only if multi-service ops demand it.

### Optimizations
- **Conversation KV cache**: load `conversations.state` by Telegram `chat_id` on each update; write-through after turn. Redis/KV or Postgres JSONB is enough initially; KV if latency matters.
- Embed / LLM response caching for identical soft queries
- Batch review ingestion; rate-limit Google + parks
- Connection pooling to Postgres
- Separate read path for RAG (replica later if needed)

### Observability
- Structured logs (conversation_id, tool calls, latency)
- Metrics: request rate, tool errors, embed queue depth, judge scores
- **Grafana** + Prometheus (or Nebius-native) dashboards
- Alerts: ingestion job failures, empty availability windows, error spikes

### Telegram
- Webhook → FastAPI → LangGraph
- Idempotent update handling
- Session restore from KV/DB

---

## 8. Writeup

Document for portfolio / handoff:

1. Problem & user (camping discovery in IL)
2. Architecture diagram (ingest → DB/RAG → agent → channels)
3. Claim splitting design + examples
4. Availability bucketing rationale
5. Conversation state model
6. Eval methodology (LLM-as-judge + golden set)
7. Production choices on Nebius (what we tried, what we kept)
8. Limitations & next steps

Artifacts: `docs/` diagrams, sample traces, CI badge, short demo video optional.

---

## Suggested phasing

| Phase | Scope | Outcome |
|-------|--------|---------|
| **P0** | Campsite crawl + extended `campsites` + description chunks | Master data + site RAG |
| **P1** | Google reviews ingest + claim splitter | Rich claims RAG |
| **P2** | Availability/price scraper + buckets | Date/budget/party queries work |
| **P3** | Conversation state table + merge logic | Multi-turn memory |
| **P4** | Streamlit harness + tool wiring in LangGraph | Demoable agent |
| **P5** | Golden eval + LLM judge + CI | Regression safety |
| **P6** | Nebius deploy, Telegram, caching, Grafana | Production path |
| **P7** | Writeup | External narrative |

---

## Decisions to lock early

1. Google reviews: official API only vs alternatives  
2. Vacancy source of truth on parks.org.il (and scrape legality)  
3. Party-size / stay-type bucket validation on real pages  
4. Conversation store: Postgres JSONB first vs Redis KV first  
5. Nebius target shape: serverless vs small always-on + jobs  

---

## Success criteria (MVP)

- User can ask in Hebrew for a quiet tent weekend under a budget and get **cited** recommendations  
- Availability/price for the next 2 weeks influence results when relevant  
- Multi-turn prefs persist across messages  
- Golden set passes CI judge threshold  
- Deployed endpoint reachable from Telegram with basic monitoring  
