# Trippy — Product & Engineering Plan

Campsite recommendation agent for Israel (parks.org.il + Google reviews), with RAG over claims/site data, availability/price search, and a Telegram-facing agent.

---

## Progress log

### Done (2026-08-27)

Spent the day mostly on amenities + getting a real local loop on the agent.

Morning on `pull-amenities`: taught accommodation types to carry richer listing detail (description, what’s not included), cleaned up the repo layout so agent / scraper / tests / docs aren’t dumped at the root, then pulled image URLs off the INPA HTML so each type can keep up to three photos.

That amenity work landed on `main` in the afternoon as a proper package — Nebius Qwen extracts structured details from tooltips, we embed amenity names, track LLM usage, and store policies / check-in·out / room_count. Availability scrape wires into that enrichment path now.

Then switched to `hook-agent-to-search-and-RAG` so we can poke the LangGraph without Telegram: threw up a Streamlit chat that hits the same graph, with a sidebar that dumps node/LLM/tool traces. Pointed agent chat at Nebius Qwen instruct (same model as amenity extract) and moved claims search embeddings onto the same Qwen embedder as amenities — no OpenAI for those anymore. Cleared the old claims rows so we don’t mix embedding spaces. Also chased a nasty empty-reply bug (Qwen was returning blank content when tools were bound; keep/drop was too strict on the Hebrew trip ask). Left a failing test for next time: “אני רוצה משהו לשישי הבא עם מים זורמים” should come back as `{date: [next Friday], amenities: [running water]}` instead of the old semantic_constraints blob.

### Done (as of 2026-08-24)

**Discovery / master data**
- Campsite listing crawler (`source/scraper/discover_sites.py`) → `campsites` (`id`, `name`, `url`)
- Booking-engine hotel ID discovery (`populate_availability_id.py`) → `campsites.booking_hotel_id` (e.g. `9_1`)
- Name matching between parks.org.il titles and secure-hotels.net names

**Availability scraper (INPA / secure-hotels.net)**
- `populate_availability.py` queries `BE_Results.aspx` HTML (no public API; prices live in embedded `roomData` JSON)
- Rolls next **14 nights**, one night at a time, for configured adults (default 2)
- Parses room offerings; strips `מספר N` suffixes and aggregates → `room_count`
- Upserts into:
  - `accommodation_types` (`id`, `name`) — created on first sight
  - `availability` (`site_id`, `start_date`, `end_date`, `accommodation_type_id`, `price`, `adults_no`, `room_count`, `scraped_at`)
- Re-scrape for a site/night **deletes existing rows first**, then inserts (avoids stale room types)
- Config: `source/scraper/config.json` (`nights`, `adults`, `limit_campsites`, …)
- SSL: OS trust store + relax `VERIFY_X509_STRICT` (corporate MITM)

**DB / migrations**
- Alembic + SQLAlchemy models (`db/models.py`, `alembic/versions/001_initial.py`)
- Nuke-and-pave local workflow documented in `db/README.md`
- Docker init only installs extensions; schema via Alembic

**Also in place (earlier)**
- LangGraph agent + claims RAG skeleton, FastAPI, Docker Compose Postgres/pgvector

### Next — planner constraints + amenity RAG wiring

**Planner output schema (failing test drives this)**
- Replace / extend `semantic_constraints` + `numeric_constraints` with structured prefs, e.g.:
  ```json
  {"date": ["2026-08-28"], "amenities": ["running water"]}
  ```
- Resolve “שישי הבא” to a concrete ISO date (timezone IL)
- Map amenity phrases to canonical keys / embeddings (`running water` → `water_hookup` / claim RAG)

**Still open from amenities track**
- Site-level `campsites.amenities` jsonb + GIN (unit-level mostly done via enrichment)
- Agent must search **accommodation** amenities for “room with shower”, not only site amenities
- Semantic amenity match explanations (“running water ≈ sink with faucet”)
- Re-embed `claims` with Qwen after clearing OpenAI vectors (table was truncated 2026-08-27)

### Later — second booking source + standardization

**Source:** [SimpleBooking glamping portal](https://www.simplebooking.it/portal/145/hotel/10516?lang=HE&cur=ILS&tid=99&guests=A%2CA&in=2026-08-24&out=2026-08-25)

**Problem:** INPA HTML `roomData` vs SimpleBooking (different URLs, payloads, naming) — need one internal model.

**Standardization sketch**
| Canonical field | Meaning |
|-----------------|--------|
| `source` | `inpa` / `simplebooking` / … |
| `external_hotel_id` | Per-source site id |
| `external_unit_id` | Per-source room/pitch id |
| `accommodation_type.name` | Normalized Hebrew/English label |
| `stay_kind` | `tent` / `trailer_pitch` / `glamping` / `room` / … |
| `amenities` | Shared jsonb schema (same keys across sources) |
| `availability` | Same table; source tagged or via site FK only |

Approach: adapter per source → normalize → upsert into the same `campsites` / `accommodation_types` / `availability` tables. Canonical amenity keys + optional embedding of raw labels for fuzzy match.

---

## Current baseline

Already in repo:

- LangGraph agent (`source/agent/graph.py`) with claims RAG + campsite list tool; production channel = Telegram (`main.py`)
- Local Streamlit harness (`scripts/streamlit_chat.py`) with node/LLM/tool traces
- Nebius Qwen instruct for agent chat; Nebius Qwen embeddings for amenities + claims queries
- Postgres + pgvector + Alembic (`campsites`, `claims`, `amenities`, `accommodation_types`, `availability`)
- Scrapers under `source/scraper/`: discovery, booking IDs, availability/prices, amenity enrichment

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
  → embed (Qwen3-Embedding-8B via Nebius, 1536 dims — same as amenities / claims search)
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
Rough graph (current + planned):

```text
START
  → light router / cleaner (keep|drop; trivial short-circuit)
  → planner (structured constraints — migrate to {date, amenities, …})
  → tools:
       search_claims | search_amenities / accommodation RAG | search_availability | search_campsites
  → recommender reply (never empty)
END
```

Tools must be grounded: no invented prices or amenities. Chat + claim query embeddings on Nebius Qwen (not OpenAI).

**Near-term:** make `test_planner_next_friday_running_water_constraint_schema` pass (`date` + `amenities` lists).

### Streamlit (dev harness) — landed on `hook-agent-to-search-and-RAG`
- Chat UI calling the same graph (no Telegram token)
- Sidebar / expanders: graph messages, per-turn LangGraph trace (nodes, prompts, tools)
- Reset conversation; JSON download of state/trace
- Not for production users — Telegram remains the product channel

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
| **P4** | Streamlit harness + tool wiring in LangGraph | Demoable agent — **Streamlit + Qwen wiring in progress on `hook-agent-to-search-and-RAG`** |
| **P4b** | Planner `{date, amenities}` schema + amenity/availability tools | Failing test → green; grounded date+amenity answers |
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
