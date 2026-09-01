# Trippy — Product & Engineering Plan

Campsite recommendation agent for Israel (parks.org.il + Google reviews), with RAG over claims/site data, availability/price search, and a Telegram-facing agent.

---

## Progress log

### Done (2026-09-01)

**Reviews + claims ingest.** Dropped the old `claims` table (author/date/stars on the claim, text `campsite_id`) and added `reviews` + FK’d `claims` (`014_reviews_and_claims`). Splitter is **one Google review per 235B call**, then one embed batch; drop `confidence < 0.5`; no aspect/locus yet. Claims store `claim` + `evidence_span` (`016`); dropped unused `claim_uid` (`017`); sentiment is `is_positive` bool, not a polarity string (`019`). **30B visit gate** before split: ads / brochure / history dumps stay on `reviews` with `skip_reason` / `skip_note` and get **no claims** (`020`). Experiments and locked choices: `docs/claims.md`. Hurshat Tal gold split vs 235B is judged by 30B (`test_hurshat_tal_claim_split.py`).

**Places fetch (legacy) into ingest.** `campsites.google_place_id` (`018`) from Text Search on `campsites.name`, **first hit only**. Dedicated חניון לילה pin when it exists, else the enclosing park — phase one does not hunt sibling listings. `just populate-reviews` pulls Place Details (newest weekly; `-- --most-relevant` also seeds Google’s best-of 5). CLI does not read JSON; tests still pass a reviews dict into `populate_reviews_and_claims()`. Independent of the INPA availability scrape. Pin mixing (day-visit vs overnight on the same park pin) is unsolved. Splitter still sometimes glues comma-lists of amenities into one claim.

**Up next:** recommender node (request → cited rec). Then pin mixing / amenity-list split quality, not more ingest plumbing.

### Done (2026-08-31)

Landed `info_website_names` so INPA booking types link to parks.org.il rate cards (exact name, else Qwen 30B — no more fuzzy match), then reworked dates around `date_intent` (Saturday one-night stays; weekend is Friday night only), split search out of `graph.py`, and added Ruff. Spiked Google reviews on legacy Places: seed `most_relevant`, refresh `newest`; Places API (New) has no newest sort. **Up next:** finish reviews ingest (started today), then the recommender so request → recommendation is a complete path.

**Google reviews spike (legacy Places)**
- `GOOGLE_API_KEY` Text Search for `חורשת טל חניון לילה` → one pin, **חורשת טל** (`ChIJDUZZZ2-8HhURv7LbSjS_yG0`, 4.3 / 2096). Overnight camping is not a separate listing.
- Places API (New) has no `reviewsSort=newest`. Legacy Place Details does: `reviews_sort=newest` vs `most_relevant` (hard cap 5; newest often star-only / empty text). Probe dumps in `temp/`.
- Locked ingest path: see §1.

**Info-site names vs booking types**
- `info_website_names` is the parks.org.il lodging product (`site_id` + classified name). Price scrape upserts those names and `list_prices` tariffs only — it no longer creates `accommodation_types`.
- Availability scrape get-or-creates types from INPA booking names and sets `info_website_name_id`: exact name first, else Qwen 30B over that hotel’s info-site names. Contains/fuzzy matching is gone. Planner quotes join `accommodation_types.info_website_name_id = list_prices.info_website_name_id`.

### Done (2026-08-30, evening)

**Two-stage planner (vacancies → amenity intersection)**
- Stage 1: catalog vacancies for the extractor stay window. Availability is always **one-night** rows; a type must have a row for every night in `[start, end)` (`GROUP BY` + `HAVING COUNT(DISTINCT start_date)`). Named park → `site_id`; else all sites. Party size uses `max_occupancy` (scrape is 1 adult — do not filter `availability.adults_no`). Prices from `list_prices` + `quote_night`.
- Stage 2: official amenity RAG **only on those type ids**. AND groups / OR values. A hit counts only when pgvector `<#>` ≤ **−0.8** (drops “least-bad” matches like tent ≈ air conditioning). Review claims are extra evidence, not the filter.
- One `ChatMessage`: `fits` (with `why`), `rejected` sample, `rejected_count`, `open_slots_query` (interpolated SQL). Recommender may pick only from `fits`. Empty `fits` and `rejected_count: 0` means stage 1 found no vacancy rows.
- Streamlit traces show the real vacancy SQL and fits payload, not a synthetic date tool.

### Done (2026-08-30, later)

**Info-site rate card (`source/scraper/info_site/`)**
- New `list_prices` table (`011_list_prices`): published parks.org.il tariffs (guest type, weekday/weekend, regular class), not INPA date slots
- Scraper reads `#table1` / `.tableMain[data-id=1]` (רגיל), classifies lodging rows with Qwen 30B into `info_website_names`, then snapshots `list_prices`
- Fee rows (`תוספת…`) are parsed and skipped; failing test left until persist lands
- Newsflash helpers (`newsflashes.py`) + failing persist test exist but are **not** called from `scrape.py --prices`
- Availability searches **1 adult**; creates booking `accommodation_types` and links them to `info_website_names` (exact or 30B)

### Done (2026-08-30)

Locked instruct model on **Qwen3-235B-A22B-Instruct-2507** for amenity extract + light/recommender (`QWEN_INSTRUCT_MODEL`). Agent **planner / query-constraint extract** stays on **Qwen3-30B-A3B** for now (`QWEN_INSTRUCT_30B_MODEL`) — easy to bump later. 30B failed to generalize named-place amenities off few-shot (Eilat); 235B passed on the same prompt. See “Locked — chat / extract model” below.

Schema: `notices` table for ephemeral official-site banners (e.g. “hot showers temporarily out of order”). Not catalog amenities, not review claims. Row is keyed by `site_id` + SHA-256 of the exact HTML element; next scrape deletes the row if that element is gone. Scraper + planner RAG not wired yet (`010_notices`).

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
- Rolls next **14 nights**, one night at a time, for configured adults (default **1**)
- Parses room offerings; strips `מספר N` suffixes and aggregates → `room_count`
- `accommodation_types` are created by the **info-site** scraper; availability only matches
- Upserts into:
  - `availability` (`site_id`, `start_date`, `end_date`, `accommodation_type_id`, `adults_no`, `room_count`, `scraped_at`)
- Re-scrape for a site/night **deletes existing rows first**, then inserts (avoids stale room types)
- Config: `source/scraper/config.json` (`nights`, `adults`, `limit_campsites`, …)
- SSL: OS trust store + relax `VERIFY_X509_STRICT` (corporate MITM)

**DB / migrations**
- Alembic + SQLAlchemy models (`db/models.py`, `alembic/versions/001_initial.py`)
- Nuke-and-pave local workflow documented in `db/README.md`
- Docker init only installs extensions; schema via Alembic

**Also in place (earlier)**
- LangGraph agent + claims RAG skeleton, FastAPI, Docker Compose Postgres/pgvector

### Next — sequenced (2026-08-31)

**1. Recommender node — close request → recommendation.** Extractor + planner `fits` exist; the recommender must pick from `fits` (with `why` / claims), never empty, so a Hebrew ask becomes a cited rec end-to-end. That is the first complete product path. Then other stuff.

**2. Google reviews leftovers** (ingest job itself has landed — `docs/claims.md` + §1). Pin mixing (campground inside a nature reserve); splitter sometimes under-splits amenity comma-lists. Planner already treats review claims as extra evidence, not the vacancy filter.

**3. Then other stuff** (not the current queue):
- CI (GitHub Actions): unit tests on PRs into `main` (`-m "not llm"` / no secrets). Golden-eval / LLM-judge later (§6).
- Extractor policy: “arrive Saturday afternoon” is a **policy / check-in** search — no extractor field or planner path yet. Weather + stargazing + Sat→Sun one-night are covered by `test_extractor_nice_weather_stars_saturday_afternoon_one_night`.
- Site-level `campsites.amenities` jsonb + GIN
- Notice scraper (`info_site/newsflashes.py`; not wired into `scrape.py` yet)
- Planner third RAG: `operator_notices` next to `stated_amenities` / `review_claims`
- Persist fee rows from the rate card (`תוספת יציאה מאוחרת`, extra caravan adult/child)
- Scrape other `#tableN` tabs (מנוי, חייל, קבוצה, אזרח ותיק, …) and `ציוד להשכרה`
- Listing-level **מה חדש** / site-wide ticker when `site_id` is unknown
- **Google pin mixing:** campground inside a nature reserve (חורשת טל) — reviews mix day-visit vs overnight. Unsolved; see §1.

**4. Conversation memory (after the rec path works).** One rolling **preferences list** vs keep the **entire transcript** for the LLM? Unsolved — see §4. Group trip: who wants what is **phase 2**, not MVP.

**5. Cloud / production-ready (later).** “Cloudifying everything” is the goal; we don’t know what that means yet (host, jobs, secrets, Telegram webhook, scraping cadence). Spike when the local path is complete; see §7.

**Extractor + planner (landed)**
- Structured prefs: `date: {start, end}`, `amenities` (AND list + `{op:"or", values}`), plus numeric/semantic leftovers
- Relative dates resolved in Python (Asia/Jerusalem)
- Named-place → type expansion is done by the **extract LLM** at ingest (not a place list / regex tool): e.g. Kineret also yields lake + body of water; Negev also yields desert. Same rule should apply when splitting review claims.
- Planner stage 1 filters one-night availability for the stay; stage 2 intersects official accommodation amenities (`<#> ≤ −0.8`) with `why` on each fit
- Booking types link to `info_website_names` (exact or 30B); quotes use `list_prices` via that id

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
- Nebius **Qwen3-235B-A22B-Instruct-2507** for amenity extract + agent light/recommender; **Qwen3-30B-A3B** for agent planner / query-constraint extract; Qwen embeddings for amenities + claims queries
- Postgres + pgvector + Alembic (`campsites`, `reviews`, `claims`, `notices`, `amenities`, `accommodation_types`, `availability`, `list_prices`)
- Scrapers under `source/scraper/`: discovery, booking IDs, info-site rate cards, availability/prices, amenity enrichment

This plan extends that into a full ingestion → retrieval → agent → eval → production stack.

---

## 1. Google reviews → claim RAG

### Goal
Ingest Google reviews per campsite, split into atomic claims, embed, store in `claims`.

### Locked — Places API (legacy) (2026-08-31)

**Choice:** [Places API (legacy)](https://developers.google.com/maps/documentation/places/web-service/search-text) only — Text Search + Place Details. Env: `GOOGLE_API_KEY`.

| Job | `reviews_sort` | Why |
|-----|----------------|-----|
| **Initial population** | `most_relevant` | Longer, more useful text; Google’s “best of” 5 |
| **Periodic updates** | `newest` | Catch fresh signal; many of the 5 are star-only (empty `text`) |

Hard cap is **5 reviews per Details call**. `newest` and `most_relevant` do not overlap (spike on חורשת טל).

**If legacy is ever deprecated:** Places API (New) (`places.googleapis.com/v1`) can search and return reviews, but **has no newest sort** (`reviewsSort` is rejected). Switching would lose incremental recency unless we move the “all recent reviews” path onto a scrape vendor (below). Revisit then; do not migrate early.

**AI review summary vs full reviews:** Google now ships a “what do people say about this place” / `reviewSummary` blurb. That is not enough for claim RAG (no dates, no atomic evidence, no overnight vs day-visit split). **Pulling recent reviews is still the right ingest.** Official API only gives 5; for **all** recent reviews use a **scraping service** (SerpAPI / Outscraper / similar) on a cadence, then the same splitter → embed → `claims` path. Prefer API 5+5 for the first pipeline; scrape-all is the completeness upgrade, not a replacement for splitting claims.

### Pipeline (cloud)

```
Campsite list → legacy Places Text Search (place_id)
  → Place Details: most_relevant (seed) | newest (refresh)
  → later: scrape vendor for the rest of recent reviews
  → populate_reviews_and_claims(campsite_id, reviews_dict)
       → upsert `reviews` (full text, stars, author, published_at)
       → 30B visit gate (drop ads / history dumps → skip_reason, no split)
       → split one review per 235B call
       → drop confidence < 0.5
       → embed kept `claim` (Qwen3-Embedding-8B, 1536)
       → replace that review’s `claims`
```

Places fetch is landed: `populate_google_place_id.py` then `just populate-reviews` (newest; `--most-relevant` for seed). Tests may still pass a reviews dict; production CLI does not read JSON.

### Claim splitting

Locked 2026-09-01 — details and probe tables in `docs/claims.md`.

- **Visit gate:** Qwen3-30B before split. Not a visit account (ad, brochure, history lecture) → `reviews.skip_reason = not_personal`, `skip_note`, zero claims. Splitter prompt and `confidence < 0.5` unchanged.
- **Model:** Qwen3-235B-A22B. Not 30B (over-splits incidents).
- **Batch:** one review per chat call (missing facts on 5-in-1 mattered more than stream/pool glue).
- **Filter:** omit generic overall judgments; drop any row with `confidence < 0.5`.
- **Aspect / locus:** not stored. Add later if we need SQL topic filters or amenity-key alignment; `text_en` stays the retrieval string.
- Stars, author, full text live on `reviews`. Claims have `review_id` + `campsite_id` only for those; recency is `reviews.published_at`. Sentiment is `is_positive` (nullable bool); splitter JSON still says `polarity: positive|negative`.

### Open questions
- Dedup: same claim from many reviews → keep multiplicity or collapse with frequency weight?
- Campground-inside-reserve pins: how to weight / filter mixed day-visit vs overnight reviews (see Next). Phase one stores the first Text Search hit.
- Splitter under-splits amenity comma-lists (lawns + faucets + fire pits as one row). Incident merge is locked; this is the remaining quality miss.
- Which scrape vendor for “all recent reviews” once the 5-cap is not enough

Weekly refresh is **newest** (`just populate-reviews`). Seed **most_relevant** is opt-in (`-- --most-relevant`).

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
- **notices RAG** → live official banners (outages / temporary closures); overrides stated amenities while the row exists  

### Table: `notices` (schema landed; scraper not wired)

Ephemeral operator notices from the official page — a third evidence type, not `stated_amenities` and not review `claims`.

Example: catalog still lists `hot_showers`; the site banner says “hot showers do not work temporarily.”

| Field | Purpose |
|-------|---------|
| `site_id` | FK `campsites.id` (CASCADE) |
| `source` | `inpa` / parks.org.il / … |
| `page_url` | Page where the banner was found |
| `notice_he` / `notice_en` | Normalized notice text for RAG / display |
| `html_element` | **Exact HTML node** that carried the notice |
| `html_element_sha256` | Unique with `site_id` (btree-safe; element text can be long) |
| `embedding` | Same Qwen 1536-d space as amenities / claims |
| `first_seen` | When we first stored this element |
| `last_seen` | Last scrape that still found the element |

**Lifecycle (scraper, later):**

1. Load existing notices for the site (`html_element` + hash).
2. If that exact element is still in the page → bump `last_seen`.
3. If the element is **missing** → `DELETE` the row (notice is gone).
4. New banner elements → insert (embed text, keep the raw HTML for the next check).

Do not put these in `claims` with `review_date = last_scraped`. `last_seen` is liveness (“we still see this banner”), not a guest stay date. A live notice beats `stated_amenities` for current status; reviews can corroborate but do not outrank a live official outage.

---

## 4. Conversation history

### Goal
Persist Telegram (and Streamlit) sessions so the agent can resume a trip plan across turns.

**Open — what to keep.** One rolling **preferences list** (merge/overwrite structured state, discard chatter) vs the **entire conversation** as LLM context (plus optional compacted prefs). Prefs-only is cheaper and stabler for hard constraints (dates, party, budget) but loses “we already ruled out X” nuance unless we store exclusions. Full transcript is faithful but long, noisy, and PII-heavy. Likely hybrid: structured prefs + last N turns + `last_recommendations` / exclusions. Not decided; do this **after** the request→recommendation path works.

**Phase 2 — group preferences.** A trip is often several people: one wants quiet, another wants a water park, someone else has a dog. Need per-person (or per-role) prefs, conflict surfacing (“Omri: quiet / Dana: kids water”), and whose constraint is hard vs soft. Out of MVP — single-user prefs first. Telegram groups make “who said what” a real identity problem (`from.id` vs chat id).

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
- Prefs list vs full transcript vs hybrid (above) — lock before building the table
- PII / retention policy for Telegram
- Group trips: per-person prefs + conflict UI (phase 2)

---

## 5. Agent (LangGraph) + Streamlit test UI

### Agent (extend `graph.py`)
Rough graph (current + planned):

```text
START
  → light router / cleaner (keep|drop; trivial short-circuit)
  → extractor (date + amenities OR groups + numeric/semantic)
  → planner / searcher:
       search_claims | amenity OR expand | search_campsites
  → recommender reply (never empty)
END
```

Tools must be grounded: no invented prices or amenities. Chat + claim query embeddings on Nebius Qwen (not OpenAI).

**Next after reviews ingest:** make the **recommender** the end of a complete path — request → extract → vacancy `fits` → amenity (+ claims) evidence → cited recommendation. Date search and accommodation amenity RAG already landed in the planner; recommender still needs to consume `fits` properly and never reply empty.

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

## 7. Productionize / cloudify (later; meaning TBD)

“Cloudifying everything so it’s production ready” is the intent. We **don’t know what that means yet** — host (Nebius vs other), always-on vs jobs, secrets, Telegram webhook, scrape cadence, what “done” looks like. Spike after the local request→recommendation path works. Notes below are a starting sketch, not a decision.

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
| **Now** | Recommender node | Complete path: request → cited recommendation |
| **Then** | Review leftovers | Pin mixing, amenity-list split quality, scrape-all vendor |
| **Then** | Other leftovers | Notices, rate-card tabs, CI, site amenities, … |
| **P0–P2, P4, P4b** | Crawl, availability, Streamlit, planner vacancies | **Landed** |
| **P1** | Reviews ingest | **Landed** (place_id + Places fetch + splitter + `is_positive`) |
| **P3** | Conversation memory | Prefs list vs full transcript — undecided |
| **P3b** | Group preferences (who wants what) | **Phase 2**, not MVP |
| **P5** | Golden eval + LLM judge + CI | Regression safety |
| **P6** | Cloud / production-ready | Meaning TBD; see §7 |
| **P7** | Writeup | External narrative |

---

## Decisions to lock early

1. **Google reviews — locked:** legacy Places API; seed `most_relevant`, refresh `newest` weekly; scrape vendor later for full recency. First Text Search hit → `campsites.google_place_id` (no sibling-park merge). Do not use Places API (New) until legacy dies (no newest). Do not substitute Google’s AI review summary for raw reviews. Splitter: 235B, one review/call, drop conf &lt; 0.5, no aspect/locus yet; `is_positive` on `claims` (`docs/claims.md`).  
2. Vacancy source of truth on parks.org.il (and scrape legality)  
3. Party-size / stay-type bucket validation on real pages  
4. Conversation store: one prefs list vs entire transcript vs hybrid — undecided; group prefs are phase 2  
5. Cloud / production-ready: meaning TBD (Nebius vs other, jobs vs always-on)  

### Locked — chat / extract model: Qwen3-235B-A22B (2026-08-30)

**Choice:** Nebius **Qwen3-235B-A22B-Instruct-2507** for amenity ingest extract and the agent light/recommender nodes (`QWEN_INSTRUCT_MODEL`). Exception: the agent **planner / query-constraint extract** stays on **Qwen3-30B-A3B-Instruct-2507** (`QWEN_INSTRUCT_30B_MODEL` → `planner_model` in `graph.py`) until we decide it needs the larger model.

**Why not stay on 30B-A3B:** the 30B extract prompt generalized poorly off few-shot place examples. “חוף אילת” invented Dead Sea / lake; Ramon and Kineret (in the prompt) worked. 235B with the **same** prompt passed all three (Eilat → beach + Red Sea, no Dead Sea).

**Why not a cheap 30B extract + dedicated place node:** after a prompt fix that path also hit 3/3 and was only ~1.6× extract-only 30B (~+$0.013 / 200 listings). We still picked 235B for ingest because (1) the $ delta vs 30B-only is small at our volume (~2× token price → about **+$0.02 per 200-listing scrape**), (2) one hop / one prompt is simpler. Agent query-constraint extract stays on 30B until we decide otherwise.

**Cost (Nebius Token Factory, 2026-08-30):** $0.20 / $0.60 per 1M in/out vs $0.10 / $0.30 on 30B-A3B. Embeddings stay `Qwen/Qwen3-Embedding-8B`.

**Revisit if:** scrape volume jumps an order of magnitude, or 235B latency/availability becomes a problem. Then consider 30B extract + 235B (or 30B) place node.

---

---

## Success criteria (MVP)

- User can ask in Hebrew for a quiet tent weekend under a budget and get **cited** recommendations  
- Availability/price for the next 2 weeks influence results when relevant  
- Multi-turn prefs persist across messages  
- Golden set passes CI judge threshold  
- Deployed endpoint reachable from Telegram with basic monitoring  
