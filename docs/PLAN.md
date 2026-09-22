# Trippy — Product & Engineering Plan

Campsite recommendation agent for Israel (parks.org.il + Google reviews), with RAG over claims/site data, availability/price search, and a Streamlit-facing agent.

---

## Progress log

### Done (2026-09-22, session keepalive replies nest like the interval)

**Session pings copy the LangSmith context onto their worker threads.**
Interval traces already showed `keepalive-light` and
`keepalive-recommender` under the parent. Session traces left those
same calls as separate roots, and the claim-judge reply was absent.
The judge reply is now a child too.

### Done (2026-09-22, recommender model call is on the trace)

**The Kimi stream keeps the caller’s LangSmith context.** The
first-token thread was dropping the model run, so the recommender
node input was the only thing on the trace. The packed prompt is now
a child of that node.

### Done (2026-09-22, unstated children are age 10)

**A child with no stated age is quoted as 10.** The recommender says
once that some lodging is priced differently by age. The booking link
no longer adds “Adjust the date on the booking page.”

### Done (2026-09-22, why does not split rules from amenities)

**The recommender writes amenities and rules as one account.** It still
names a miss: a rule or claim that says the thing is absent, an
explicit no, or a limit that misses the ask (entry at 20:00 vs 21:00).
Prompt only.

### Done (2026-09-22, why is stated amenities; rules follow the judge)

**`why` is only stated amenities on the unit.** Campsite rules stay
only when the judge names them in `relevant_rules`. Why-not names
campsites when there are one or two, and only a count from three up.
Supersedes leaving retrieved rules and embedding claims on `why`.

### Done (2026-09-22, why keeps only judge-relevant claims)

**A claim in `why` stays only if the judge named it in `relevant_claims`.**
The embedding hit that sent the site to the judge is not left on the
fit when the judge left that sentence out. Supersedes treating that
pre-judge claim as a pass.

### Done (2026-09-22, judge width 5 and a session warmup)

**Live claim judges run 5 at a time again.** Ten-wide made real turns
slower. A new Streamlit session also sends the claim-judge system
prompt alone, beside the model `hi` pings. Supersedes the
concurrency-10 default below.

### Done (2026-09-22, why-not splits rules from missing amenities)

**A polarity-false rule is its own why-not line.** Judge drops with no
such rule stay on the missing-amenity line. Price misses stay on the
quote's price line. Supersedes folding every judge drop into "missing".

### Done (2026-09-22, why-not keeps judge drops)

**Sites the claim judge rejects stay on the why-not line.** The line was
built before the judge, so a query whose amenity hits all went into fits
and then were dropped showed no why-not. Supersedes “why_not is only the
pre-judge funnel” in design.md.

### Done (2026-09-22, prod just recipes on Windows)

**`just prod-up`, `prod-load-sandbox`, and `prod-scrape` are no longer Unix-only.** They were hidden on Windows. The bodies are `docker compose`, so they run from the laptop too.

### Done (2026-09-22, search phase line)

**The assistant bubble names the phase.** Searching, then how many
candidates availability returned, then Ranking when recommend runs.
Supersedes the Thinking spinner in design.md.

### Done (2026-09-22, no periodic keepalive)

**Streamlit no longer starts the 10-minute model ping.** A new browser
session still sends one `hi` per model endpoint. Supersedes the
interval thread in design.md; `start_model_keepalive` remains for an
explicit call.

### Done (2026-09-22, reply language follows the query)

**An English query gets English why and intro.** The pack sets
`reply_language` from the query, and the prompt obeys that field.
Hebrew campsite names stay as stored. Supersedes “the model picks the
language” in the recommender section of design.md.

### Done (2026-09-22, why-not names the other sites)

**The why-not line names the other available campsites and the reason.**
In the query's language: "3 other sites have availability [A, B, C] but
they don't have pools." Price misses are the same shape. Supersedes the
count funnel in the recommender entry below.

### Done (2026-09-22, date ranges and booking party)

**Several fitting dates render as ranges, with one booking link.**
Check-ins collapse to `21–28.9`, then `11.10–15.10` for a later cluster. The link
is once per site, with a note to change the date. `ad1` is adults only;
stated children are `ch1`, so 2+2 is not four adults. Supersedes the
per-night link in the recommender entry below.

### Done (2026-09-22, recommender lists every date, top 3, and why not)

**The reply shows 2–3 stays, every fitting night, and a filter funnel.**
Consecutive one-night windows render as one check-in span. Price misses
stay on `rejected` (`reason: price`). `why_not` counts campsites with a
vacancy, inside the price range when the user set one, and missing each
requested amenity. The model still does not see `rejected`. Supersedes
“pick 1 or 2” in the recommender section of design.md.

### Done (2026-09-22, claim-judge concurrency 10)

**Live claim judges run 10 at a time.** Ten copies of one fridge job
were 9.9s and 11.7s at that width, against 13.1s and 21.3s at 5;
all 41 calls agreed and none errored. Each call slowed, and the
second wave disappeared. Supersedes the concurrency-5 default in
the 2026-09-10 compact-judge entry below.

### Open (2026-09-22, graph container)

**TODO: run the graph in its own container; Streamlit talks to it over HTTP.** Today `scripts/streamlit_chat.py` imports the graph, calls `build_graph`, and patches nodes and search functions in-process. The chat client should post a turn and read the reply, not host the agent.

### Done (2026-09-22, quote worker exits after its batch)

**A waiting worker runs one batch, returns the results, and exits.** A replacement is started as soon as that process dies, so the pool stays full and the next batch does not wait on spawn. Supersedes “workers stay up” in the long-lived quote workers entry below.

### Open (2026-09-22, per-quote sandbox children)

**TODO: one short-lived child per quote again.** Four workers stay up and each runs a whole batch with a 10s timeout, because spawning a process per quote used the 0.5s budget on startup (68 quotes, ~9s, every one "price function exceeded time limit"). Bring the per-quote child back once startup is not on that clock.

### Done (2026-09-22, long-lived quote workers)

**A `/quote` batch runs on one long-lived worker, 10 seconds for the batch.** Four workers stay up; a batch that exceeds 10s kills that worker and a new one takes its place. Supersedes four short-lived children in the entry below.

### Done (2026-09-22, weekend is Friday–Saturday)

**A stay is `weekend_holiday` when a night is Friday or Saturday.** Sunday is a weekday; the week starts on Sunday. `_rate_period_for_stay` had used Python's `weekday() >= 5`, which is Saturday and Sunday.

### Done (2026-09-22, search returns vacancies; the planner quotes)

**`search_open_slots` only returns vacant rows.** `planner_fits_payload` runs `quote_open_slots` and retrieve together on that list. The judge still waits until both finish. A price limit drops rows after retrieve, so a fit never keeps a slot the quote rejected.

### Done (2026-09-22, quote batch dedupes worker calls)

**`quote_batch` runs each distinct source and params once, then copies that answer onto every request in the batch.** The client posts the batch as received. Supersedes “posted once” in the four-children entry below.

### Done (2026-09-22, four short-lived quote children)

**`/quote` runs at most four child processes at a time; each evaluates one `quote()` and exits.** Identical site and params are posted once, and every request id gets that answer. Supersedes one-at-a-time children in the price-sandbox section of design.md.

### Done (2026-09-22, streamlit waits for a healthy sandbox)

**`just streamlit` starts `price-sandbox`, loads quote functions, then `docker compose up --wait` until the container is healthy, then the UI.** A load that returns before Docker's next healthcheck no longer starts Streamlit against an unhealthy container.

### Done (2026-09-22, local sandbox test stays off CI)

**`pytest -m local` checks the laptop price sandbox `/health` is ok and `loaded` is at least 1.** CI runs `pytest -m "not llm and not local"` because the runner has no sandbox container.

### Done (2026-09-22, restore drops experiments first)

**`just restore` / `just restore-latest` starts `db` and drops `experiments` before `pg_restore`.** `experiments.availability_frozen.id` defaults to `public.availability_id_seq`, so `--clean` could not drop that sequence. The schema is a disposable copy; `extensions` stays. Supersedes “`experiments` stays” in the restore-latest entry below.

### Done (2026-09-22, restore-latest)

**`just restore-latest` pulls the newest `postgres/trippy-*.dump` from `BACKUP_S3_BUCKET` and restores `public` into local Postgres.** `just restore` still takes an explicit path or `s3://` key. The Nebius console folder `postgres/` is that prefix.

### Done (2026-09-22, Streamlit requires a healthy sandbox)

**Streamlit does not start unless `/health` is ok.** Local `just streamlit` no longer passes `--if-up`. Prod starts `db` and `price-sandbox`, loads functions, then Streamlit, which `depends_on` `service_healthy`. The loader waits for `service_started` (503 is still listening); waiting for healthy deadlocked the load. `quote_night` remains the fallback only after a run has started. Supersedes “`--if-up` is laptop Streamlit only”.

### Done (2026-09-22, empty price sandbox is not healthy)

**`/health` is 503 until at least one `quote()` is loaded.** `ok: true` with `loaded: 0` let Docker mark the container up, and every quote then returned `unknown site`. urlopen fails on 503, so the container stays unhealthy. The loader still connects on 503 to POST `/load`, and exits 1 when that push stores nothing. Supersedes “empty until load” as a healthy state.

### Done (2026-09-22, extractor emits planned_exit_time)

**Departure is `planned_exit_time` (`HH:MM`), forwarded to `QuoteParams` like arrival.** The compiled `quote()` already takes it for the late-exit row. Omitted when the user did not say when they leave. Supersedes the arrival-only clock in the query-extractor section of design.md.

### Done (2026-09-22, extractor emits child_num and child_ages)

**The query extractor emits `child_num` and `child_ages`.** Those are the `QuoteParams` fields the compiled `quote()` already takes. `party_size` stays the whole party for occupancy; the quote subtracts `child_num` from it for `adults_num`. `guest_type` is still unset. Supersedes “child_num and child ages stay off until the extractor grows those fields” in design.md.

### Done (2026-09-22, search package init stays empty)

**`source/agent/search/__init__.py` does not re-export.** Callers import the submodule, the same way `recommender/` does. Supersedes the re-export sentence in the search-package entry below.

### Done (2026-09-22, search package)

**Catalog search lives in `source/agent/search/`.** Sandbox quotes, open slots, campsite names, query embeddings, amenities, rules, and claims are separate modules. `source.agent.search` still re-exports the names the planner calls. `_LAST_OPEN_SLOTS_QUERY` is read from the availability module, because that name is rebound on each search.

### Done (2026-09-22, one availability SQL for one night or many)

**A single stay uses the same query as several windows.** `date_range` is a one-item `date_windows`. `_open_slots_sql` is that query; the old single-range SQL is gone. Supersedes the split in the entry above.

### Done (2026-09-22, one availability search and one quote)

**The planner searches every stay window in one SQL, then quotes once.** A week of one-night stays was one `search_open_slots` and one `price_sandbox_quote` per night. Weekday and weekend prices share that quote; the rate follows each slot's dates. Supersedes the per-window loop. design.md "Planner claim/rule judge" and the LangSmith tool list.

### Done (2026-09-22, subcamp quote uses the parent function)

**`/quote` falls back to `parent_site_id` when the slot's site is not loaded.** Achziv north/south (37, 38) were `unknown site` because availability is on the subcamp and `site_price_functions` is on the parent (site 2). Open slots select `c.parent_id` and the sandbox uses that function only when the child id misses. design.md "Per-site price functions".

### Done (2026-09-22, date windows cap 20)

**`MAX_DATE_WINDOWS` is 20.** A full week fits, so Friday/Saturday
preference no longer trims `kind=week`. A longer horizon still
truncates and says so. Supersedes “the planner caps at 4 windows”
in design.md. `test_date_resolve.py`, `test_extractor_next_week.py`.

### Done (2026-09-22, recommender warmup removed)

**`warmup_recommender` is gone.** Keepalive and the Kimi→Super
first-token fallback cover a cold replica. Removed the package latch
and `test_recommender_warmup.py`. Supersedes
“`warmup_recommender` remains for its unit test” (2026-09-19).
design.md “Recommender”.

### Done (2026-09-22, recommender package)

**Recommender code lives in `source/agent/recommender/`.**
`recommend.py` packs, streams, and renders. Model ids and thinking-off
flags are `models.py`. Clocks, usage, and the timing log are
`timing.py`. Stream deadline is `stream.py`; Kimi→Super is
`fallback.py`. Supersedes the flat `recommend_*.py` modules below.
`test_recommender_kimi_fallback.py`. design.md “Recommender”.

### Done (2026-09-22, recommend_from_payload keeps the loop)

**`recommend_from_payload` is pack + original stream/finish again.**
Model-keyed client, first-token deadline, and a wrap through
`recommend_with_fallback`. Dropped `_stream_one` / `_finish_recommend`.
Supersedes the extra helpers in the fallback-module split below.
`test_recommender_kimi_fallback.py`. design.md “Recommender”.

### Done (2026-09-22, recommend fallback module)

**Kimi→Super lives in `source/agent/recommend_fallback.py`.** Timer,
primary call, Super arming, and the timeout retry. `recommender.py`
packs, streams one model, and finishes. `recommend_stream.py` is the
chunk iterator and deadline. Supersedes “recommender.py still …
falls back” below. `test_recommender_kimi_fallback.py`. design.md
“Recommender”.

### Done (2026-09-22, recommend stream types in recommend_stream.py)

**`RecommendCall`, `RecommendStream`, and `FirstTokenTimeout` live in
`source/agent/recommend_stream.py`.** Chunk iteration and the first-token
deadline sit with them. `recommender.py` still packs, paints, and
falls back. design.md “Recommender”.

### Done (2026-09-22, Kimi fallback is not recommend_from_payload)

**Fallback is `_recommend_with_fallback`, not the pack/render
entry.** `recommend_from_payload` packs fits, picks the primary
call, and finishes the stream. One-model stream is `_stream_one`;
Kimi→Super lives in `_kimi_super_fallback` + the timeout wrapper.
Supersedes the “stuffed into recommend_from_payload” shape of the
entry below. `test_recommender_kimi_fallback.py`. design.md
“Recommender”.

### Done (2026-09-22, Kimi recommend falls back to Super)

**Kimi first-token 10 s, then Nemotron Super.** Token Factory can
queue Kimi for minutes while Super answers a `hi` in ~1 s. The
recommender waits `TRIPPY_KIMI_TTFT_SEC` (default 10) for a stream
token, then replays the pack on Super. Injected `chat` does not fall
back. `test_recommender_kimi_fallback.py`. design.md “Recommender”.

### Done (2026-09-22, always forward planned_entry_time)

**Planner always passes `planned_entry_time`.** Omitting it when unset
needed an extra kwargs dict so exact mock matches would still pass.
`search_open_slots` already treats None the same as omitted; tests now
include `planned_entry_time=None`. Supersedes “omit unset
planned_entry_time” below. design.md “Query extractor: date_intent”.

### Done (2026-09-22, omit unset planned_entry_time)

**Planner does not pass `planned_entry_time=None`.** `search_open_slots`
already defaults it. Forwarding None made mock assertions miss the
kwarg and would hide a missing default. Only pass the clock when the
extractor set one. design.md “Query extractor: date_intent”.

### Done (2026-09-22, resolve_dates is not a tool)

**Extractor does not wrap `resolve_dates` as a LangChain tool.** It was
never `bind_tools`'d (empty Qwen content); `extractor_node` invoked it
after JSON, and still swallowed leftover `tool_calls`. `normalize_constraints`
calls `resolve_dates` directly. Dropped `resolve_dates_tool`,
`constraints_from_tool_calls`, and the prompt's "resolve_dates tool".
design.md “Query extractor: date_intent”.

### Done (2026-09-21, tomorrow is on=tomorrow)

**“for tomorrow” / מחר is `on=tomorrow`, not today.** Schema `on`
was `"YYYY-MM-DD" | "today"` and the prompt forbade ISO, so the 235B
mapped tomorrow onto the only relative token. `resolve_dates` now
offsets `tomorrow` by one day (`tonight` stays this night). Few-shot
of the live English miss plus bare `מחר`.
`test_date_intent_tomorrow.py`, `test_extractor_tomorrow.py`.
design.md “Query extractor: date_intent”. experiments.md 2026-09-21 §1.

### Done (2026-09-20, availability scrape is 4 weeks)

**`scrape-availability` walks 28 nights, not 14.**
`config.json` `availability.nights` and the populate fallback.
design.md “Where it runs”.

### Done (2026-09-20, jail → prices sandbox)

**Comments and design say prices sandbox, not jail.** Compose
`price-sandbox`, package `source.price_sandbox`. PLAN.md entries that
said jail are left as written. design.md “Where it runs”.

### Done (2026-09-20, sandbox loader is required)

**GitHub Actions always runs `price-sandbox-loader` after a scrape.**
Skipping when the compose service was missing left Streamlit on last
week's `quote()` (or `quote_night`) with a green job. `no such
service` fails the run. `--if-up` stays laptop-only (`just streamlit`
when Compose is down). design.md “Where it runs”.

### Done (2026-09-19, planned_entry_time)

**“אפשר להיכנס אחרי 19” is `planned_entry_time`, not dropped.** The
extractor prompt used to omit arrival / check-in until a policy field
existed, and the summer-stargazing few-shot demonstrated dropping
Saturday afternoon. Schema field `planned_entry_time` (`HH:MM`); still
not semantic RAG. Normalize accepts `19` / `19:00`. Planner passes it
to sandbox quotes. Sites are not yet filtered on gate hours.
`test_planned_entry_time.py`, `test_extractor_late_entry.py`.
Supersedes the “no extractor field or planner path yet” note below.

### Done (2026-09-19, keepalive 10 min)

**Keepalive interval is 10 minutes, not 4.** Default
`DEFAULT_INTERVAL_SEC=600`. Same skip-if-just-used window.
`TRIPPY_KEEPALIVE_INTERVAL_SEC` still overrides.
`test_default_keepalive_interval_is_ten_minutes`. Supersedes the
4-minute default below.

### Done (2026-09-19, why availability is 2 adults)

**A 1-adult INPA search returns tent vacancies that only fit one
person; the same stay is empty for two.** That is treated as a
booking-engine bug, so the scrape always asks for 2 adults and those
singleton pitches never enter `availability`. Planner party size is
still `max_occupancy`. `test_config_and_search_url_use_two_adults`.
Supersedes the entry below.

### Done (2026-09-19, scrape availability as 2 adults)

**`scrape-availability` GETs INPA as 2 adults.** Config
`availability.adults` and `search_url` / scrape fallbacks are 2.
Planner party size is still `max_occupancy`; `availability` still has
no `adults_no`. Supersedes the “scrape is 1 adult” notes below.

### Done (2026-09-19, skip warm embed keepalive)

**Embed keepalive no-ops if retrieve (or a ping) just used
`Qwen3-Embedding-8B`.** A 4s interval ping after a planner embed was
a second cold replica, not a useful keep-warm. A second
`model-keepalive` thread after a Streamlit rerun is also refused.

### Done (2026-09-19, Streamlit Ctrl+C is not clear-cache)

**`client.toolbarMode=viewer`.** Streamlit’s “c” shortcut opened
Clear cache on Ctrl+C in the page. Developer menu items (rerun,
clear cache) stay in the hamburger’s absence; refresh still reruns.

### Done (2026-09-19, keepalive one ping per model)

**Keepalive is per Nebius endpoint, not per role.** Light and
extractor share the 235B so they were pinged twice. The embedder
(`Qwen3-Embedding-8B`) is in the same round.

### Done (2026-09-19, retry attempts on scrape cost)

**Timed-out Nebius chat attempts are added to `LlmUsage`.**
`nebius_chat_create` records each try (real tokens on success,
chars/4 prompt estimate on connect/timeout). The OpenAI client
`max_retries` is 0 so the SDK cannot bill a retry the report never
sees. Compile fix/regen turns were already counted.

### Done (2026-09-19, date-window tests match unit fits)

**`test_planner_loops_date_windows` and `test_planner_caps_windows_at_four`
assert one fit and the nights on `dates`.** Vacancy search is still
one `search_open_slots` per window. Supersedes the “two date-resolve
tests still expect one fit per window” note below.

### Done (2026-09-19, one fit per site+type)

**Same unit across date windows is one fit.** Retrieve already keyed
on campsite + accommodation type; the judge already keyed on
campsite + query. The planner still emitted one fit per night, so
the recommender saw four copies of the same tent. Fits now carry
`dates` (each night’s price and stay). `start` / `end` stay the
first night. Two date-resolve tests still expect one fit per window.

### Done (2026-09-19, quote cache per request)

**Quote memo is per user request, not process-wide.**
`planner_fits_payload` opens `price_quote_cache()` around the date
windows so four Fridays still share one jail POST, and the next
chat turn starts empty. Supersedes the process-lifetime cache in
the entry below.

### Done (2026-09-19, quote cache)

**Jail quotes are memoized by site + lodging + party + weekday/weekend.**
The planner calls `search_open_slots` once per date window, so four
Fridays used to POST the same campsite four times. Hits skip the
sandbox (`cached: true` on the Streamlit/LangSmith row). List-price
fallback is memoized the same way. `clear_price_quote_cache` after a
sandbox reload.

### Done (2026-09-19, sandbox quote batch)

**Jail quotes over 30 were all `no_function`.** The sandbox
`MAX_BATCH` is 30; `search_open_slots` can send 80 unique
site+lodging keys in one POST, the server rejects the batch, and
the client dropped `ok: false` rows. `quote_replies` now chunks and
keeps the real price or jail error on the LangSmith/Streamlit row.

### Done (2026-09-19, sandbox quote trace)

**Jail quotes show in Streamlit and LangSmith.** `POST /quote` lived
inside `search_open_slots` with no span, so a load miss looked like
`quote_night`. `price_sandbox_quote` is a `@traceable` tool (one row
per campsite: params, price, explanation, or skip/error). Streamlit
expands **Price sandbox** on the turn.

### Done (2026-09-19, availability_with_names)

**`availability_with_names` view.** Same job as
`campsite_rules_with_names`: campsite name and accommodation type
name next to each vacancy row. Alembic `041`.

### Done (2026-09-19, laptop sandbox port)

**Laptop `price-sandbox` was healthy but not on `127.0.0.1:8503`.**
The container was only on the internal `quote` network, so Compose
did not publish the host bind. It now also joins `quote-host` (not
`default` — that would give the jail a route to Postgres). Loader
and host Streamlit can reach `http://127.0.0.1:8503`.

### Done (2026-09-19, uv trampoline)

**`just streamlit` died after the loader on Windows:**
`uv trampoline failed to canonicalize script path`. `uv run streamlit`
goes through `.venv/Scripts/streamlit.exe`; `uv run python -m streamlit`
does not. Same change for `just update-tables` (`alembic`).

### Done (2026-09-19, OPENSSL_Applink is SSLKEYLOGFILE)

**`just streamlit` died in `load-price-sandbox`.** Norton sets
`SSLKEYLOGFILE=\\.\nllMonFltProxy\…`. `urllib.request.urlopen` builds
an HTTPS handler even for `http://127.0.0.1:8503/health`, and that
calls `ssl._create_default_https_context` — still the stdlib helper
after the earlier `create_default_context` patch. OpenSSL then
`fopen`s the device and aborts (`OPENSSL_Applink`). `tls.py` now
drops `SSLKEYLOGFILE` on win32 and replaces both helpers. The
OPENSSLDIR leftover in the entry below was the wrong cause.
Supersedes “Windows OPENSSL_Applink” below for the crash itself.

### Done (2026-09-19, Windows OPENSSL_Applink)

**`ssl.create_default_context()` crashes uv Python 3.14 on Windows**
when OpenSSL-Win64 leftover OPENSSLDIR is
`C:\\Program Files\\Common Files\\SSL` (`OPENSSL_Applink`).
`source/scraper/tls.py` now builds via `SSLContext` + certifi and
replaces `ssl.create_default_context` so LangChain's import survives.
Import tls before `langchain_openai`. Same certifi /
`TLS_TRUST_OS_STORE` policy as the 2026-09-04 TLS note.

### Done (2026-09-19, keepalive per session)

**Session `hi` is per browser tab; the 4-minute loop is per process.**
`ping_new_session(st.session_state)` on Streamlit session init; Reset
does not re-ping. `start_model_keepalive` only starts the interval
thread (first ping after 240 s, not immediately). LangSmith run names
`model-keepalive-session` vs `model-keepalive-interval`.

### Done (2026-09-19, keepalive 4 min)

**Keepalive stays at 4 minutes.** ~$1.20/month for three `hi` pings
is cheap enough. Supersedes the hourly default in the entry below.

### Done (2026-09-19, keepalive hourly)

**Keepalive interval is 1 hour, not 240 s.** Default
`TRIPPY_KEEPALIVE_INTERVAL_SEC=3600`. Same three `hi` pings. Supersedes
the 240 s cadence in the entry below.

### Done (2026-09-19, model keepalive)

**Streamlit keeps recommender, light, and extractor warm.** Nebius
was going cold after idle; the old one-shot Kimi `hi` never showed
on LangSmith (background invoke, no run name) and never pinged 235B.
On first load a daemon thread now pings all three with `hi`
(`max_tokens=5`) and repeats every 240 s. LangSmith: tag `keepalive`,
run name `model-keepalive`. `TRIPPY_KEEPALIVE_INTERVAL_SEC` overrides
the interval. Supersedes the 2026-09-12 Streamlit Kimi warmup for the
chat path; `warmup_recommender` remains for its unit test.

### Done (2026-09-17, print site_price_functions store times)

**scrape-prices prints `scraped_at` / `updated_at` from the upsert.**
The table is one row per campsite (`site_id` PK); a GitHub Actions
re-run updates that row, it does not add another. The banner, the
`compile:` line, the run-end recap, and `report.md` all show the
timestamps RETURNING from Postgres. design.md compile / store.

### Done (2026-09-17, eu-north1)

**VM and object store are both Finland (`eu-north1`).** Not
`me-west1`. Scrape/backup cron stays IDT (when we want the job, not
the datacenter clock). Supersedes the “bucket not the VM's region”
note below. cloud.md; design.md “Where it runs”.

### Done (2026-09-17, backup region)

**Object store is `eu-north1`, not the VM's `me-west1`.** Endpoint
`https://storage.eu-north1.nebius.cloud`. PutObject AccessDenied was
the `accesskey-e00…` resource id pasted as `AWS_SECRET_ACCESS_KEY`.
cloud.md; `.env.example`.

### Done (2026-09-17, backup summary)

**Actions backup must upload.** `BACKUP_S3_BUCKET` unset used to
succeed after a local dump only. `backup.yml` now fails that, and
the Summary is `Backup was written to s3://…` on success. cloud.md.

### Done (2026-09-17, backup dir)

**Actions dumps to `~/.trippy-backups`.** `gh-actions` is not root;
`/var/lib/trippy/backups` from bootstrap is `700` and the first
`backup.yml` run died with permission denied. Same home-dir pattern
as scrape reports. cloud.md; design.md “Where it runs”.

### Done (2026-09-17, afternoon dump)

**One dump per day at 14:00 IDT**, not before a scrape.
`backup.yml` (`0 11 * * *`; 13:00 IST in winter) SSHs
`scripts/cloud/backup.sh`. Availability, reviews, and manual
`scrape.yml` pass `backup: false`. Supersedes the
“dumps immediately before scrape-availability” bit of the entry
below. design.md “Where it runs”; cloud.md.

### Done (2026-09-17)

**Public-schema dumps to Nebius object storage.** `just backup` /
`just restore` (docker compose cp; not a shell redirect) and
`scripts/cloud/backup.sh` write `pg_dump -n public -Fc` only —
`experiments` stays out. GitHub Actions dumps immediately before
`scrape-availability` (and before manual `scrape.yml`); reviews do
not. No VM cron. Destructive laptop `scrape-info` / `clear-*` dump
first unless `TRIPPY_SCHEMA=experiments`. design.md “Where it runs”;
cloud.md.
### Done (2026-09-17, store AST-ok quote() even when gold fails)

**scrape-prices writes `site_price_functions` after retries if AST
passed**, including gold misses (בארות / תל ערד). AST/static still
does not store. Report keeps them under Failures as
`gold failed (stored updated)`.

### Done (2026-09-17, GuestType only on per-person rates)

**Compile prompt:** soldier / Matmon / miluim / student / senior /
disabled change per-person rates only. Per-unit (חושה, family tent)
ignores `guest_type` unless that unit has its own identity rows.

### Done (2026-09-17, מעל X is X and up; per-person vs per-unit keys)

**Compile prompt:** "מעל X לנים" / "X ומעלה" → `GROUP_MIN` is X, not
X+1 (הקסטל used 31). Per-person schedules (tent, group) use `adult` /
`child`; per-unit (חושה, family tent) use `weekday` / `weekend` and
must not read `rates["adult"]`.

### Done (2026-09-17, drop Scrape / Scrape job from Actions)

**Deleted `scrape.yml`** (claims / info / sites / place-ids dropdown).
Those stay `just prod-scrape <job>` on the VM. **`scrape-job.yml` is
now `.github/actions/scrape-job`** so GitHub does not list it as a
runnable workflow. Availability, reviews, and prices checkout the
repo and call the composite action.

### Done (2026-09-17, scrape-prices GitHub Action)

**`scrape-prices.yml`** is `workflow_dispatch` (extra args `--site 2`),
same SSH/`concurrency: scrape` path as availability and reviews.
`job.sh prices` already existed. Summary is `prices.md` (`PRICES_REPORT_PATH`);
dumps go to `PRICES_REPORT_DIR=/reports` → `~/.trippy-scrape/<timestamp>/`.
Loader still runs after the scrape.

### Done (2026-09-17, one folder per scrape-prices run)

**Each scrape-prices run writes `reports/scrape_prices/<timestamp>/`**
(`2.py`, `2_v1.py`, prompts, `report.md`). Older flat
`reports/price_functions/` dumps are leftover from before this.

### Done (2026-09-17, gold fail banner + interrupt report; late-exit is a surcharge)

**Gold misses print `!!! PRICE FUNCTION GOLD FAILED !!!`** like AST, and
an interrupted scrape still writes the Markdown report (Ctrl+C at משמר
lost the summary). Listing-match prompt: תוספת יציאה מאוחרת / תוספת אדם
are rate words on that unit — אכזיב skipped `תוספת יציאה מאוחרת חושה`
at 0.60 so gold 675 never saw 225.

### Done (2026-09-17, more quote() builtins: any, all, dict, …)

**Allowlisted the remaining harmless sequence/conversion builtins** on
generated `quote()`: `any`, `all`, `reversed`, `dict`, `set`,
`frozenset`, `iter`, `divmod`, `isinstance`, `pow`, `format`. Not
`open` / `eval` / `getattr`. תל ערד had failed on `any`.

### Done (2026-09-17, filter allowlisted in price functions)

**`filter` is an allowed builtin** in generated `quote()` (`ast_check`),
same as `map` / `next`. מצדה's first compile used it and was rejected
until a fix rewrite.

### Done (2026-09-17, retry Nebius/page connection errors)

**scrape-prices retries DNS/connect blips.** `nebius_chat_create` waits
2s / 8s / 20s on `APIConnectionError` / timeout; the OpenAI client
`max_retries` is 6. Page fetches retry twice. A site that still fails
is recorded and the run continues (Castel died the whole job on
`getaddrinfo failed`).

### Done (2026-09-17, compile prompt has no park names or live tariffs)

**Compile SYSTEM_PROMPT is generic.** No park name, no INPA band
(76/58), no חושה 350/450, no `GROUP_MIN = 30`, no "עד 4/5 לנים".
Example rates are invented (10/8, 100/120). Occupancy is עד N / cap M.
The user message still names the campsite being compiled.

### Done (2026-09-17, beerot gold uses catalog lodging names)

**בארות gold `lodging` is `info_website_names`, not rate-card nicknames.**
`חדר צוות קטן` → `חדרי צוות`; `חדר צוות כפול` → `חדר צוות מאובזר כפול`;
`חדר צוות גדול` (rooms 5–6, same tariff) → `חדר צוות מאובזר` and
`חדר צוות מאובזר ומונגש`. Prices unchanged.

### Done (2026-09-17, try/except allowed; Tel Arad numbers out of the prompt)

**Price-function AST allows `try`/`except`** (`Exception` / `KeyError` /
`TypeError` in the sandbox builtins). Compile prompt no longer forbids
it, and no longer names תל ערד 860/3080 — that few-shot was copied
onto the small mahal. scrape-prices Markdown report omits campsite
URLs.

### Done (2026-09-17, scrape-prices 13/15/17 re-run)

**Re-ran the three compile misses on experiments.** הבשור stored on
the first compile. תל ערד occupancy regen fired (3096 vs 3080) and
still failed. בארות fix turned gold lodging-name misses into
`try/except`. Report `reports/scrape_prices/2026-09-17_063211.md`.
experiments.md 2026-09-17 §2.

### Done (2026-09-17, scrape-prices run report and versioned fail dumps)

**Each scrape-prices run writes a Markdown report** under
`reports/scrape_prices/<timestamp>.md` (stored vs failed, retry kind,
failing gold / AST lines, dump paths, cost by role). Failed compile
attempts are kept as `reports/price_functions/<id>_vN.py` and
`<id>_vN.prompt.txt`; `<id>.py` is still the latest attempt. The fix
turn dumps `FIX_SYSTEM_PROMPT`.

### Done (2026-09-17, compile retries: fix vs occupancy regenerate)

**One retry after a failed price-function compile.** AST / syntax /
NameError / static → `compile_quote_fix` (the function + error text,
`price_function_compile_fix`). Gold price misses only → same rate-card
prompt plus occupancy class, no expected numbers
(`price_function_compile_retry`). `assess_compiled_source` decides
which. Supersedes “retry policy … not wired” below.

### Done (2026-09-16, while/insert and occupancy in the compile prompt)

**`while` and `list.insert` are allowlisted** (`ast_check`). Yehudiya
and Mishmar failed those on the 18-site scrape, not on prices.
Compile prompt now states included occupancy = עד N on the unit row
(תוספת is N+1; a notes cap is not included), two published sizes are
separate Lodging members (Tel Arad 860 vs 3080), and free under-5s
live in one name (`toddler_count`) — the NameErrors were typos
(`todder_count` / `toddlers_count`), not prompt variants. Retry policy in design.md (not wired): AST/syntax/NameError is a
fix-the-function turn; gold occupancy is a regenerate without the
expected number; not temp>0. Supersedes the “retry is a maybe” note
below.

### Done (2026-09-16, gold JSON per campsite)

**Gold is prices in JSON, not Python helpers.** One
`source/price_sandbox/gold/sites/<slug>.json` per park; `gold/runner.py`
loads the file whose `match` sits in the URL and runs `quote()` against
`expected_price`. Deleted `cases.py` `BAND_76` / `BAND_64` / `BAND_47`
and the per-site `*.py` modules that imported them — a shared band hid
that בארות student is 53₪. Numbers from the published cards (prompts
1–16 plus a live fetch of all 18 parks.org.il pages). Supersedes
“gold file per campsite” Python modules below.

### Done (2026-09-16, gold file per campsite)

**Every campsite has full gold, not only Achziv.** One module per park
under `source/price_sandbox/gold/sites/` (מצדה, חורשת טל, …). Tent
identities include miluim / student / disabled; group occupancy 30 on
every card that publishes קבוצה (all captured INPA cards); extra cases
for each lodging on that dump (family tent, couple tent, mahal, staff,
pitch, caravan, tokul, חושה). Numbers from scrape-prices prompts 1–16;
בארות / יוטבתה from the 64₪ band after the scrape stopped. Supersedes
“wait on a captured card”. test_price_gold_all_sites.py; design.md
“Per-site price functions”.

### Done (2026-09-16, strip quotes from lodging and guest_type)

**Lodging and guest_type identifiers never keep quotation marks.**
`צה"ל` in `נכה צה"ל ומלווה` made the 235B emit an unterminated Python
string. `strip_type_quotes` on `QuoteParams` and on compile prompt
names; gold `DISABLED` is `נכה צהל ומלווה`. design.md “Per-site price
functions”.

### Done (2026-09-16, one-shot sandbox loader)

**A short-lived loader fills the sandbox; Streamlit only quotes.**
`python -m source.price_sandbox.load` reads `site_price_functions`,
`POST /load`, exits. `just load-price-sandbox` (laptop; also from
`just streamlit` / `scrape-prices` / `run-eval` with `--if-up`).
Prod: `price-sandbox-loader` on `default`+`quote`, `just prod-load-sandbox`,
hooked from `prod-up`, `prod-scrape`, bootstrap, and deploy. The jail
image deletes `load.py`. Search no longer `POST /load`s. Supersedes
the “maybe a one-shot sandbox loader” note below.
design.md “Per-site price functions”.

### Open (2026-09-16, maybe a one-shot sandbox loader)

**Streamlit (or the planner on first quote) still `POST /load`s
functions into the sandbox.** That is the right isolation today: the
jail has no Postgres. A later shape could be a short-lived loader
process — reads `site_price_functions`, `POST /load`, exits — so
Streamlit only quotes. Same security split (loader has the DB, sandbox
does not). Worth it if `/load` from the UI becomes a footgun (two
clients, last writer wins) or if compose-up is expected to leave the
sandbox already filled. Not doing it at ~one Streamlit and a handful
of functions. design.md “Per-site price functions”.

### Done (2026-09-16, copy drops leftover experiments tables)

**`just setup-experiments copy` now drops experiments tables that are
not in `public`.** Pytest fixtures and ad-hoc scripts leave extra
tables; `clone_tables` only rebuilt the public set, so leftovers
survived. `availability_frozen` is kept. `db.experiments.drop_experiments_leftovers`;
`just setup-experiments status` lists leftovers if any remain.

### Done (2026-09-16, loud store-ok for price functions)

**A passing compile prints `PRICE FUNCTION ADDED TO DB` (or UPDATED /
UNCHANGED).** `store_price_function` returns `inserted` when there was
no prior row — the Achziv run’s `updated` was that first upsert.
scrape.py `_print_store_ok`.

### Done (2026-09-16, loud AST compile failure)

**AST allowlist / static-check failures print a banner.** The quiet
`price function compile failed: call to 'map'…` line was easy to miss
under listing-match noise. scrape.py `_print_ast_failure`.

### Done (2026-09-16, map is allowed in quote())

**`map` is on the price-function allowlist.** Achziv compiles were
dying on `map(int, planned_exit_time.split(":"))` before gold.
test_price_function_map.py; design.md “Per-site price functions”.

### Done (2026-09-16, gold covers every identity guest_type)

**Matmon was already in gold; miluim / student / disabled / senior were
not.** Achziv now has one tent case per identity tab. Every other site
has regular, Matmon, soldier, and senior. Miluim/student/disabled on
other parks wait on a captured card. test_price_gold_coverage.py;
design.md “Per-site price functions”.

### Done (2026-09-16, gold covers soldier, group, each lodging)

**Gold recall was five mixed prices; that missed most of the card.**
Every site now has a soldier case. Achziv has occupancy-30 group
(regular and Matmon-override) and one weekday case per חושה product.
Horashat adds staff, wood staff, and caravan. `_tent_band(..., extra=)`
appends instead of replacing, so extra lodgings no longer drop soldier.
Group cases for other sites wait on a captured threshold. Supersedes
“five gold cases per site”. test_price_gold_coverage.py; design.md
“Per-site price functions”.

### Done (2026-09-16, קבוצה is an occupancy override)

**GuestType is never group.** The קבוצה tableTab is a schedule selected
by party size, not an identity the caller can pass. Compile omits it
from the GuestType enum list; `GuestType.GROUP` is a static compile
failure. When `adults_num + child_num` meets that site's notes
(`GROUP_MIN`), group rates always override identity (including Matmon).
Syntactic unreachable code (after return/raise, `if False`) is rejected
without executing quote(); that is AST control flow, not dataflow.
Compile retries with the error in a follow-up turn are a maybe — after
the prompt is settled, if AST/gold failures stay common. Do not retry
gold by sending the expected price.
test_price_function_group.py, test_price_function_unreachable.py;
design.md “Per-site price functions”.

### Done (2026-09-16, quote() does not scan labels)

**Rate-card labels are interpreted at compile time.** The prompt
requires `RATES[lodging][guest_type]` values to be named fields
(`adult`, `child`, `weekday`, `weekend`, `late_exit`, …), not
`{"label", "price"}` rows. `"מבוגר" in label` is a compile failure
(`runtime_string_scan_hits`). Achziv 2.py still has that pattern
because it was compiled before this rule. test_price_function_rate_keys.py;
design.md “Per-site price functions”.

### Done (2026-09-16, enum vars keep parameter names)

**Parsed enums reuse `lodging` and `guest_type`.** Compile prompt:
`lodging = Lodging(lodging)`, `guest_type = GuestType(guest_type)` —
no `unit` / `tab`. Supersedes the `unit =` / `tab =` sentence in the
enum entry below.

### Done (2026-09-16, quote() uses Lodging and GuestType enums)

**Categorical inputs are parsed to enums, then compared as members.**
The compile prompt requires `class Lodging(Enum)` and
`class GuestType(Enum)` with Hebrew values; `quote()` does
`unit = Lodging(lodging)` / `tab = GuestType(guest_type)` and branches
on `is`. String compares inside the function are out. AST allowlist
accepts Enum/StrEnum classes and `from enum import Enum`.
test_price_function_enums.py; design.md “Per-site price functions”.

### Done (2026-09-16, compile drops low-confidence extras)

**A tab is a guest_type only if its rows match catalog lodging.**
Compile calls `resolve_listing_ids(..., force=False)` and keeps a row
only when the listing match is exact or ≥ `UNCERTAIN_BELOW`. Rental
SKUs (השכרת מזרן, השכרת פלטה on ציוד להשכרה) no longer attach to חושה
and do not enter `GUEST_TYPES`. `list_prices` still force-matches
lodging labels. test_compile_match_confidence.py; design.md “Per-site
price functions”.

### Done (2026-09-16, compile uses canonical lodging + guest_type)

**Compile prompt is catalog names and a tab parameter.** Rate-card
labels go through the same `match_info_website_name` path as
`list_prices`; the user prompt’s `lodging` is the canonical
`info_website_names` string. `guest_type` is the rate-card tab (רגיל,
מנוי, …) and a `quote()` argument; the boolean discount flags are gone.
`quote()` must reject unknown lodging / guest_type against constant
tuples. Reports always get `<site_id>.py` plus `<site_id>.prompt.txt`.
Achziv 2.py had invented `"tent"`, `next()`, and flag-to-tab mapping.
Supersedes “dump failed quote()” (dumps on pass too) and the
“No `childs_num` / flags” bits of the group-flag entry. compile_price,
QuoteParams, gold, ast_check (`next` + a few str/dict methods);
design.md “Per-site price functions”.

### Done (2026-09-16, dump failed quote())

**A failing compile is still visible.** AST or gold failure prints the
model's `quote()` and writes `reports/price_functions/<site_id>.py`.
It is not stored on `site_price_functions`. scrape.py
`_dump_failed_quote`; design.md “Per-site price functions”.

### Done (2026-09-16, quote() takes child_num)

**`child_num` is an explicit quote input.** Occupancy, extra-person,
and קבוצה thresholds use `adults_num + child_num`; `child_ages` only
splits toddler / child / adult-rate. If omitted on a mapping, it
defaults to `len(child_ages)`. Supersedes the “No `childs_num`”
sentence in the group-flag entry below. QuoteParams, compile_price
prompt, gold `_params`; design.md “Per-site price functions”.

### Done (2026-09-16, group is not a quote flag)

**`is_group` dropped from `QuoteParams`.** Group rates are a per-site
occupancy policy (30 vs 80, …), not a guest identity like Matmon or
soldier. `quote()` deduces the קבוצה tab from `adults_num` plus
`child_ages` against that site's card. No `childs_num` either: the
ages tuple is the count and the toddler / child / adult split.
`is_weekend_or_holiday` stays an input. compile_price prompt;
design.md “Per-site price functions”. Handwritten tests still accept
`is_group=` as an unused default.

### Done (2026-09-15, gold price explanations)

**Gold cases now say how the expected price was built.** Each of the
five cases per site carries an `explanation` (late checkout, extra
person, toddler free, Matmon, included occupancy). Compile still
gates on the numeric price; the string is for humans and for the
failure line. design.md “Per-site price functions”.

### Done (2026-09-15, sandboxed per-site price functions)

**Prices are compiled to `quote()`, not summed by the recommender.**
`scrape-prices` still snapshots `list_prices`, then gathers every
rate-class tab plus `מידע למבקר` pricing rules, asks the 235B for one
`quote(...)` per site, AST-checks it (`math` only), and stores the
source on `site_price_functions` only when five gold cases pass.
Unchanged hash bumps `scraped_at` only; a fail keeps the previous
row. Planner quotes via the `price-sandbox` Docker (params in, price
plus explanation out) and falls back to `quote_night` when
`PRICE_SANDBOX_URL` is unset or the site has no passing function.
Supersedes the 2026-09-13 “executable numeric rules” note for
**prices** (other numeric subjects are still open). A compile-quality
experiment (18 × 1 call) is not in experiments.md yet — needs a
`scrape-prices` yes.

### Done (2026-09-14, scrape-reviews then embed)

**Daily reviews fetch then classify.** `just scrape-reviews` still
upserts Place Details, then immediately visit-gates / splits / embeds
rows with `is_relevant IS NULL` (already-classified rows stay put).
`--embed-only` skips Google (`just populate-claims` is the same
classify step alone). The Actions Summary has LLM cost and claims
written. Modules stay split. Supersedes “populate-claims stays
manual” in the daily scrape-reviews entry below. claims.md ingest;
design.md “Where it runs”.

### Done (2026-09-14, scrape workflows split)

**Availability and reviews are separate workflows.**
`scrape-availability.yml` (08:00 IDT) and `scrape-reviews.yml` (09:00
IDT) each call reusable `scrape-job.yml` for the SSH + Summary copy.
`scrape.yml` is dispatch-only for claims / info / sites / place-ids.
Supersedes the “same scrape.yml, two crons” bit of the daily
scrape-reviews entry below. design.md “Where it runs”; cloud.md.

### Done (2026-09-14, daily scrape-reviews)

**Daily reviews at 09:00 IDT** (`0 6 * * *` in
`.github/workflows/scrape.yml`; 08:00 IST in winter), an hour after
availability so `concurrency: scrape` does not queue them. Same SSH
compose-run path; the Actions Summary is `# scrape-reviews` (new rows,
already stored, skipped sites, Google errors). `populate-claims` stays
manual. design.md “Where it runs”; cloud.md.

### Done (2026-09-14, לשבוע הבא is kind=week)

**Bare "next week" is `kind=week`, not tonight.** Live extract of
`משהו לשבוע הבא במדבר… אחד שומר שבת` (no weekday) emitted
`kind=on, on=today, horizon_days=7`; `resolve_dates` ignored the
horizon on `on` and returned 14–15 Sep. `kind=week` + `when=next`
enumerates that ISO week (Mon–Sun); the 4-window cap keeps Friday and
Saturday. `kind=on` + `horizon_days` now enumerates consecutive nights.
`שומר שבת` stays semantic. Distinct from `לשבוע הבא בחמישי` (named
Thursday, still `kind=weekday`). experiments.md 2026-09-14 §2.
design.md "Query extractor: date_intent".

### Done (2026-09-14, desert glued to weekday)

**Extractor keeps desert when it sits next to a weekday.** Live miss on
`משהו לשבוע הבא בחמישי במדבר ל3 אנשים, חשוב לנו ניקיון. אחד שומר שבת`:
party, cleanliness, shomer shabbat, and next Thursday all landed;
`במדבר` did not. Region/vibe was already in the schema (`Negev` →
semantic, not `campsite`) but no few-shot showed a weekday glued to a
landscape. Added `לשבוע הבא בחמישי במדבר` → Thursday + `desert`, and
the rule that the glue is two constraints. 5/5 at temperature 0
(experiments.md 2026-09-14 §1). Test:
`test_extractor_thursday_desert.py`. design.md "Query extractor:
date_intent".

### Done (2026-09-14, pin production search_path)

**Production `connect()` sends `search_path=public,extensions`.** Migration
033 moved pgvector into `extensions` and `ALTER DATABASE SET search_path`,
but a `pg_restore` of a data dump does not replay that setting. Streamlit
on the VM then failed every amenity/rules/claims retrieve with pgvector's
`vector type not found in the database` (`register_vector` looks up
`vector` on the session path). Experiments already pinned the path;
production now does too. design.md "Experiments live in the `experiments`
schema".

### Open (2026-09-14, hosted ReAct + search MCP)

**Move the chat loop to a hosted ReAct agent (Claude Code, GPT
Codex); search / trip become an MCP the agent is instructed to
use.** Ingest, RAG, availability scrape, and `search.py` stay ours.
The MCP is those typed tools (`search_open_slots`, date resolve,
quote, claims/rules), not generic Postgres. The 2026-09-08 note
rejected hosted Claude Code *plus a SQL MCP* (no embedder, no
`quote_night`, one session ≠ N chats); this is a different cut.
`docs/react_vs_graph_agent.md` still applies for calendars: the
backbone must not own `בשישי הקרוב`. Streamlit / LangGraph remain
the live path until this is spiked. Supersedes the “closest fit is
ReAct on Qwen 235B” close of the 2026-09-08 ReAct vs graph entry.

### Done (2026-09-14, scraped_at vs updated_at on skip)

**Hash skip bumps `scraped_at` only.** `updated_at` is the last vacancy
(or offers-hash) change. Rewrite still sets both. Same split on
`booking_page_hashes` (`html_sha256` may still refresh on skip).

### Done (2026-09-14, drop availability.adults_no)

**`availability.adults_no` is gone** (and the same column on
`booking_page_hashes`). The scrape still GETs INPA as 1 adult; party
size for search is `max_occupancy`. Unique slot is site + dates + type.
Migration `039_drop_adults_no`.

### Done (2026-09-14, drop availability before today)

**Each `scrape-availability` deletes rows with `start_date` before
Israel today** (and matching `booking_page_hashes`). The rolling 14-night
window starts from `today_il()`; leftover nights from earlier runs would
otherwise stay in search. Count is on the Actions Summary (`dropped past`).
Supersedes nothing; sits on the scheduled-scrape entry below.

### Done (2026-09-14, scheduled availability scrape + hashes)

**Daily availability at 08:00 IDT** (`0 5 * * *` in
`.github/workflows/scrape.yml`; 07:00 IST in winter). GitHub-hosted
Actions SSHs in as `gh-actions` (not the operator’s login key). The
scraper stores `booking_page_hashes` (`html_sha256` of the raw INPA
page, `offers_sha256` of aggregated room counts) and skips the
availability rewrite plus unit-match LLM when offers are unchanged.
The change report is the Actions run Summary tab. Info/claims stay
`workflow_dispatch`. design.md “Where it runs”; cloud.md §6.

### Done (2026-09-14, drop in-memory query-embed cache)

**No `_query_vec_cache`.** Same phrase in one `_query_vec_literals`
call is still embedded once (`dict.fromkeys`); a later turn hits
Nebius again. The in-process dict was premature. A Postgres
phrase→vector table stays open (entry below). scaling.md §5.

### Open (2026-09-14, persistent query-embed cache)

**Planner query embeddings are only an in-process dict
(`_query_vec_cache` in `search.py`).** A first `embed_query` is ~4–5s
on Qwen3-Embedding-8B; repeats in the same Streamlit/agent process are
free; a restart pays Nebius again. Not avoiding that for now. Later:
a Postgres phrase→vector table (same 1536-d space as claims) so
identical amenity strings survive process restart. Catalog claim/amenity
rows are already stored; this is only the *query* side. scaling.md §5.

### Done (2026-09-14, embed_query tools on the LangSmith timeline)

**Each query embedding is a `embed_query` tool**, same wrench as
`claim_judge` / `resolve_dates`. Parallel workers copy the LangSmith
parent. Amenity retrieve SQL is also a tool. Recommender and extractor
were already ChatOpenAI spans. design.md "LangSmith".

### Done (2026-09-14, claims/rules SQL on LangSmith spans)

**`search_review_claims` and `search_campsite_rules` are traced.** Each
span's Inputs include the interpolated SQL (`<vector>` in place of the
embedding). Vacancy SQL is on `search_open_slots` the same way.
design.md "LangSmith".

### Done (2026-09-14, planner LangSmith splits SQL / embed / retrieve)

**Planner is not 7s of idle then judge.** `claim_judge` children sit
inside the planner bar and run in parallel (~max of the four, not the
sum). `@traceable` spans `search_open_slots`, `embed_queries`, and
`retrieve` so the remaining time is visible. design.md "LangSmith".

### Done (2026-09-14, drop empty planner queries dump)

**Streamlit no longer attaches a `queries: []` list to the planner
trace.** That array was hooked search-tool calls, not extractor amenity
strings, and was empty even when `fits` was not.

### Done (2026-09-14, Streamlit errors are generic)

**UI says `Something went wrong.`; the real exception goes to the
terminal and the log.** Reloads a stale `db.connect` so
`DatabaseUnavailable` imports after a hot reload. Postgres-down still
fails in 3s. design.md "Experiments live in the experiments schema".

### Done (2026-09-14, Postgres down fails in seconds)

**`db.connect` times out in 3s and Streamlit pings before a turn.** Docker
Desktop off used to hang planner SQL for minutes (`sql=260s`). The chat
shows `st.error` with `docker compose up -d`; `connect()` raises
`DatabaseUnavailable`. design.md "Experiments live in the experiments
schema".

### Done (2026-09-14, live turn stages in local Streamlit)

**Local chat prints each node/LLM/tool as it starts.** The assistant
caption and the `just streamlit` terminal show `turn +Ns extractor …`
instead of a silent Thinking spinner. Public UI is unchanged.

### Done (2026-09-14, local Streamlit on 8502)

**Laptop `just streamlit` binds 8502.** An SSH `-L 8501` to the VM
otherwise wins `localhost:8501` and the tab shows prod (`TRIPPY_PUBLIC_UI=1`)
instead of Last turn trace. design.md "Where it runs".

### Done (2026-09-14, claim_judge StructuredTool)

**Judge calls show up in LangSmith like `resolve_dates`.** Live
`apply_claim_rule_judgements` `.invoke`s `claim_judge_tool` per
(campsite, request); worker threads `copy_context()` so the tool run
nests under planner. Replaces the `RunTree.create_child` spans that
never attached. design.md "LangSmith".

### Done (2026-09-14, LangSmith claim/rule judge)

**Each judge call is a LangSmith child under the planner.** Inputs are
the claims and rules the 235B received; outputs mark each claim
`relevant` and the site `satisfies` / `satisfy_by` / `reason`. The
model does not emit a per-rule keep/drop. Thread-pool workers inherit
the parent run. design.md "LangSmith".

### Done (2026-09-14, Dockerfile CMD + CI image build)

**Prod image `CMD` is one JSON line.** A multiline exec-form array
parsed as a new instruction (`"/app/.venv/bin/streamlit"`). CI job
`image` runs `docker build -t trippy:ci .` on every PR / `main` so
that class of parse error fails in GitHub, not on the VM.

### Done (2026-09-14, drop Telegram / FastAPI)

**No Uvicorn in prod or laptop Compose.** Deleted `main.py`, `scripts/startup.sh`,
`scripts/replay.py`, and `test_webhook.py` (the live embedding search moved
to `test_embedding_search.py`). Laptop compose is Postgres only; Streamlit
is `just streamlit`. Dropped `fastapi`, `uvicorn`, `pytest-asyncio`, and
`TELEGRAM_TOKEN`. Supersedes “Telegram is unwired; token optional” in the
Nebius VM entry and the dummy-token CI note below.

### Done (2026-09-14, LangSmith on LangGraph)

**Tester turns go to LangSmith.** Streamlit (and Telegram, if wired)
enable tracing when `LANGSMITH_API_KEY` is set and stamp each graph
run with `thread_id` + `channel`. Public UI still hides the sidebar
trace; smith.langchain.com is how we see what the agent did.
Ingest `job.sh` forces tracing off so scrapes do not share the
project. design.md "LangSmith".

### Done (2026-09-14, warmup prints Kimi's hi)

**Warmup prints the ping and Kimi's reply.** On Streamlit load:
`recommender warmup ping=hi model=…` immediately, then
`recommender warmup reply='…' in Xs` when the one-token call
returns (or `warmup failed`).

### Done (2026-09-14, recommend timing reaches Streamlit)

**Last recommend timing is a process snapshot, not a ContextVar.**
LangGraph copies context into the node, so Streamlit’s
`last_recommend_timing()` was always empty and the TTFT/thinking
line never printed. The recommend call also `print`s that line
(flush) — Streamlit/Uvicorn hide `logger.info`. Restart Streamlit
to see warmup, extra_body, and `reasoning=` / `thinking_stream=`.

### Done (2026-09-13, recommend thinking logs)

**Recommend logs whether thinking is still on.** One info line
with model, extra_body, TTFT, in/out, `reasoning_tokens`, empty
prefix chunks, and `thinking_stream`. Warning if reasoning or
thinking text appeared. Streamlit caption + warning; eval dump
stores the flags.

### Later (2026-09-13, executable numeric rules)

**Today the recommender is doing the arithmetic, and it should not.**
Query path for amenities/rules is: retrieve nearest `campsite_rules` →
the claim/rule judge decides amenity yes/no (`satisfies` / relevant
claims) → Kimi picks and *explains*, including numeric and compound
rules it was never meant to compute. A sentence like “100 per person
on a weekday unless Matmon, plus a 12 surcharge, unless active reserve
duty, depending on leaving time” is one rule with branches. Flat
`<topic>_<scope>_<predicate>` statements plus a `qualifier` cannot
carry that, and asking an LLM to apply it at recommend time is the
wrong tool. Amenities stay a judge yes/no; this is for the numeric /
conditional shelf (price, times, capacity: “4 children or 2 adults”).

**Ingest idea — compile related statements into one Python function,
store that on the row.** Not started. Stages:

1. Gather official rules by subject (price, check-out, occupancy, …).
2. Formulate a single Python function for that subject at that site
   (or unit), e.g.
   `def payment(people_count, is_matmon, is_active_reserve_duty, leaving_time)`.
3. DB row still has the subject (or subjects) embedded like today, so
   retrieve stays vector search.
4. Judge still only says whether the subject is relevant to the ask.
5. Agent asks the user for the function’s parameters, or assumes
   documented defaults (`is_matmon=False`).
6. Compute deterministically in code.
7. Recommend from the numbers, not from the model doing the sum.

Scope is open: maybe only the few subjects people actually ask
(times, payments, capacity), not every numeric rule.

**Can Postgres compute this at query time?** Store the source, yes;
`exec` it inside SQL, no. Postgres has no safe “run this Python from
a TEXT column in a `SELECT`”. `plpythonu` is untrusted (OS user of
the server) and is the wrong place for LLM-written code. So:

- **Recommend path (after retrieve):** the DB holds the blob
  (source + signature + defaults). The agent loads the relevant rows,
  the judge filters, Python evals in-process (sandboxed / restricted
  builtins). Matches the flow above. Ranking among already-fetched
  fits does not need SQL eval.
- **Planner / SQL filter path** (“under 400 ₪ for 4 people with
  Matmon”, “can this tent take 2 adults + 4 children”) needs the
  formula to be *data*, not Python. Options: (a) a JSON AST /
  jsonlogic the SQL walks, (b) denormalize the common cases into
  columns the planner already has (`list_prices`, occupancy), (c)
  hybrid — Python function for the exact quote the recommender
  cites, columns or AST for the SQL gate. (a) is “can the DB do
  this”; (b) is what we already do for list rates; full Python in
  SQL is not.

Open vs the 2026-09-04 compound-rule note (atomic statements vs a
semantic tree): a function is a third shape. The tree/AST is the
one a `SELECT` can evaluate; Python is the one an LLM can write
from a messy parks.org.il paragraph. Decide which subjects need
query-time SQL before picking.

### Done (2026-09-12, scrape via SSH)

**GitHub-hosted Actions SSHs into the VM** and `docker compose run`
the scrape container there. No self-hosted runner. Secrets:
`TRIPPY_VM_HOST`, `TRIPPY_SSH_USER`, `TRIPPY_SSH_KEY`. Supersedes the
self-hosted runner in the Nebius VM entry below.

### Done (2026-09-12, Nebius VM)

**Phase-1 cloud is one always-on Nebius CPU VM**, self-hosted
Postgres, Streamlit behind a Cloudflare Tunnel, ingest as one-shot
`trippy:prod` containers (`scripts/cloud/job.sh`), dumps to disk plus
Object Storage. `TELEGRAM_TOKEN` is optional; `main.py` no longer
raises at import. No VM start/stop Actions yet. Runbook:
`docs/cloud.md`. Supersedes “keep Streamlit off the internet” in
`scaling.md` for this phase, and the dummy `TELEGRAM_TOKEN` note in
the 2026-09 CI entry.

### Done (2026-09-12, Streamlit Kimi warmup)

**Streamlit pings Kimi with a one-token `hi` on first load**
(background thread, once per process) so the first recommend is
not a cold replica.

### Done (2026-09-12, eval --limit)

**`--limit N` keeps the first N cases per difficulty.** `--limit 2`
is 2 easy + 2 hard (E01, E02, H01, H02 on planner_v1). Works with
`--from-planner` and `--ids`.

### Done (2026-09-12, recommend TTFT in eval)

**Eval prints recommend TTFT** (first SSE chunk, first paintable
stay) next to `recommend=`. Dump fields `ttft_chunk_ms` /
`ttft_spoken_ms`. Streamlit already had these.

### Done (2026-09-12, intro is a comparison)

**Two-pick `intro` notes there are multiple stays, names them,
and compares them.** Phrasing is free — “there are X options
here” was an example, not a template. Prompt only.

### Done (2026-09-12, Kimi default + intro)

**Recommender default is Kimi-K3.** Prompt: listing and reviews
that agree on existence are said once (no “there are tents and
guests say there are tents”). Two picks set `intro` (“יש כאן שתי
אפשרויות…”) above the numbered list. Super and 235B stay aliases
(experiments.md 2026-09-12 §2).

### Done (2026-09-12, Kimi recommender pick)

**Kimi-K3 wins the why bake** on phrasing, not Latin.
235B/397B invent or garble Hebrew; GLM is close but coins
`משוערפים`. Listing+review doubles are a prompt fix. Super
still the running default until the test/alias flip
(experiments.md 2026-09-12 §2).

### Done (2026-09-12, recommender phrasing bake)

**Same packs, four models.** Kimi-K3 then GLM-5.2 write natural
Hebrew why (0 Latin, 0 query-echo). 397B glues English. 235B
leaks `outlets` and garbles H12. Super stays default
(experiments.md 2026-09-12 §1).

### Done (2026-09-12, recommender pack cache)

**Eval dumps a compact recommender `pack` per case** (query,
extract, compact fits). `--recommender --from-planner
reports/evals/<stamp>.json` replays only the picker. Old dumps
have no pack; this pass writes one. Prompt also stops echoing
the query and coining Hebrew (`מרחביים` / `שמדליות`).

### Done (2026-09-12, recommender why shape)

**Recommender `why` leads with the match, then related
listing-vs-review notes.** Contradictions and listing-silent
concrete amenities cite review recency; vibes can be review-only.
Quality caveats about the asked thing come after; unrelated
complaints stay out. Prompt only (`RECOMMENDER_SYSTEM_PROMPT`).

### Done (2026-09-12)

**Planner query embeddings run 5 at a time.** Distinct semantic
query statements in one retrieve (`_query_vec_literals`) hit Nebius
in parallel, capped at `QUERY_EMBED_CONCURRENCY = 5`. One query
stays a single call. Supersedes “one query embedding” in the
planner retrieve description: a stay can have several statements.

### Done (2026-09-11, Super recommender)

**Recommender is Nemotron Super 120B-A12B, thinking off.** Hebrew
`why` had 0 Latin leaks on the five long recs vs 235B
`pitch`/`outlets`. Not faster, 1.5× $. Extractor/judge stay 235B
(experiments.md 2026-09-11 §9–§10).

### Done (2026-09-11, Nemotron recommender probe)

**Five long-why recs, same planner pack:** Lightning is English
on all five and often picks two stays. Super writes clean
Hebrew (no `pitch`/`outlets`) but is not cheaper and invented
`טוקול` on E06. Stay 235B (experiments.md 2026-09-11 §9).

### Done (2026-09-11, GLM batch judge full eval)

**`planner_v1` with one GLM-5.2 judge call per case, thinking
off:** 22/27, same as 235B singles. Judge 31s×18 vs 71s×189.
E10 recovered, E05 Be'erot dropped. Stay per-job 235B
(experiments.md 2026-09-11 §8).

### Done (2026-09-11, batched judge thinking off)

**Same 2000-token batch with thinking disabled.** All
`reasoning_tokens=0`. 397B and GLM **78/80** (E03 tent 20/20);
235B still 74/80 (16/20 tent). GLM wall **8.4s**. Stay per-job
235B until a full eval (experiments.md 2026-09-11 §7).

### Done (2026-09-11, batched judge max_tokens=2000)

**Same four-model batch as §5 with completion capped at 2000.**
235B still 74/80 (`stop`). 397B 0/80 and GLM 10/80 (`length`,
no JSON). DeepSeek 73/80, still `stop`. Cap does not make
thinking models batch-safe (experiments.md 2026-09-11 §6).

### Done (2026-09-11, batched judge × four models)

**One compact batch call per case on E03/H02/H07/E04/H10.**
Vs stored one-by-one 235B: 397B 77/80 (E03 tent **20/20**),
current 235B 74/80 (E03 still 16/20 tent_pitch), DeepSeek
68/80, GLM 60/80 (H07 prose, hit 8k). Stay on per-job 235B
(experiments.md 2026-09-11 §5).

### Done (2026-09-11, full graph without Streamlit)

**Five live E15 turns via `build_graph`, no AppTest:** mean 18.6s
(8.9–44.6). First turn 44.6s (recommend 33s); later 10–17s. The
47s live AppTest was a slow Nebius turn, not Streamlit
(experiments.md 2026-09-11 §4).

### Done (2026-09-11, light_node frozen vs live)

**Five interleaved `light_node` calls on E15:** frozen mean 0.69s
(0.35–1.81), live mean 0.73s (0.31–1.37). All KEEP. Occupancy pins
do not explain Streamlit's 12s light (experiments.md 2026-09-11 §3).

### Done (2026-09-11, live Streamlit judge counts)

**Same three queries on `public.availability`:** 48s / 43s / 46s,
judge **10 / 20 / 10** (same N as frozen; slower 235B waves). Sea
finished. First 300s timeout was not extra jobs (experiments.md
2026-09-11 §2).

### Done (2026-09-11, Streamlit matches eval on frozen occupancy)

**E15/E03/E04 through Streamlit on `availability_frozen` were 16s /
17s / 12s** (judge 10 / 20 / 10 calls). Live occupancy had been 67s /
77s / 300s timeout. Trace now logs `collect_stages()` and
`judge_calls` (experiments.md 2026-09-11 §1).

### Done (2026-09-11, recommend stream survives bad JSON escapes)

**`parse_partial_json` crashed eval on E04** (`Invalid \escape`) when
the 235B wrote `\pitch` inside `why`. Stream parse drops illegal
backslashes and retries; the final payload does the same
(`source/agent/recommender.py`).

### Done (2026-09-11, E15 couple on 17 Sep)

**`planner_v1` gained E15** (`קמפינג לזוג ב-17 בספטמבר 2026 ללילה אחד`).
Same capacity ask as E02, vacant night 17 Sep (Masada + Yotvata
couple tents). The other two Streamlit trial queries were already
E03 and E04. Set is 27 (15 easy / 12 hard).

### Done (2026-09-10, booking URL on fits)

**Each planner fit gets a `BE_Results.aspx` booking URL** (hotel id,
dates, party size) after vacancies + judge. Subcamps inherit the
parent `booking_hotel_id`. The recommender copies `booking_url` from
the fit; the spoken reply prints the looked-up URL, not a
model-invented one (`source/agent/booking.py`).

### Done (2026-09-10, recommender Hebrew leaks)

**Recommender still leaked Latin/CJK after the one-language prompt**
(`ゲuests`, `.pitch`, `בungalו` on `2026-09-10_195239`). Cause:
packed evidence is English and the prompt said “say guests report”.
Prompt now paraphrases claims, uses `evidence_span` not subject
keys, and writes אורחים מספרים in Hebrew (experiments.md
2026-09-10 §8).

### Done (2026-09-10, stream recommend)

**Recommender streams tokens.** `recommend_from_payload` uses
`ChatOpenAI.stream` with `stream_usage=True` (usage still matches
invoke; experiments.md 2026-09-04 §1). Streamlit paints the spoken
reply as JSON fields become parseable, not the raw JSON. Telegram
still waits for the finished node.

### Done (2026-09-10, recommender one language)

**Recommender `why` / `empty` stay in one language:** mostly-Hebrew
query → Hebrew only; mostly-English → English only. Prompt change after
the first dump mixed Latin/CJK into Hebrew (experiments.md 2026-09-10
§7).

### Done (2026-09-10, recommender rewrite)

**Rewrote `recommender_node` as a 235B JSON picker**
(`source/agent/recommender.py`). Packs the original query, extractor
JSON, and compact fits (`why`, relevant `review_claims` including nos,
retrieved rules unsifted, `claim_judge`). Picks 1 or 2 stays, validates
against `fits`, renders Hebrew. Judge unchanged. `just run-eval --
--recommender` on `planner_v1` dumped recs
(`reports/evals/2026-09-10_185136`; 15 one-rec, 11 empty, 0 two-rec;
experiments.md 2026-09-10 §7). Supersedes the first-draft prose node
that dumped HumanMessages + planner ChatMessages.

### Done (2026-09-10, scrape cadence on the MVP path)

**Scheduled scrapes are part of cloudify** (`docs/plan-to-mvp.md` §2).
Vacancy daily, reviews+claims weekly, sites/info/place-ids about
monthly.

### Done (2026-09-10, plan-to-mvp)

**Forward path is `docs/plan-to-mvp.md`.** Sequence: recommender →
cloudify → give people to try → final fixes → Medium + LinkedIn.
This file stays the historical log.

### Done (2026-09-10, compact judge ×5 is the default)

**Live claim-judge default is compact JSON and 5 parallel calls.**
Streamlit, Telegram, and `just run-eval` all read
`judge_compact()` / `judge_concurrency()` (`TRIPPY_JUDGE_COMPACT`
default on, `TRIPPY_JUDGE_CONCURRENCY` default 5). Opt out with
`TRIPPY_JUDGE_COMPACT=0` / `--no-judge-compact`. Supersedes the
“default off” / sequential note in the compact-flag entry below.
Eval `2026-09-10_131406` already ran this way (23/26).

### Done (2026-09-10, English listing match)

**`scrape-sites` crawls `en.parks.org.il/camping/` titles and the 235B
matches Hebrew names to that closed list** (one call). Subcamps append
North/South. Replaces inventing English. Yehudia has no English listing
card, so it stays unmatched.

### Done (2026-09-10, named-site pg_trgm)

**Removed `_NAMED_CAMPSITE_ALIASES` and `ILIKE`.** H03 failed because
the extractor emitted `Achziv` and lookup only aliased Horshat
spellings to Hebrew. `campsites.english_name` is filled on
`scrape-sites` by one 235B call. Lookup ranks `name` /
`english_name` with pg_trgm (`similarity` + `word_similarity`). Rule:
ask before string lists / regex / `"str" in "str2"`
(`.cursor/rules/ask-before-string-vocabularies.mdc`). English names
are empty until the next `just scrape-sites` (or a one-off fill).

### Open (2026-09-10, H07 caravan bays)

**Judge returns true on caravan parking bays for a tent stay.** Eval
`2026-09-10_131406` (compact, ×5): 23/26; H07 still fails
`must_exclude_types` עמדת/עמדות חניה לקרוואן פרטי (sites 1, 13, 15).
Leave it; fix later.

Cause: extractor splits `בלי קרוואן` into a second site-locus ask
`not caravan`. Retrieve pins `electric_hookup` on the bay itself.
The judge scores `(campsite_id, query)`, not the unit, so tent /
`tent_pitch` rules from other types at the same site make
`tent with electricity` true (`tent and hookup provided`). `not
caravan` is treated as a grant, not a type filter (Tel Arad: `tent
lodging provided`; Besor: `area:south implies not caravan`; Horshat:
`no caravan rule needed`). Prompt already says a caravan-bay hookup
does not satisfy electricity for a tent stay — it never fires because
the site also has tents.

### Done (2026-09-10, `_SlotKey` NamedTuple)

**Vacant-unit dict keys are `_SlotKey(campsite_id, accommodation_type_id)`**,
not `tuple[str, int]`. Rule now: TypeAlias / NamedTuple / dataclass —
anonymous tuples including lookup keys
(`.cursor/rules/named-records-not-tuples.mdc`).

### Done (2026-09-10, `_SemanticWhy` dataclass)

**`_semantic_why_by_slot` returns a named record**, not a 4-tuple of
dicts. Rule: same-typed tuple members → dataclass
(`.cursor/rules/named-records-not-tuples.mdc`).

### Done (2026-09-10, judge no longer retrieves)

**Planner embeds the semantic query once and fetches claims, amenities,
and official rules with that vector.** The judge only scores the
payload (`campsite_rules` + `review_claims` + the query). Duplicate
query strings reuse the cached vector.

### Done (2026-09-10, E02 compact vs quoted tokens)

**E02 235B sequential: compact judge out 376 vs quoted 824** (10
calls). Judge wall 13.4s vs 20.4s. Prompt *up* ~3.5k from the
compact suffix. Both PASS. experiments.md 2026-09-10 §6.

### Done (2026-09-10, eval token counts)

**Eval prints token in/out after each case and on the report.** Extractor
and claim-judge usage land on one `LlmUsage` sink. Markdown gets a
Tokens table (in / out / extract / judge).

### Done (2026-09-10, compact judge output flag)

**`TRIPPY_JUDGE_COMPACT` / `--judge-compact`, default off.** Compact
asks the judge for claim indices (`relevant`) and a 4–5 word
`reason` instead of quoting claim text. The planner keeps the
retrieved claims in memory and maps those indices back before the
recommender. Production still quotes.

### Done (2026-09-10, 30B judge on E03/H02/H07)

**30B one-by-one vs stored 235B: 59/60 satisfies.** The miss is a
truncated JSON, not a semantic flip. Sequential wall ~same as 235B
(~1s/call). $0.013 vs $0.026. experiments.md 2026-09-10 §5.

### Done (2026-09-10, judge batch size 5 and 10)

**Chunks of 5 or 10 vs one-by-one: 17/20, 20/20, 19/20 (size 5) and
17/20, 20/20, 18/20 (size 10).** E03 tent/`tent_pitch` still misses.
Not wired in. experiments.md 2026-09-10 §4.

### Done (2026-09-10, batched judge probe)

**One 235B call for all (site, query) jobs vs one-by-one.** E03/H02/H07
(20 jobs each). Satisfies 16/20, 20/20, 20/20. Batch faster; E03 tent
missed tent_pitch. Not wired into the planner.
experiments.md 2026-09-10 §3.

### Done (2026-09-10, judge prompt: region is not the sea)

**Breadcrumb `area:*`/`region:*` satisfy that region by name, not
`near the sea` via the word ים/sea.** 235B probe 8/8: Masada Dead Sea
no longer grants sea; Akhziv beach, north, Negev→desert, and named
Dead Sea still grant. experiments.md 2026-09-10 §2.

### Done (2026-09-10, Dead Sea breadcrumb judge probe)

**Hebrew label + English slug still grants `near the sea` for Masada.**
Six 235B judge calls, no embed/DB. Akhziv breadcrumbs (`גליל מערבי`)
do not grant sea. experiments.md 2026-09-10 §1.

### Done (2026-09-10, eval model override and parallel judges)

**`just run-eval -- --model 30B` and `--judge-concurrency 4`.**
Extractor+judge read `TRIPPY_INSTRUCT_MODEL`; live claim judges can
run 4 at a time. Default remains 235B sequential.

### Done (2026-09-09, eval party phrasing varied)

**Party size phrasing in `planner_v1` is mixed:** אדם אחד, זוג /
שני מבוגרים, שלושה חברים / 3 מבוגרים, 4 חברים, ארבעה זוגות
(8), משפחה של 6. Occupancy-sensitive gold unchanged (couple tent,
hut occ=4 vs family of 6, E03 per-person price).

### Done (2026-09-09, eval queries all state party size)

**Every `planner_v1` query names a party.** E03 is `אדם אחד` plus
`עד 80 שקל לאדם` — party is one person; לאדם stays a per-person
price unit, not party_size. Gold `party_size` is scored even on
no-date skips (E14, H11).

### Done (2026-09-09, eval report traces extract/RAG/judge)

**Each eval case dumps extractor constraints, planner queries,
retrieved claims/rules, and the judge verdict** in
`reports/evals/*.md` under Cases. Fits also keep `retrieved` next to
`claim_judge`.

### Done (2026-09-09, eval report includes per-query seconds)

**`reports/evals/*.md` table has an `s` column** for extractor+planner
wall time per case.

### Done (2026-09-09, run-eval exits 0 on case fails)

**`just run-eval` exits 0 after writing the report even when cases
FAIL.** Missing frozen table, unknown `--ids`, and runtime errors
still exit non-zero.

### Done (2026-09-09, run-eval copies public first)

**`just run-eval` copies `public` → `experiments` before scoring,
except `availability`.** Occupancy stays `availability_frozen`.
`--no-copy` skips the refresh.

### Done (2026-09-09, breadcrumb claim labels)

**Breadcrumb claims are `area:north` and `region:upper-galilee`, not
bare slugs.** Same ingest, prefixed so a north query hits the area
row and an Upper Galilee query hits the region row.

### Done (2026-09-09, breadcrumb region claims)

**`just scrape-info` embeds parks.org.il `#breadcrumbs` as claims.**
Trail `צפון` / `גליל עליון` → `area-north` / `upper-galilee`, straight
embed (the review splitter emitted nothing). `notes` is `no review,
region by breadcrumbs`; `review_id` is NULL. Quick ingest; may change.
experiments.md 2026-09-09 §1.

### Open (2026-09-09, satellite features)

**Later: take a satellite image of the area, pass it through an image
model, and ask what features it can see.** Deduce "by the sea" / "in
the desert" when the site page does not say so.

### Done (2026-09-08, just run-eval)

**`just run-eval` scores extractor + planner against
`evals/planner_v1.json`.** Frozen occupancy, pinned today, markdown
report under `reports/evals/`. `--ids E01,H02` for a subset.

### Done (2026-09-08, planner benchmark v1)

**26 queries (14 easy / 12 hard) with website gold and frozen
occupancy.** `experiments.availability_frozen` is a snapshot of
`public.availability` (7–19 Sep 2026). Amenity/rule gold is the
parks.org.il page. `evals/planner_v1.json`. experiments.md
2026-09-08 §11.

### Done (2026-09-08, replay fridge query 3)

**Akhziv tents now fit `עם מקרר`.** Same query as §7: 2 fits
(37 north, 38 south). Official `refrigerator` is in `why`; the
judge grants by rule and keeps fridge complaints as caveats.
experiments.md 2026-09-08 §10.

### Done (2026-09-08, judge yes is OR; nos do not veto)

**Rules scan is this site + parent, never a sister.** Judge always
sees rules and claims. `satisfies` is any granting rule or claim;
complaints stay in `relevant_claims` for the recommender.
experiments.md 2026-09-08 §9.

### Done (2026-09-08, claim and site amenity both retrieve)

**Site-locus retrieve always runs claims and site amenities;
satisfaction is OR.** A claim hit no longer skips the listing.
`search_campsite_rules` for a subcamp includes the child's rows
and the parent's, so visitor-info on Akhziv 37/38 reaches the
judge. experiments.md 2026-09-08 §8.

### Done (2026-09-08, query 3 with fridge)

**Fridge instead of electricity: still 0 fits.** Akhziv tents pass
sea (beach claim) and die on fridge: retrieve was kitchen-fridge
*complaints*, so the official `מקררים (3)` never reached the judge.
Huts with mini-fridges were not vacant 17 Sep. experiments.md
2026-09-08 §7.

### Done (2026-09-08, Akhziv scrape-info + query 3)

**Visitor-info `sea` puts Akhziv tents through the sea gate; they
still fail electricity.** Full `scrape-info --site 2` on
`experiments`, then planner query 3: 0 fits / 19 rejected. North
and south tent pitches miss only electricity (`phone_charging_points`
is not a hookup). Inland powered sites still fail the sea judge.
experiments.md 2026-09-08 §6. No planner change.

### Open (2026-09-08, ReAct vs graph agent)

**Chatbot loop is undecided; ingest/RAG/availability stay.** Logged
tradeoffs in `docs/react_vs_graph_agent.md`: stuffing pages after
slots rejected as a RAG replacement (lost-in-the-middle, no unit
amenities or claims); hosted Claude Code + generic Postgres MCP
rejected as a Telegram runtime (no embedder, no `quote_night`, session
≠ N chats). Closest fit is ReAct on Qwen 235B with `search.py` as
typed tools. Buried `בשישי הקרוב` is why the backbone must not own
calendars: 30B full was 2/5 on the Horshat Tal query, 235B extract
30/30 (experiments.md 2026-09-08 §2–§3). No spike yet. Follow-ups
(“why?”, “the first one”, “cheaper”) need `last_recommendations` —
still the open item in §4 below.

### Done (2026-09-08, experiments harness)

**`TRIPPY_SCHEMA=experiments` points scrapes and search at a full
copy of `public`.** `db.connect.connect` is the one `psycopg.connect`
wrapper; `scripts/setup_experiments.py copy` rebuilds the schema
(DDL + rows + views), `--empty` truncates named tables after. `just
on-experiments scrape-info -- --site 2`. No custom clone scripts.
Supersedes ad-hoc `clone_tables` + seed in experiment scripts.

### Done (2026-09-08, visitor-info extract re-run)

**`cant_` rewrites to `can_` (polarity false); the 18:00 late-fee
cutoff is its own subject.** `to_positive_subject` rewrites token
`cant_` / `cannot_` instead of dropping them
(`cant_be_without_muzzle` → `can_be_without_muzzle` false). The merge
judge (235B) has a few-shot that 18:00 is not `check_in_end_time`.
Live Akhziv re-extract in `experiments`: 36 stored, 28/29 gold
(the visiting-hours pointer is the only miss, as asked), 0 naming
drops, `$0.009`, 70 s. Report:
`reports/visitor_info_ingest/2026-09-08_154130.md`. experiments.md
2026-09-08 §5. Supersedes the reservation-drop and 18:00-merge notes
in the visitor-info accordion entry below.

### Done (2026-09-08, visitor-info prompt follow-ups)

**Sea/desert/forest are amenities; infix `_without_` no longer drops;
visiting-hours pointers are non-statements.** Site extract now keep-and-
generalises a natural setting the way the unit prompt already did, so
`בקרבת הים` is `sea` + `near_water`, not brochure. `_without_` in the
middle of a name (`entry_without_reservation_allowed`) is kept; prefix
`without_electricity` is still rewritten. "You may only enter during
visiting hours" is emit-nothing, like a pointer at another section.
experiments.md 2026-09-08 §4. No new category: retrieve already searches
amenities.

### Done (2026-09-08, visitor-info accordion)

**`מידע למבקר` is fetched and extracted like `מה בחניון?`.** The tab
is AJAX (`data-cnt` per site, same loadmore endpoint as lodging). One
section, one extract call. Akhziv experiment: 46 rows in
`experiments.campsite_rules`, 26/29 gold lines, $0.014, 193 s.
Reservation-required was dropped by the positive-phrasing guard;
18:00 late-fee cutoff merged into `check_in_end_time`.
experiments.md 2026-09-08 §4. Report:
`reports/visitor_info_ingest/2026-09-08_142435.md`.

### Done (2026-09-08, just pr merges on green CI)

**`just pr` squash-merges with `--admin` after CI passes, then checks
out main and pulls.** A solo owner cannot approve their own PR; Protect
main still requires a review, so the merge uses the admin bypass. Failed
CI still prints the job logs and leaves you on the feature branch.
Supersedes “just pr stays on the feature branch”.

### Done (2026-09-08, extractor on 235B)

**Query extractor is the 235B; `planner_model` renamed
`extractor_model`.** Buried הקרוב was 2/5 on the 30B full prompt, 5/5
on 235B (experiments.md 2026-09-08 §2–§3). p50 4.0s → 2.4s, ~$0.00025
→ ~$0.00050 per extract. `planner_node` was never that client — it is
SQL. Supersedes “query-constraint extract stays on 30B”.

### Done (2026-09-08, date-intent consistency)

**30B full extractor is 27/30 on date_intent; 235B full and 30B
dates-only are 30/30.** The only miss is buried `בשישי הקרוב` (Horshat
Tal query): 2/5 `when=this`, 3/5 `when=next`. Bare הקרוב is 5/5 on the
30B. Not shipped — still one 30B extract call. experiments.md
2026-09-08 §2.

### Done (2026-09-08, caravan-bay hookup names)

**Caravan-bay חיבור חשמל ומים is `caravan_bay_electric_hookup` /
`caravan_bay_water_hookup`, not bare `electric_hookup`.** Retrieve
cannot see the unit type on the row, so the generic names looked like a
guest socket. Unit prompt: one exception to never-name-the-unit, plus
that listing as a few-shot. PITCH tent power stays `electric_hookup`.
experiments.md 2026-09-08 §1. Stored rows unchanged until re-extract.

### Done (2026-09-07, amenity gate −0.7 in prod)

**Amenity retrieve −0.7; the judge sifts listing hits.** Same design as
the §8 probe: outlets/hookups enter, tent-as-desert is retrieve noise
the judge drops. `AMENITY_MATCH_MAX_DISTANCE = -0.7`. Amenity-only fits
no longer skip the judge. experiments.md 2026-09-07 §8. Supersedes
“prod stays −0.8 until listing-only fits go through the judge” above.

### Done (2026-09-07, judge sifts amenity −0.7)

**Judge can sift amenity −0.7 listing hits; despite-split did not
unglue Mamshit.** After populate-claims: 59/60 vs gold (1 gold-string
miss on Horshat Tal night-noise ban). Desert tents dropped; electricity
outlets / PITCH / site points kept; Besor caravan-bay dropped. Mamshit
despite claim still glued at −0.577 for `"desert"`. Prod amenity gate
stays −0.8 until the planner sends listing-only fits through the judge.
experiments.md 2026-09-07 §8. Supersedes the “judge cannot veto a
stated amenity” reason in the −0.7 note below — it can, if it sees
them; it currently does not.

### Done (2026-09-07, amenity gate −0.7)

**Do not loosen amenity −0.8 → −0.7.** `"electricity"` extras are real
(`electric_outlet` −0.750, `electric_hookup` −0.704, top 5) **and**
`electric_stove`. The same −0.7 makes `tent` (−0.719) satisfy desert and
quiet — 25–32 types — and the judge cannot veto a stated amenity.
`electric_hookup` is caravan bay **and** PITCH tents **and** site-wide
נקודות חשמל, not trailer-only. Gate stays −0.8. experiments.md
2026-09-07 §7.

### Done (2026-09-07, retrieve K=5/10/20)

**Labeled gold is rank 1 at −0.6; K=10/20 add no gold.** Same 35 packs,
nearest 20 claims and 20 rules. Claim gold −0.615…−0.767; `dogs_allowed`
−0.897; `electric_hookup` −0.704. Desert has no location rule in 20 —
tents at −0.719 fill the list and sit closer than electricity. Keep
claim top-5. Rule gate should not be −0.6 (flood) or −0.8 (drops
hookup). experiments.md 2026-09-07 §6.

### Done (2026-09-07, claim-rule judge)

**One 235B call does relevant + satisfies, with campsite rules in view.**
Same 35 packs as the polarity probe, top 5 claims at −0.6 plus nearest 5
rules (all categories). Combined JSON: 35/35 satisfies and 35/35
relevant-exact. Pet-forbidden stays relevant and does not satisfy.
Gate −0.6; planner filters on `satisfies`, recommender gets the relevant
claims. Did not split into two calls. experiments.md 2026-09-07 §5.

### Done (2026-09-07, claim-judge polarity)

**Judge polarity few-shots fix pet-friendly.** Same −0.6 retrieve as §3;
prompt now requires `is_positive` to match the request. Horshat Tal /
Ashkelon “pets not allowed” → no. Desert/Mamshit/electricity still yes.
35/35. Gate −0.6 kept. Not in the recommender until you say so.
experiments.md 2026-09-07 §4.

### Done (2026-09-07, claim-verify judge probe)

**235B can filter a −0.6 claim gate for location.** All claims with
`<#> ≤ −0.6` per site, one judge call, “most may be irrelevant”: desert /
in-the-desert 22/22 vs gold (Masada + Mamshit despite; Khan Be'erot and
Hai-Bar animals no). Pet-friendly: 2 false yes on “pets not allowed”.
Not wired; polarity must be in the prompt first. experiments.md
2026-09-07 §3.

### Done (2026-09-07, despite-aside split)

**Claim splitter splits concessive asides.** "Despite X, Y" / `בכל זאת מדבר
ורוחות` was one cleanliness claim, so `"desert"` RAG missed Mamshit (−0.578
vs gate −0.7). One few-shot: 15/15 on the 235B (English, Hebrew span, full
Mamshit review × 5). Stored claims unchanged until a claims rebuild.
experiments.md 2026-09-07 §2.

### Done (2026-09-07, claims_with_reviews name)

**`claims_with_reviews` includes the campsite name.** The view still
exposes `campsite_id`; it now joins `campsites` so a row is readable
without a second lookup. Migration `035_claims_campsite_name`.

### Done (2026-09-07, date_intent few-shots)

**Extractor date_intent few-shots for הקרוב / הבא / בעוד N שבועות.** The 30B
already emitted intent and `resolve_dates` did the calendar; it labelled
`שישי הקרוב` as `when=next` and dropped `weeks_from_now` on `בעוד שבועיים`.
Three shots + `weeks_from_now: N` in the schema: 25/25 on the 30B (5 prompts ×
5 trials, today frozen to Mon 7 Sep). Stayed on 30B; no second date-only call.
experiments.md 2026-09-07 §1. `test_extractor_date_intent_karov.py`.

### Done (2026-09-07, just pr)

**`just pr` stays on the feature branch.** Title defaults to the branch name
(hyphens to spaces); a `just pr "Title"` argument still overrides. After create
it waits for GitHub checks and prints pass or the failed job logs. No checkout
of main.

### Done (2026-09-07, empty reviews)

**Empty review text is `is_relevant = false`.** scrape-reviews upsert sets it on
insert/update; populate-claims bulk-marks leftover empty unclassified rows
before the visit gate. They are not sent to the 30B.

### Done (2026-09-07, claims per site)

**populate-claims commits after each campsite.** A restart skips
`is_relevant IS NOT NULL` (already gated). Failed site rolls back; finished
sites stay. Supersedes the same-day split entry's one-transaction-for-the-table.

### Done (2026-09-07, claims split)

**Google fetch no longer classifies.** `just scrape-reviews` upserts the 5+5
Place Details rows into `reviews` and stops. `just populate-claims` visit-gates
and splits rows with `is_relevant IS NULL`. `just clear-claims` deletes `claims`
and nulls that flag only, so review text is kept and Google is not re-fetched
to rebuild claims. Migration `034`. Supersedes the same-day entry that still
ran gate/split inside scrape-reviews. claims.md ingest.

### Done (2026-09-07, reviews)

**Reviews scrape always fetches newest then most_relevant.** Two Place Details
calls per site (cap 5 each), concatenated newest-first, overlap dropped, then
the same visit-gate / split / embed path. `--most-relevant` is a no-op.
Supersedes the weekly-newest / seed-relevant split. claims.md ingest.

### Done (2026-09-07)

**Availability type match lowercases the 30B view.** Ma'ayan Harod's lodging
panel stores `מתחם PITCH`; the booking engine offers `מתחם pitch`. Exact SQL
misses (Postgres is case-sensitive) and the 30B pick must copy a candidate
exactly, so it returned null and 17 pitches were skipped. The name and the
candidate list are now lowercased for that call; the original catalog casing
is restored before the alias is stored. design.md "The rate-card listing
match runs on the 235B".

### Done (2026-09-06, night)

**Two rate lines can no longer collapse onto one product.** `colliding_rows`
groups resolved rows by `list_prices_unique_rate` itself, so a clash is found
before anything is written and without a model -- which is what catches the case
confidence cannot: Tel Arad matched both Canaanite structures to one listing at
1.00 and 0.80 and silently lost the 860 row. A colliding pair gets one
`pick_pair` call that must return two different candidates; anything else leaves
the rows alone and reports. Full run: 95 rows from 93 rate lines, nothing lost
at any site, one collision call at $0.0001. experiments.md §23.

`snapshot_list_prices` now resolves every row before writing any, since a clash
is only visible once both halves exist.

### Done (2026-09-06, late)

**A doubted listing match gets a second pass that may name several products.**
One rate line can price two rooms (`חדרים 5 ו-6`), and a single pick has to be
wrong about one of them. Below `UNCERTAIN_BELOW`, or on a refusal,
`pick_names` runs under `MULTI_MATCH_PROMPT` and the price is filed against each
product named; `match_verdict` gains `"split"` so the report shows a rate that
wrote more rows than the card has lines. Fired twice in a full run for $0.0003,
both correct; Khan Be'erot 8 rows -> 13, all right. experiments.md §22.

**Still open:** two rate lines can collapse onto one listing *confidently* --
Tel Arad's single and double Canaanite structures -- and the second silently
overwrites the first on `list_prices_unique_rate`. Detectable in code without an
LLM; not built.

### Done (2026-09-06, evening)

**Listing match → 235B; brackets stripped from the name.** The 30B answered one
Khan Be'erot label wrongly 6 times in 7 at temperature 0, always the same pick
at 0.40, and correctly 6 times in 6 on the same bytes with the brackets swapped
— so its answer turned on bracket direction. The 235B: 12/12 across both forms
at 1.00. Removing the brackets is measured inert on the 235B, so
`strip_brackets` now runs on the name (never the candidates). The availability
type matcher keeps the 30B, pinned explicitly — different prompt, unmeasured.
experiments.md §20, design.md "The rate-card listing match runs on the 235B".

Supersedes §18's reading, which blamed the prompt's room-number clause for this
run's failures; §19 established the numbers were never sent to the model at all.
That is fixed too: the model is now shown the raw rate-card label
(`full_label`), while the free exact test still runs on the normalised type. The
report's prompt appendix prints the user message only.

Verified on a full run (`reports/prices_235b.log`): flagged matches 12 -> 2,
forced refusals 3 -> 0, rows kept 87 -> 91 of 93, run cost $0.0099 -> $0.0145.
Both remaining flags are `(חדרים 5 ו-6)`, one label naming two products --
experiments.md §21.

### Done (2026-09-06, later)

**Tests own the `experiments` schema; extensions move out of `public`.**

Three pre-existing test files wrote to production tables and were moved to
clones in `experiments` (`.cursor/rules/no-test-data-in-prod.mdc`). The clones
are `CREATE TABLE experiments.x (LIKE public.x INCLUDING ALL)`, so the checks
and indexes under test are production's own and stay in step with every
migration. Three things `LIKE` does not carry across, each of which had to be
replayed or fixed: foreign keys (and `pg_get_constraintdef` qualifies them as
`public.` when `public` is off the search_path, which pointed every replayed key
back at production), a `serial`'s sequence (the copied default drew ids from
`public.campsites_id_seq`), and index *names* (`list_prices_unique_rate` arrived
auto-named, so `ON CONFLICT ON CONSTRAINT` failed only here).

`vector` and `pg_trgm` moved to a new `extensions` schema (migration 033).
An extension is installed once per database, into one schema; with pgvector in
`public` the `%(embedding)s::vector` cast in `subjects/resolve.py` could not
resolve under `search_path=experiments`. Production now runs
`"$user", public, extensions` and the tests `experiments, extensions` -- so the
type is reachable from both and `public` is on neither test path. The rejected
alternative was appending `public` to the test path, which is precisely what the
rule exists to prevent: a table that had not been cloned would resolve to the
real one.

Anything holding an open connection -- the `api` container especially -- must
reconnect before an unqualified `vector` resolves again.

**The prices run report carries its prompts.**

`UNCERTAIN 0.60: 'x' -> 'y'` shows the answer and not the question, so a wrong
match could not be attributed to the prompt or to the model. `InfoWebsiteNameMatcher`
now keeps a `MatchCall` per call -- system, user, raw reply, pick, confidence --
and `run_prices` ends with the full prompt for every flagged one. `match_verdict`
derives "forced" / "uncertain" from the record rather than being told, so
`snapshot_list_prices` is unchanged. See `docs/experiments.md` §18 for what the
first read of it found.

### Done (2026-09-06)

**Only the owner merges and fires paid Actions.** Ruleset `Protect main` now requires one approving review, code-owner review (`CODEOWNERS`: `* @simcoster`), re-approval after the last push, squash-only, and green `lint` + `test`. Repository admins may bypass on a PR (a solo owner cannot approve their own PR) but cannot push to `main`. Fork PRs wait for the owner to approve workflows (`all_external_contributors`). Manual LLM tests use the `llm` environment, reviewer `@simcoster`. Completes the “protect `main` so lint and test are required” note in the CI entry below.

**The `אפשרויות לינה` panel is fetched, parsed and ingested — in the testbed only.** Accommodation types are to come from the info page rather than from booking-engine unit names, because the panel lists what *exists* (48 bungalows, 45 חושה, 7 caravan bays) while the booking engine shows only what is free on the nights scanned, and carries no counts. That the booking engine is "structured" bought nothing: its `.tt-desc` is the same Hebrew paragraph the panel's `<p>` is, so the LLM pass over prose was unavoidable either way (experiments.md 2026-09-06 §1). New `rules_ingest/lodging.py` productionises `temp/subcamp_detect_probe.py`'s AJAX fetch — `panel_request` reads `body[data-id]`, the panel's own `data-cnt` (never assumed; the order differs per site) and the inline nonce, folding away the zero-width space Tel Ashkelon writes inside the panel title. `parse_lodging_blocks` gives the panel as indexed blocks; `LodgingSegmenterLLMClient` labels them and normalises names but never rewrites text, so spans stay quotable; `assemble_units` enforces **never merge listings** over whatever the model returns — one `<h4>` is one row, and a name collision sends both headings back to their verbatim text, which is what keeps Metsada's two accessible rooms (7 sleepers and 5) apart. `<h3>` is captured as `scope`: Akhziv's `חניון צפוני` / `חניון דרומי` route units to campsites 37 and 38 by exact match on the operator's own heading, which is the reconciliation design.md:634-641 wanted instead of `unit_name_contains` substring routing. `unit_section` puts the unit name in the *text*, not only the title — Yehudia's whole panel is one `<h4>`, and a name is a description. Two passes over each unit, not chained: pass 1 (`ExtractorLLMClient`, narrowed to beds/occupancy/room_count) and pass 2 (`ingest_unit_rules`) both read the paragraph as written, with the exclusion stated in `unit_prompt` rather than subtracted from the text — subtracting made 48% of evidence spans quote text that was never on the page (experiments.md §2). `check_in_time`, `check_out_time` and `policy_rules` are gone from `AccommodationExtract` and from the `UPDATE`; they have no readers anywhere and become rules. `miscategorised_rule()` clears the category of a `boolean_rule` that coins a predicate, reported in both run reports (experiments.md §3).

**Where this stopped.** Everything above is testbed-only: it writes `experiments.lodging_*` and touches no production table (verified in-run). **Not yet written:** the migrations (`032` dropping the three columns, `033` adding `aliases TEXT[]`, `unit_count` and `campsites.ingest_notes`), the `db/models.py` changes, the lodging pass inside `ingest_site`, `scrape-availability` matching-instead-of-creating with the alias list, the `scrape-all` reordering, and `docs/design.md`. `test_amenity_extraction.py` still exists and its two tests have not moved. The plan is `~/.claude/plans/i-want-to-do-compressed-knuth.md`.

**Known before production.** (1) `ingest_unit_rules` builds a `RuleExtractorLLMClient` per unit because `unit_prompt` interpolates the unit name; each one builds its own transport, and `ssl_context()` costs **5.2 s per call** on a machine with `TLS_TRUST_OS_STORE` set because it reloads the OS certificate store — six minutes of setup over 68 units, which is why the testbed took ~25 min against `scrape-rules`' 17 for twice the rules. Mitigated by passing a shared `openai_client`; the real fix is to stop interpolating the name, since the user message already carries it twice, making `unit_prompt` a constant and one client enough. (2) Yehudia fails: its panel is a single block, and asked to list paragraph indices when there are none the segmenter emits a runaway `[1, 2, 3, … 78 …]` and blows past the JSON. Skip the segmentation call when a panel has no `para` blocks — there is nothing to attribute. (3) One wrong column in 67: Khān Be'erot's `חדרי צוות חדרים 3-4` got `room_count=3` from `3 מיטות קומותיים`; it is one room. (4) 8 of 421 spans are the 235B part-translating a word (`מיקרוגל` → `מיקרוwave`); a substring check at write time would catch every one without a model call.

**Unit amenities go through the rules pipeline, evidence span and all.** `scrape-availability` had its own amenity path: `ExtractorLLMClient` returned `amenities` / `not_included` as bare strings and `write_unit_amenities` wrote `(campsite_id, accommodation_type_id, subject_id, polarity)` — every per-unit row had a NULL `evidence_span`, `source_url`, `confidence` and `qualifier`, so a site rule could be checked against its sentence and a unit rule could not. The merge judge was also called without `states=` or `campsite_id=`, deciding sameness with neither side's assertion in view, and a collision was settled by letting the later write win rather than being filed. New `rules_ingest/units.py`: the tooltip becomes a `Section(title=type_name, source_url=booking_url)` and goes through `rules_from_sections` → `_resolve_statements` → `upsert_campsite_rules(accommodation_type_id=...)`, the same path a page section takes. `unit_prompt` prefixes the production prompt (in front, for the reason `subcamp_prompt` gives) and countermands its "this section describes the CAMPSITE AS A WHOLE" line. Per-unit collisions now reach `resolve_page_conflicts` and `conflict_cases`, and the run writes the same report `scrape-rules` does, titled `scrape-availability`. `ExtractorLLMClient` keeps the type's own columns (beds, occupancy, `room_count`, times, `policy_rules`) and no longer extracts amenities at all; its cost role is renamed `amenity_extract` → `unit_details_extract`. Named-place expansion moved into `unit_prompt`, and its three tests moved from `test_amenity_extraction.py` to the new `test_unit_rules.py`. `ensure_amenities`, `write_unit_amenities` and the never-called `PlaceEnrichmentLLMClient` are now dead — left in place, not yet removed. The LLM half is unverified: `pytest -m llm source/test/test_cases/test_unit_rules.py` is the check, and no scrape was run.

**`just scrape-prices` fixed: the entry point runs as a module.** The recipe ran `python source/scraper/info_site/scrape.py`, so `sys.path[0]` was `source/scraper/info_site` and it shadowed the repo-root `db` package with `info_site/db.py`; `amenity_enrichment/db.py`'s `from db.models import SubjectCategory` then reached `info_site/db.py` and died on its relative `from .schemas import ClassifiedPriceRow`. Adding `info_site/db.py` is what armed it — the justfile's `PYTHONPATH` was never the problem, a script's own directory always comes first. `scrape.py` now imports through `source.scraper.*` with no `sys.path` insert (so no `# noqa: E402`), `info_site/classify.py`, `info_site/match_listing.py` and `populate_availability.py` import `amenity_enrichment` / `info_site` the same way, and the recipe is `uv run python -m source.scraper.info_site.scrape --prices`. `PYTHONPATH` is unchanged: the tests and the other scrapers still import bare. Supersedes design.md's "`rules_ingest/fetch.py` re-implements `fetch_page_html` because scrape.py cannot be imported by module path" — it can now; the two copies are still not hoisted.

**pytest results on the GitHub run.** CI writes `--junitxml=reports/pytest.xml` (gitignored) and `dorny/test-reporter@v3.0.0` publishes a Check named `pytest` plus a job summary. The test job needs `checks: write`. Same for the manual LLM workflow (`pytest (llm)`). Not a native Tests tab — GitHub does not have one.

**CI does not need a Telegram secret.** `test_webhook.py` imports `main` at collection, and `main.py` raises if `TELEGRAM_TOKEN` is unset. The test job (and the manual LLM workflow) set a dummy `TELEGRAM_TOKEN`; nothing calls Telegram.

**setup-uv pinned to `v10.0.1`.** The first GitHub Actions run failed at "Set up job": `astral-sh/setup-uv` publishes immutable tags (`v10.0.0`, `v10.0.1`) and no floating `v10`. Both workflows now use `astral-sh/setup-uv@v10.0.1`.

**Telegram webhook tests skipped.** Four tests in `test_webhook.py` invoked the live LangGraph graph (`graph.invoke` → `light_model`) and still passed because `telegram_webhook` swallows errors. Telegram is not live; Streamlit is the current client. Marked `@pytest.mark.skip` as deprecated until rewritten. Empty-update cases and the `llm` embedding test are unchanged.

**Scraper OpenAI clients come from one factory.** `make_nebius_openai_client` is the only `OpenAI()` in scraper/ingest code. It calls `ensure_live_llm()` and then constructs; pytest turns that off except `@pytest.mark.llm` (`LiveLlmDisabled`). Wrapper classes (`ConflictResolverLLMClient`, extractors, embeddings, …) request the factory rather than constructing. LangGraph `ChatOpenAI` is a separate path and is not gated. Supersedes the deferred-proxy note in the CI entry below.

**Live LLM is gated at the factory.** `make_nebius_openai_client` and `make_agent_chat_model` go through one `ensure_live_llm()` check (`amenity_enrichment/llm.py`). Production allows it; pytest disables it and turns it back on only for `@pytest.mark.llm` (`source/test/conftest.py`). First use of a client built at import (`graph.py` chat models, `search.py` embedder) raises `LiveLlmDisabled` instead of calling Nebius. Previously unmarked live extractor tests now carry the mark. `test_live_llm_gate.py`.

**CI on every PR.** GitHub Actions runs two required-to-be-required checks on PRs and on pushes to `main`: `lint` (`uv run ruff check .`) and `test` (`pytest -m "not llm"` against a fresh `pgvector/pgvector:pg16`, `alembic upgrade head`, then one seeded campsite + subject). A third workflow, "LLM tests (manual)", is `workflow_dispatch` only so Nebius never fires on a PR (`no-unasked-scrape-runs.mdc` in workflow form). `.github/workflows/ci.yml`, `.github/workflows/llm-tests.yml`, `scripts/ci_seed.py`.

Lint is green by config, not by rewriting Alembic: `per-file-ignores` for `E402` on the dotenv bootstraps (`main.py`, `scripts/streamlit_chat.py`), `I001` on the `sys.path.insert` test (untouched; `existing-tests-permission.mdc`) and on `alembic/**` (the local `alembic/` directory makes `import alembic` first-party, so every generated revision would fail I001). Five import blocks were re-sorted (`db/__init__.py`, `scripts/replay.py`, `source/agent/constraints.py`, `info_site/classify.py`, `info_site/match_listing.py`). No `ruff format` gate.

The test job ignores files that cannot pass on an empty CI database: accommodation amenities/RAG (scraped embeddings) and `test_planner_multi_room.py` (unimplemented; docstring says it currently fails). Live extractor tests are marked `llm` and excluded by `-m "not llm"`; the factory is the backstop if a new test forgets the mark.

Manual step after the first green run: protect `main` so `lint` and `test` are required, and add the `NEBIUS_API_KEY` (and optionally `GOOGLE_API_KEY`) repository secret for the manual workflow.

Supersedes the plan's two-PR split: this branch (`add-ci`) was already clean off `main`, and the lint job cannot pass until the ignores land, so both go together.

### Done (2026-09-05)

**Rate-card notes parked.** The `הערות למחירון` section is per-rate by construction (each tooltip is prefixed with its rate label) and the extractor dropped the label every time, so the 10:13 run stored `adult_min_age 14` / `child_min_age 5` / `child_max_age 14` (the private-tent price bands), `weekend_min_nights 2` (the air-conditioned bungalow) and, on Yehiam, `mattresses 4` (a family-tent rental) as facts about the campsite; the judge then merged the hut's `weekend_hut_min_nights` into the bungalow's and `mattresses_included` into `mattresses`. `ingest.sections_to_extract` now drops `PARKED_SECTION_TITLES` after parsing (the parser and its tests are unchanged) and the log prints what was parked. `test_rate_notes_parked.py`. Consequence to accept: `test_rules_extraction.py`'s `night` and `age` cases read the fixture through `parse_sections` directly, so they still see the notes and still pass, but they now assert on facts the ingest no longer stores.

**Referent field (next).** Every statement should say WHAT it is about, not only what it states: `applies_to` with a closed kind — `campsite`, `unit` (a rate or accommodation type), `guest_type` (adult / child bands), `membership` (מנויי מטמון) — plus the label as written on the page. Then: (1) the ingest routes `unit` statements to `campsite_rules.accommodation_type_id` when the label matches an accommodation type (the subcamp module already matches unit names) and skips them at site level otherwise, which un-parks the rate notes; (2) the judge is shown both referents and told that different referents never merge, whatever the names — the hut/bungalow and tent/site merges above had matching names and different referents; (3) `guest_type` bands become `<band>_rate_min_age`-style pricing facts rather than admission rules. The cases are pinned as strict xfails in `test_referent_scope_llm.py` (four extractor, two judge) so the suite flips when the field lands.

**First run through the new judge and explainer** (10:13, three pages, $0.039): 36 judge calls, 9 merges, no cross-kind merge; the accessibility rule held (`accessible_huts` for the two huts); the two collisions were the counted `accessible_toilets` against the accessibility clause's uncounted one — the same fact stated twice, which the upsert still labels CONFLICTING because one side has a number and the other none. Explainer: 2 calls, $0.0005, picked the kept side correctly, called the cause `extractor_wrong_value` where "duplicate, one side richer" would be right. Open: a same-polarity collision where only one side carries a number should be labelled a duplicate, not CONFLICTING.

**Resolver confidence is not a gate** (experiments.md §17): asked for a confidence, the resolver said 0.95 on 16 of 16 right actions and on 8 of 10 wrong ones; the wrong answers are confident misreadings, not borderline calls, so unlike the judge (§12) nothing can be gated on it. Field kept and printed; the fix is worked examples and a name-shape check.

**Conflict resolver probed** (experiments.md §16; `rules_ingest/resolve_conflicts.py`, proposal only, nothing applied). Actions: `drop_new`, `rename_new` (override the merge), `rename_old` (≤ 3 citing rows, enforced in code), `enrich_kept`, plus `reassign_kept` added on reading the test set — the kept row is the misfiled one (23:00 "leave by" stored as `check_out_time`; urinals as `accessible_toilets`). On the 14 distinct collisions of the last three reports, twice each: 19/26 actions right after fixing our own validation; causes right 25/26; the model picks `rename_new` 22 times and never `reassign_kept` or `rename_old`; 6 of its 22 names break the naming shape. Next if adopted: worked examples for `reassign_kept` and the naming shape in its prompt, a shape check before applying, and the apply step itself (un-alias + insert, row move, rename with ≤ 3 rows, row update). `test_rules_conflict_resolver.py`.

**Conflict resolver adopted, narrowed to one action.** Migration `031` adds `conflict_cases`; every collision is diagnosed (the explainer's role, now inside the resolver) and filed with both sides, the verdict and `status = open`. The one automatic action is `rename_new` — undo a wrong merge by releasing the alias and giving the new statement its own subject (or joining an existing subject of that name); the old subject and kept row are never touched. `none` files the case for review. Runs after each page's commit in its own transaction; role `conflict_resolver` in the cost log; the terminal summary and run report print the resolution and case id. design.md "Conflicts are filed, and one action is automatic". Tests: `test_rules_conflict_resolver.py` (fake cursor, SQL by SQL), `test_conflict_cases_db.py` (live table, rolled back). `explain.py` stays as the shared mechanics prompt. Not yet run live.

### Done (2026-09-04, later)

**Three categories adopted.** The split from experiments.md §7 is in production code: `SubjectCategory` is `AMENITY 1 / BOOLEAN_RULE 2 / NUMERIC_RULE 3` (`RULE` kept as an alias of 2), migration `030` widens the CHECK, adds the `category = 3` partial HNSW index and teaches `campsite_rules_with_names` the two names. The extractor prompt tags each statement `amenity` / `boolean_rule` / `numeric_rule` (allowed + required → boolean; every other predicate → numeric), the payload coercers accept the labels, a bare `rule` now means "search every shelf", and the classifier prompt knows 1 | 2 | 3. Existing category-2 rows are not reclassified — rebuild with `scripts/clear_rules.py --subjects` + `just scrape-rules`. design.md "Category: three shelves".

**Run report file.** `scrape-rules` now writes `reports/rules_ingest/<start time>.md` after the run (`rules_ingest/report.py`, `RULES_REPORT_DIR` overrides): per page, every merge with both sides' original sentences, new subjects with their sentences, upsert collisions with both phrasings, resolver drops, and the cost table by role and model. `ResolutionTrace` gained `context` so a merge can show the term's own sentence. `test_rules_run_report.py`.

**Numeric ranges.** The extractor prompt now says a numeric range is a min and a max of one topic (`30-80 לנים` → `_min_occupancy 30` + `_max_occupancy 80`; `מעל 80` → only a min), with the group-booking sentences as worked examples, and `min_occupancy` joins the predicate list. `test_rules_category_split.py` pins the coercers and prompt without tokens, and (marked `llm`) checks a 20-50 range becomes two numeric statements and a permission-plus-deadline sentence lands on two shelves.

**Smoke run of the ported code** (experiments.md §8, isolated schema, production untouched): the two target merges stay gone and the range example yields min + max at the extractor — but the judge then merged `family_and_friends_group_min_occupancy` into `group_min_occupancy` (30 lost, 2 of 3 runs) and `late_check_out_on_saturday_evening_allowed` into `late_check_out_allowed`. Same-shelf narrowing is the judge's remaining failure mode; the split cannot reach it.

**Later — compound rules need a better way to split.** Accepted for now: the judge folding `late_check_out_on_saturday_evening_allowed` into `late_check_out_allowed` (−0.917; the embedding cannot tell a scoped variant from its parent, and the judge says yes about one time in three). The real problem is upstream: a sentence like "late fees on Saturdays for sites 1 and 4 but not 6 are 50% unless you hold a voucher" is one rule with a topic, several scopes, an exception and a number, and flat `<topic>_<scope>_<predicate>` names cannot carry that without either fragmenting into many near-identical subjects or collapsing them. Options when it matters: an iterative extraction that splits a compound sentence into atomic (topic, scope, predicate, value) statements before naming, or a semantic tree — topic node, scope children, values at the leaves — with `campsite_rules` rows pointing at leaves. Neither is started.

**Existing tests updated for three categories** (with permission): `test_rules_db.py::test_statement_category_is_coerced` (bare `rule` → None; `boolean_rule` / `numeric_rule` / 3 added), `test_rules_extraction.py` (`check_in` / `check_out` / `night` / `age` → `NUMERIC_RULE`, `dog` and `barbecue_allowed` → `BOOLEAN_RULE`) and `test_subject_adjudication_llm.py::test_classify_assigns_the_right_category` (`min_weekend_nights`, `check_out_time` → 3). The two `llm` files were not run. No-token suite: 6 failures remain, all pre-existing and dependent on availability data not loaded locally (accommodation amenities, RAG, planner multi-room, weekend extractor).

**Two more probes, both clean** (experiments.md §10, §11; production untouched). Judge: showing each side what it *states* (30 vs 80, 16:00 vs 17:00) flips all three of the 19:55 run's bad merges to null, 8/9 with the values alone and 9/9 with one added prompt sentence, while all 12 control calls on true merges with differing counts still merge. Today's presentation merged the bad pairs 9/9 — systematic, not flaky. Extractor: an accessibility bullet with a worked example turns `הונגשו: חניה, שירותים, מקלחות` into `accessible_*` subjects on both Akhziv clauses and leaves the neighbouring sections unchanged. Neither adopted yet. Also open: a merge is a single judge answer that the alias table makes permanent for every later page — a wrong one on page 1 costs the fact on all of them.

**Confidence from both models** (experiments.md §12, §13). Judge: asked for a confidence next to its answer, every right answer came back 0.95 and every wrong merge 0.30–0.85 — a ≥ 0.9 gate on matches would have caught all 11 wrong merges in 63 calls and lost none of the 52 right answers. Adding the field also moved the borderline decisions (the Friday time and the caravan hookup swapped sides), so it is a prompt change, not a free observation. Extractor: `confidence` is 1.0 on all 96 statements including the two-huts misread; carries nothing.

**Accessibility rule in the extractor prompt (adopted).** `הונגשו X, Y, Z` → `accessible_x` … as amenities, with a worked example; held on two re-runs with the neighbouring sections unchanged (experiments.md §11, §13). `test_rules_extraction_accessibility.py`. The `שתי חושות` misread (huts → fountain / senses) is untouched; a glossary line is the proposed cure.

**Judge proposal now on 87 calls** (experiments.md §14): values on both sides + one rule sentence + confidence, match accepted at ≥ 0.9. On the 24 real merges of the 19:55 run: 20 of 21 true merges kept (lost: `picnic_table` → `picnic_tables_and_benches` at 0.8), all 3 bad merges rejected — one of them only by the gate (`electric_hookup` matched at 0.3). Not adopted yet.

**Conflict explainer probe** (experiments.md §15, not implemented): the 235B, shown both sides of each collision, diagnosed all 8 judge-side collisions correctly (including a unit-specific `mattresses` line we had missed) and 1 of 8 extractor-side ones; it does not know the resolver mechanics or the naming shape, and proposed `toilets_count`. Worth wiring in only with those in its prompt.

**Judge changes adopted.** `pick_match` now shows each side a `states:` line (the term's polarity/number; a candidate's existing rows from `campsite_rules`, the current page marked "same page" — `SubjectStore.rules_table`, `format_states`, `campsite_id` threaded from `_ingest_scope` to `resolve_subject`), the prompt gained the one-page-two-numbers sentence and asks for `confidence`, and a match is accepted only at ≥ 0.9 (`MATCH_MIN_CONFIDENCE`); a refused match is traced as `ADJUDICATOR said … at confidence 0.30 < 0.9: rejected`. A reply without confidence is accepted as before. design.md "Predicates are the judge's call". `test_subject_judge_confidence.py`.

**Glossary line adopted.** חושה = hut in the extractor prompt; `שתי חושות` in an accessibility list → `accessible_huts`. `llm` test added.

**Conflict explainer wired in.** `rules_ingest/explain.py`: after each page, one 235B call per upsert collision (role `conflict_explainer`) with the pipeline mechanics, naming shape, accessibility rule and חושה in its prompt; the diagnosis prints under the collision in the terminal summary and the run report. Advisory only. `test_rules_conflict_explainer.py`. Not yet run live — the next `just scrape-rules` is the first real pass.

### Done (2026-09-04, evening)

**Live two-site run reviewed.** `just scrape-rules` on Hurshat Tal + Akhziv (64 subjects, 145 rules, $0.028). Two over-merges, each one judge call that then propagated for free through the alias table: `late_check_out_end_time` → `late_check_out_allowed` (17:00 lost on all three campsite rows) and `early_arrival_fee_required` → `early_check_in_fee_percent` (site 19/20 hold a percent subject with polarity and no number; the 50% dropped). Both pairs are worked "null" examples in the judge prompt. Also: the `נגישות` section re-emits `toilets` / `showers` as bare amenities and collides with the counted rows, and once invented a fountain count from `שתי חושות`; the extractor summed shower/toilet stalls on two sites and took the first number on the third; פלטות read as `mats`.

**Experiment: three categories (isolated, not adopted yet).** `rule` split into `boolean_rule` (predicates `allowed` / `required`, answered by polarity) and `numeric_rule` (every other predicate, answered by a number), tagged by the extractor, candidates restricted to the same category. Run against a cloned schema via `search_path`, production untouched. Both target merges gone for the predicted reason (the boolean twin was never a candidate); judge calls 32 → 25; Akhziv collisions 9 → 5. New: one same-category wrong merge (`family_and_friends_group_stay_min_occupancy` into `group_stay_min_occupancy`, the pair production kept apart) and one extractor mislabel (`hot_water_in_showers` as `boolean_rule`). experiments.md §7. Decision open.

### Done (2026-09-04)

**Local environment moved off Docker Desktop.** The Mac had Docker Desktop 2.2 (March 2020, x86/hyperkit) which cannot run on Apple Silicon; replaced with Colima + brew `docker` / `docker compose` / `buildx`. `.env` had been named `env` — unignored, so every `load_dotenv()` found nothing and the secrets were committable; renamed. ngrok dropped everywhere (Dockerfile, `startup.sh`, `pyproject`, `main.py`): its apt repo does not verify on Debian trixie and nothing read `NGROK_URL`. `OPENAI_API_KEY` removed from compose; the `openai` SDK stays — it is how Nebius is reached. `/app/.venv` in the `api` container is a **named volume**, so `--build` never updates deps: `down` → `docker volume rm trippy_trippy_venv` → `up --build`; never `down -v` (takes `pgdata` with it).

**TLS.** Ten modules each built their own `ssl.create_default_context()`; on a python.org framework Python that trusts nothing until `Install Certificates.command` runs. One `source/scraper/tls.py` now: certifi by default, `TLS_TRUST_OS_STORE=1` in `.env` opts the other PC (TLS-inspecting proxy) into the OS store + relaxed strict checks.

**`just` standardized on `scrape-*`.** `populate-reviews` → `scrape-reviews`, `ingest-rules` → `scrape-rules`. `branch` and `pr` were PowerShell-only (`powershell.exe: command not found` on the Mac); each is now defined twice with just's `[windows]` / `[unix]` attributes, the Unix side driving a new `scripts/open_pr.sh` that mirrors `open_pr.ps1` step for step — ad hoc, since the Mac is not the main dev machine. Entries below use the old names and were accurate when written. Four old mentions in this log were find-and-replaced before the append-only rule was stated — revert pending a decision.

**Rules ingest: visible and explained.** Extraction streams (`stream=True`, `include_usage` verified against Nebius: prompt/completion counts match a non-stream call exactly) with dots on the `extract:` line, per-section timing and tokens, per-site elapsed. A per-site report lists every subject the page touched, which term reached it by which path (`alias` / `merged` / `existing` / `inserted`), and every upsert collision with both phrasings and the resolver's reasoning. It surfaced `gas_stove_in_field_kitchen` merged into `field_kitchen` (the "no gas" fact dropped as CONFLICTING) and `late_check_out_*` fragmented into nine subjects from one section.

**Predicate gate removed; the judge decides.** `naming.same_predicate` / `PREDICATE_SUFFIXES` compared trailing tokens against a fixed tuple; any suffix not in it (`until`, `applies`, `percent`) read as "bare noun", so `_end_time` / `_available_until` never met the judge while `fee_percent` / `fee_applies` were judged one predicate and a boolean was merged into a numeric (site 20 holds `fee_percent = True`). Removed. The distinction lives in `ADJUDICATE_SYSTEM_PROMPT` with the synonym groups that *do* merge, plus "identical contexts are not evidence of sameness" — a reproduced, rolled-back run showed the judge merging the field-kitchen pair on exactly that signal, with the counter-example already in its prompt. Under the new prompt that pair is rejected and `late_check_out_available_until` merges into `late_check_out_end_time` (both reproduced). When the judge rejects every neighbour, `classify(near=…)` is asked for a canonical name distinct from them, with worked examples. `_resolve_positive` checks aliases before a classifier canonical can become a new subject — `late_check_out_fee_applies` had been both #38's alias and #64's name, leaving #64 unreachable. `opposed()` kept for now. **Supersedes the `_percent is redundant` note in "Next — sequenced" below: `PREDICATE_SUFFIXES` no longer exists.**

**Extractor canonical shape.** `<topic>[_<scope>]_<predicate>` for rules with a closed predicate vocabulary (`allowed required time fee_ils fee_percent min_age max_age min_nights max_nights max_occupancy count`; synonyms mapped onto it), `<thing>[_in_<place>]` bare nouns for amenities, scope between topic and predicate and phrased identically when it recurs, temporal hedges ignored. Verified live: `מטבח שדה (1) בשלב הזה בלי גז` → `field_kitchen` (true, 1 count) + `gas_in_field_kitchen` (false). The check-in window example now reads `check_in_start_time` / `check_in_end_time`; `test_rules_extraction.py` looks for `arrival` / `latest` — one needle to revisit.

**Repo rules (git-tracked; `CLAUDE.md` imports `.cursor/rules/`).** No semantic decision by comparing strings to a constant list — the LLM calls are for that; trivial comparisons and genuinely closed enumerations excepted. PLAN.md is an append-only log (newest-first entries here); design.md is the living document. A design choice made after an experiment is written to design.md *with its reason* and the experiment itself to `docs/experiments.md` (question, setup, numbers, cost, decision); today's five are there.

**Judge vs. direction pairs — measured.** In a `judge_experiment` schema cloned from production (own id sequence, `search_path`, `opposed()` monkeypatched off), 20 antonym/direction pairs with Hebrew contexts stating the direction. Reading the schema's alias arrays afterwards (creation order = id order) rather than only each pair's B term: **6 lost facts in 40 terms** — 5 opposite-direction merges (`child_max_age`→`child_min_age`, `mattress_pickup_end`→`start`, `gate_close`→`open`, `campfire_end`→`start`, `latest_check_in_time`→`last_entry_time`) plus `car_entry_time` merged into `check_in_time` (different noun); `earliest_check_in_time` and `arrival_time` also went into `check_in_time` (defensible). Antonym pairs sit at −0.84…−0.95, nearer than most true synonyms; 2 pairs split only because the distance fell outside −0.75. Controls 4/5 (`dogs_allowed` / `pets_allowed` stayed apart — fine: a missed merge is the tolerable failure, a wrong merge is not). 69 chat calls, ≈$0.017. **`classify(near=…)` renamed 5 of 40 terms**: reordered `weekend_min/max_nights` → `min/max_weekend_nights`, dropped a word (`stay_min_nights` → `min_nights`), and invented a direction — `dogs_entry_time` ("from 16:00", a start) became `last_dogs_entry_time`. A wrong name on a real fact is as bad as a wrong merge; rename-on-insert should be disabled or restricted to place/audience facets until direction is a column.

**Decided.** `opposed()` stays as the one sanctioned exception to the no-string-lists rule: it rejects a candidate before the judge sees it and can only over-split, the tolerable failure. Prompt vs. model, 13 cases × 4 cells: 30B/current prompt **4/6** wrong direction merges; 30B + a direction/actor block **1/6**; 235B **0/6** on either prompt, 0 missed merges everywhere; 235B is 2× per token (~$0.32 vs $0.16 per 1000 judge calls, which scale with vocabulary growth, not ingest volume) → **judge moved to the 235B** and the block added to the prompt anyway. The classifier, which had moved with it (shared constant), came back to the 30B: `classify("dogs_allowed")` on the 235B was "amenity" 9/10 on one run and "rule" 4/4 minutes earlier — unstable at temperature 0 — where the 30B was 20/20; separate `MODEL` / `CLASSIFY_MODEL` now (experiments.md §6). Its category is only consulted when a caller passes none, and both production callers do pass one. **No renaming:** the extractor's term is the canonical name, stored with the probe embedding; `classify()` is asked only for a category the extractor left out; `near` removed; the "classifier canonical already exists → alias" path is gone by construction. **Alias overflow:** after each site, every subject with more than 20 aliases prints as `ALIAS OVERFLOW {json}`. `test_rules_extraction` gained the `check_in_end` needle. **Cost per scrape run:** `LlmUsage` now keeps a bucket per (role, model) — `rules_extract`, `merge_judge`, `classify_amenity_or_rule`, `embed`, `amenity_extract`, `place_enrich`, `rate_card_classify`, `listing_match`, `review_visit_gate`, `claim_split` — priced from the rate table per model; before this `cost_usd` charged every call at the 235B extractor's rate. `summary()` prints one row per role under the old headline, and each `scrape-*` CLI appends one JSON line (`kind`, totals, `by_role`) to `reports/scrape_costs.jsonl` (`SCRAPE_COST_LOG` overrides; git-ignored). Recording happens in `main()`, with the run's `LlmUsage` threaded into `run()` / `run_prices()` / `populate_google_reviews()`, so tests that drive those functions with mocks write nothing.

**Open.** (1) Direction as a column — deferred; `opposed()` covers it for now. (0) TODO rename `Adjudicator` → `MergeJudge` across all files, keeping each occurrence's letter case (`SubjectAdjudicatorLLMClient` → `SubjectMergeJudgeLLMClient`, `ADJUDICATE_SYSTEM_PROMPT` → `MERGE_JUDGE_SYSTEM_PROMPT`, `adjudicator` → `merge_judge`, …) — own branch and PR. New repo rule: no incidental reformatting — a change touches only the lines it needs; indentation, formatting and pre-existing lint fixes go on a separate branch and PR. (2) All subjects and rules to be wiped and re-populated under the new prompts; no data repair. (3) For when data is too big to re-populate: store raw extractor statements per page / section / prompt-hash so a vocabulary change becomes a re-resolution job into a shadow schema, diffed with the per-site report, then swapped; vocabulary surgery (merge X into Y) as Alembic data migrations. (4) The judge's classify→existing-name path still aliases without a sameness check; not the culprit this time, low priority. (5) Under the new prompt the judge still merges `late_check_out_fee_applies` into `late_check_out_fee_percent` (boolean into numeric); pinned as a strict xfail in `test_subject_adjudication_collisions_llm.py` — the other four live-run collisions now stay apart. Whether a fee is its own subject or a value of `late_check_out_fee` (with a unit column) is the modelling question to settle before tuning the prompt. (6) `test_rules_extraction.py::test_arrival_window_lands_as_decimal_hours` looks for `arrival` / `latest`; the canonical shape names it `check_in_end_time` — one needle, existing test, needs a yes.

### Done (2026-09-01)

**Reviews + claims ingest.** Dropped the old `claims` table (author/date/stars on the claim, text `campsite_id`) and added `reviews` + FK’d `claims` (`014_reviews_and_claims`). Splitter is **one Google review per 235B call**, then one embed batch; drop `confidence < 0.5`; no aspect/locus yet. Claims store `claim` + `evidence_span` (`016`); dropped unused `claim_uid` (`017`); sentiment is `is_positive` bool, not a polarity string (`019`). Experiments and locked choices: `docs/claims.md`. Hurshat Tal gold split vs 235B is judged by 30B (`test_hurshat_tal_claim_split.py`).

**Visit gate (personal experience).** 30B yes/no **before** the 235B split (`020`). Ads, brochure dumps, and history lectures (Yehiam fortress post) stay on `reviews` with `skip_reason = not_personal` + `skip_note`; **no claims**, no 235B call. Guest reviews still split. Claim filtering is unchanged (splitter prompt + `confidence < 0.5`) — do not add a second claim pass until the splitter leaks encyclopedia rows from reviews that passed the gate. Gold: `visit_gate.json` (ad fail / two Hurshat guest reviews pass). `just branch "title"` slugifies, checks out, pushes.

**Places fetch (legacy) into ingest.** `campsites.google_place_id` (`018`) from Text Search on `campsites.name`, **first hit only**. Dedicated חניון לילה pin when it exists, else the enclosing park — phase one does not hunt sibling listings. `just scrape-reviews` pulls Place Details (newest weekly; `-- --most-relevant` also seeds Google’s best-of 5). CLI does not read JSON; tests still pass a reviews dict into `populate_reviews_and_claims()`. Independent of the INPA availability scrape. Pin mixing (day-visit vs overnight on the same park pin) is unsolved. Splitter still sometimes glues comma-lists of amenities into one claim.

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
- **Amenity count + in-unit locus.** “next weekend, 2 showers in the room” — weekend is Friday night of next ISO week; two **in-room** showers (private / in-unit), not communal camp showers and not `party_size=2`. **Locus landed** (`semantic_constraints[].locus` = room|site, three-lane planner match); **count did not** — `min_count` is still not in the extractor schema and stage-2 RAG is boolean. Failing: `test_extractor_next_weekend_two_showers.py`.
- **Multi-room vacancy search.** Party that does not fit in one unit: compose N rooms of the same type (`ceil(party / max_occupancy) ≤ availability.room_count`) or mix types at one site so occupancies sum. Stage 1 today requires `max_occupancy >= party_size` on a single type. `room_count` on a slot is inventory; `units` is how many to book. Failing: `test_planner_multi_room.py`.
- ~~**Populate** `campsites.amenities`~~ — **done** (`just scrape-rules`). `source/scraper/rules_ingest` reads the static info page for site-level rules and amenities into the new `campsite_rules` table, and mirrors the amenity ids into `campsites.amenities`, so the planner site lane is live. Follow-up: point `search_site_amenities` / `search_stated_amenities` at `campsite_rules` and drop the JSONB columns — see `docs/design.md`.
- **Extractor naming drift → controlled vocabulary.** Measured on `שעות כניסה ויציאה`, 5 interleaved runs: the model finds all 11 facts every run and names 3 of them differently each time (`early_arrival_fee` / `early_arrival_fee_percent`, three spellings of the Saturday late-checkout rule). This is Open IE over an unbounded vocabulary; the cure is to converge on Closed IE. Highest leverage: embed the section, pull the ~30 nearest `subject_vectors` rows, and put them in the extractor prompt — "use one of these where it fits, propose a new name only if none does". Report: `temp/section_split_probe.py`, `temp/split_probe_detail.json`.
- **Constrained decoding.** `response_format: json_schema` with `strict: true` pinning `subject` to an enum of known names turns the above from a request into a constraint. Needs checking whether Nebius Token Factory supports it.
- **N runs + `times_seen`.** Self-consistency is the canonical answer to LLM variance, but worth less here than usual — fact coverage is already 5/5 once sections are split. Its real value is making single-sighting artefacts droppable, which needs a `times_seen` / `last_confirmed_at` pair on `campsite_rules` (the shape `notices` already uses: nothing currently removes a rule, so a one-off hallucination persists with the same standing as a fact seen twenty times). The upsert is idempotent, so re-runs already accumulate the union for free.
- **Fine-tune a small model** for rule extraction once the schema stops moving. The genuinely canonical answer for narrow high-volume extraction: a tuned 7B beats a prompted 235B on consistency at a fraction of the cost.
- **Prompt caching.** 92% of what each extraction call transmits is the system prompt (4,171 chars of instruction against 1,732 chars of page content for a whole site). The prompt is byte-identical across calls, so caching would make the repetition nearly free — check whether Nebius supports it before optimising anything else. Batching sections into one call is the wrong fix: a full pass already emits ~3,800 output tokens against a 2,500 `max_tokens` cap.
- **`_percent` is redundant in a subject name.** `late_check_out_fee_percent` and `late_check_out_fee` are the same rule — `qualifier_unit` already records percent — but the predicate guard blocks the merge (`fee` vs no predicate), so both sit in the dictionary. Either drop the suffix in the extractor prompt or teach `PREDICATE_SUFFIXES` about it.
- **Room-level rules.** `campsite_rules.accommodation_type_id` is always NULL today; per-unit facts still live in `accommodation_types.policy_rules` / `check_in_time` / `check_out_time`. Unifying means teaching `amenity_enrichment` to write `campsite_rules`. See `docs/design.md`.
- **Ingest the policy PDFs** linked from `נהלים, טפסים ומידע כללי` (quiet hours, group conduct, cancellation). Needs a PDF text dependency; the AJAX endpoint and nonce mechanics are recorded in `docs/design.md`.
- **Sub-campsite zones have no home in the schema.** Akhziv's `מה בחניון?` lists two full amenity sets — `חניון צפוני` and `חניון דרומי` — with different counts each (7+4 vs 9+9 shower stalls, 7 vs 5 drinking fountains, 80 vs 60 picnic tables). `campsite_rules` is keyed on `(campsite_id, accommodation_type_id, subject_id)` and a zone is neither of those, so the second list collides with the first and is dropped as CONFLICTING. Either add a `zone` column to the key, or model a zone as an `accommodation_types` row. Until then Akhziv's southern counts are lost. Raising `MAX_TOKENS` to 8000 fixed the truncation that was hiding this, but not the collision.
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
- **Not yet:** amenity counts / in-unit locus (“2 showers in the room”); composing multiple rooms so occupancy sums to the party (see Next §3)

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
- Postgres + pgvector + Alembic (`campsites`, `reviews`, `claims`, `notices`, `subject_vectors`, `campsite_rules`, `accommodation_types`, `availability`, `list_prices`)
- Scrapers under `source/scraper/`: discovery, booking IDs, info-site rate cards, availability/prices, amenity enrichment, site-level rules (`rules_ingest`)

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

Places fetch is landed: `populate_google_place_id.py` then `just scrape-reviews` (newest; `--most-relevant` for seed). Tests may still pass a reviews dict; production CLI does not read JSON.

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

Weekly refresh is **newest** (`just scrape-reviews`). Seed **most_relevant** is opt-in (`-- --most-relevant`).

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
| **P1** | Reviews ingest | **Landed** (place_id + Places fetch + splitter + `is_positive` + visit gate) |
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
