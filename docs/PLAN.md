# Trippy — Product & Engineering Plan

Campsite recommendation agent for Israel (parks.org.il + Google reviews), with RAG over claims/site data, availability/price search, and a Telegram-facing agent.

---

## Progress log

### Done (2026-09-05)

**Rate-card notes parked.** The `הערות למחירון` section is per-rate by construction (each tooltip is prefixed with its rate label) and the extractor dropped the label every time, so the 10:13 run stored `adult_min_age 14` / `child_min_age 5` / `child_max_age 14` (the private-tent price bands), `weekend_min_nights 2` (the air-conditioned bungalow) and, on Yehiam, `mattresses 4` (a family-tent rental) as facts about the campsite; the judge then merged the hut's `weekend_hut_min_nights` into the bungalow's and `mattresses_included` into `mattresses`. `ingest.sections_to_extract` now drops `PARKED_SECTION_TITLES` after parsing (the parser and its tests are unchanged) and the log prints what was parked. `test_rate_notes_parked.py`. Consequence to accept: `test_rules_extraction.py`'s `night` and `age` cases read the fixture through `parse_sections` directly, so they still see the notes and still pass, but they now assert on facts the ingest no longer stores.

**Referent field (next).** Every statement should say WHAT it is about, not only what it states: `applies_to` with a closed kind — `campsite`, `unit` (a rate or accommodation type), `guest_type` (adult / child bands), `membership` (מנויי מטמון) — plus the label as written on the page. Then: (1) the ingest routes `unit` statements to `campsite_rules.accommodation_type_id` when the label matches an accommodation type (the subcamp module already matches unit names) and skips them at site level otherwise, which un-parks the rate notes; (2) the judge is shown both referents and told that different referents never merge, whatever the names — the hut/bungalow and tent/site merges above had matching names and different referents; (3) `guest_type` bands become `<band>_rate_min_age`-style pricing facts rather than admission rules. The cases are pinned as strict xfails in `test_referent_scope_llm.py` (four extractor, two judge) so the suite flips when the field lands.

**First run through the new judge and explainer** (10:13, three pages, $0.039): 36 judge calls, 9 merges, no cross-kind merge; the accessibility rule held (`accessible_huts` for the two huts); the two collisions were the counted `accessible_toilets` against the accessibility clause's uncounted one — the same fact stated twice, which the upsert still labels CONFLICTING because one side has a number and the other none. Explainer: 2 calls, $0.0005, picked the kept side correctly, called the cause `extractor_wrong_value` where "duplicate, one side richer" would be right. Open: a same-polarity collision where only one side carries a number should be labelled a duplicate, not CONFLICTING.

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
