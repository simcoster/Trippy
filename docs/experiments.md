# Experiments log

Every design choice made on the strength of an experiment points here, and
`docs/design.md` records the choice with its reason. Dates newest first; within
a date, in the order the experiments were run. An entry is never edited after
the fact — a re-run is a new entry. Each one says what question it answered,
how production was kept untouched, what came out, what it cost, and what was
decided.

## 2026-09-04

### 1. Streaming does not change the token accounting

**Question.** If the rules extractor streams its reply (so the ingest can show
progress during a 30–90 s call), do the usage counts still arrive, and do they
match a non-stream call?

**Setup.** `RuleExtractorLLMClient.extract` with `stream=True,
stream_options={"include_usage": True}` on Qwen3-235B, one two-sentence Hebrew
section; then the same messages as a non-stream call. No writes.

**Result.** Usage arrives on the trailing choice-less chunk. Streamed:
prompt 1212 / completion 140. Non-stream: 1212 / 140 — identical. First chunk
after 0.80 s, done in 1.7 s.

**Decision.** Extraction streams; dots render on the `extract:` line. PLAN
2026-09-04 "Rules ingest: visible and explained".

### 2. The extractor follows the canonical subject shape

**Question.** Does the 235B honour `<topic>[_<scope>]_<predicate>` /
`<thing>[_in_<place>]` on real sentences?

**Setup.** Two sections through `extract()`: the Hurshat Tal field-kitchen list
item and the three late-checkout sentences. 2 calls, 3924 in / 1030 out.

**Result.** `מטבח שדה (1) בשלב הזה בלי גז` → `field_kitchen` (amenity, true,
1 count) + `gas_in_field_kitchen` (amenity, false) — the shape the user
specified. Late checkout → `late_check_out_allowed`, `_end_time 17`,
`_fee_percent 50`, `_on_saturday_evening_{allowed,fee_required}`,
`_in_accommodation_units_{allowed,fee_required}`. Caveat: those sentences are
also the prompt's examples, so this proves the shape is read and followed, not
that it generalises; the unexampled lines (`drinking_water_fountain 6`,
`refrigerator 11`) are the better evidence.

**Decision.** Shape adopted in `SYSTEM_PROMPT`. design.md "Extractor".

### 3. Why did `gas_stove_in_field_kitchen` merge into `field_kitchen`? (reproduced, rolled back)

**Question.** Which path merged them — the judge, or the classifier proposing an
existing name?

**Setup.** Production DB inside a transaction: remove the alias from #10, call
`resolve_subject` with the ingest's exact context, capture the
`ResolutionTrace`, roll back. Temperature 0.

**Result (before the prompt change).** `kind=merged`. NN offered exactly one
candidate, `field_kitchen` at −0.850 (the rest −0.65…−0.68, rejected far).
Both term and candidate carried the *same* context sentence. The judge (30B)
answered `field_kitchen` — with `"gas_in_field_kitchen" vs "field_kitchen" ->
null` already in its prompt. The shared sentence read as proof of sameness.

**Result (after adding "identical contexts are not evidence" + the pair as an
example).** `kind=inserted`; judge rejected `field_kitchen`. Second probe:
`late_check_out_available_until` (alias removed from #63) → `kind=merged` into
`late_check_out_end_time`, five candidates offered now that the predicate gate
is gone. 2 chat / 2 embed and 1 / 1 respectively.

**Decision.** Judge prompt gains the identical-context caveat, the
predicate-kind distinction and the synonym groups that do merge. design.md
"Predicates are the judge's call". Follow-up test
`test_subject_adjudication_collisions_llm.py`: 4 of 5 live-run collisions now
stay apart; `late_check_out_fee_applies` → `fee_percent` still merges (strict
xfail).

### 4. The judge alone on 20 direction pairs — `judge_experiment` schema

**Question.** With the `opposed()` antonym guard off, how often does the judge
merge min/max, start/end, open/close pairs? (Deciding between keeping the
guard, a `direction` column, or trusting the judge.)

**Setup.** `CREATE SCHEMA judge_experiment`; `subject_vectors` and
`campsite_rules` cloned with `LIKE … INCLUDING ALL`, own id sequence,
`SET search_path TO judge_experiment, public`; `resolve.opposed` monkeypatched
to `False`. 20 direction pairs + 5 synonym controls, each term with a Hebrew
context stating its direction. Real embedder, real judge (30B). Production
untouched (60 rows before and after). 69 chat / 84 embed, ≈$0.02. Schema left
in place for inspection.

**Result.** Read from the alias arrays (creation order = id order): **6 lost
facts in 40 terms** — `child_max_age`→`child_min_age`,
`mattress_pickup_end`→`start`, `gate_close`→`open`, `campfire_end`→`start`,
`latest_check_in_time`→`last_entry_time` (another pair's subject), and
`car_entry_time`→`check_in_time` (different noun); `earliest_check_in_time` and
`arrival_time` also went into `check_in_time` (defensible). Antonym pairs sit at
−0.84…−0.95, nearer than most true synonyms; two pairs split only because the
distance fell outside −0.75. Controls 4/5 — `dogs_allowed`/`pets_allowed` kept
apart, which is fine: a missed merge is the tolerable failure. Side-finding:
rename-on-insert (`classify(near=…)`) renamed 5 of 40 terms — reordered
`weekend_min/max_nights`, dropped `stay_` from `stay_min_nights`, and invented a
direction: `dogs_entry_time` ("from 16:00") became `last_dogs_entry_time`.

**Decision.** Keep `opposed()` as the one sanctioned exception to the
no-string-lists rule (it rejects before the judge and can only over-split).
Pull rename-on-insert: the extractor names subjects, the classifier supplies
only a missing category. Direction column deferred. design.md "Antonyms are
decided in code", "The classifier no longer names anything".

### 5. Is the over-merge a prompt problem or a model-size problem? (2×2 grid)

**Question.** Does a direction/actor block in the judge prompt fix the merges
from #4, does a bigger model, or both?

**Setup.** 13 cases: the 6 wrong merges from #4, the 2 defensible ones, 5
must-merge controls; single candidate each, Hebrew contexts. Prompts: current
vs current + a block naming the antonym pairs with three examples and an
actor/object rule (`car_entry_time` vs `check_in_time` → null).
Models: Qwen3-30B-A3B (the judge's model until now) vs Qwen3-235B-A22B.
`ADJUDICATE_SYSTEM_PROMPT` monkeypatched; 52 judge calls, 83 s. No writes.

**Result.**

| | wrong merges / 6 | missed merges / 5 |
|---|---|---|
| 30B, current prompt | 4 | 0 |
| 30B + block | 1 (`campfire`) | 0 |
| 235B, current prompt | 0 | 0 |
| 235B + block | 0 | 0 |

Real cost from the project's rate table: 30B $0.10/$0.30 per Mtok, 235B
$0.20/$0.60 — 13 judge calls $0.0021 vs $0.0041, i.e. ~$0.16 vs ~$0.32 per
1000 judge calls. Judge calls scale with vocabulary growth, not ingest volume.

**Decision.** Judge moved to the 235B; the block added to the prompt as well.
design.md "Predicates are the judge's call" (model paragraph). Caveat added
after §6: this grid was one pass (26 correct 235B answers); given the
instability seen in §6, repeat it before treating 0/6 as settled.

### 6. Does the classifier survive the move to the 235B? (it does not)

**Question.** The judge and classifier shared one `MODEL` constant, so §5's
switch moved both. `test_subject_adjudication_llm.py::test_classify_assigns_the_right_category[dogs_allowed-2]`
then failed: the 235B called `dogs_allowed` an amenity. Flake or systematic?

**Setup.** `classify()` on both models, five permission-style rule names
(`dogs_allowed`, `pets_allowed`, `smoking_allowed`, `barbecue_allowed`,
`campfire_allowed`), with and without a Hebrew context, twice each — 40 calls.
Then the failing test's exact call, `classify("dogs_allowed")` with no
context, ten more times on the 235B. No writes.

**Result.** First pass: 30B 0/20 wrong, 235B 0/20 wrong — `dogs_allowed`
came back "rule" 4 of 4 times on the 235B. Ten more calls minutes later:
**9× amenity, 1× rule.** The 235B's answer to the identical prompt is unstable;
temperature 0 is not a determinism guarantee on this MoE model. The 30B was
20/20 "rule". Bounding the impact: both production callers of
`resolve_subject` pass a category (`rules_ingest` sends the extractor's,
`amenity_enrichment/db.py` pins AMENITY), so with no-rename in place the
classifier's category is never consulted in production today — only by tests.

**Decision.** Split the constants: judge on the 235B (§5 stands), classifier
back on the 30B where it was stable. design.md "Predicates are the judge's
call" (model paragraph). Open: the same instability could affect the judge;
re-run §5's grid a few times before relying on its zero.

### 7. Three categories: split `rule` into `boolean_rule` / `numeric_rule` — `category_split_experiment` schema

**Question.** The live two-site run earlier today lost two facts to the judge
accepting a boolean/numeric pair as one subject: `late_check_out_end_time`
(17:00) merged into `late_check_out_allowed` on site 1 and then propagated by
alias to sites 19 and 20; `early_arrival_fee_required` merged into
`early_check_in_fee_percent` on site 19, so the 50% was dropped as CONFLICTING
on both Akhziv camps. Both pairs are in the judge prompt as worked "null"
examples. If the extractor tags each rule as boolean (answered by polarity:
predicates `allowed` / `required`) or numeric (answered by a number: every other
predicate), and candidates come only from the same category, do the merges
disappear, and what else moves?

**Setup.** `temp/category_split_experiment.py`. Schema
`category_split_experiment` with `subject_vectors` / `campsite_rules` cloned
`LIKE … INCLUDING ALL`, own sequences, category CHECK widened to 1..3, an HNSW
partial index for `category = 3`. The ingest connects with
`search_path=category_split_experiment,public`, so its bare table names land in
the schema while `campsites` still reads production; the script refuses to run
unless `to_regclass('subject_vectors')` resolves inside the schema, and asserts
production row counts before and after (64 / 145, unchanged). Nothing in
`source/` edited: `SYSTEM_PROMPT` (in `llm` and the `subcamps` copy) rewritten
to three categories — a category bullet naming the predicate split and every
`/ rule /` example relabelled by its unit — `RuleExtract` swapped for a payload
that accepts the new labels, `resolve.category_label` given the third name.
Same two sites, same models, empty vocabulary, like the production run.
20 extract / 25 judge / 70 embed, $0.0256, 459 s (production run: 32 judge,
$0.0277, 489 s).

**Result.**

| | production run | three categories |
|---|---|---|
| subjects | 64 | 64 (31 amenity, 15 boolean, 18 numeric) |
| judge calls | 32 | 25 |
| collisions (site 1 + Akhziv) | 1 + 9 | 1 + 5 |
| `late_check_out_end_time` 17:00 | lost on 1, 19, 20 | kept on 1, 19, 20 |
| `early_*_fee_percent` 50% | lost on 19, 20 | kept on 1, 19, 20 |

Both target merges are gone, and for the reason predicted: `late_check_out_end_time`
(numeric) was offered only `check_out_time` at −0.806 and the judge rejected it;
`late_check_out_allowed` was never a candidate. `early_arrival_fee_required`
(boolean) was offered only `early_arrival_allowed` and was rejected;
`early_arrival_fee_percent` (numeric) had nothing within −0.75 at all. The four
remaining Akhziv collisions are all the `נגישות` section re-emitting `toilets` /
`showers` / a fountain as bare amenities — not a category matter.

One **new wrong merge**, same category: `family_and_friends_group_stay_min_occupancy`
(30) merged into `group_stay_min_occupancy` (80), single candidate at −0.898,
and the 30 was dropped as CONFLICTING. The production run kept the same pair
apart (named `…group_booking…` that time, −0.923, rejected). Both names are in
the same numeric category, so the split cannot help here; this is the
broader-vs-narrower rule the prompt already states, decided the other way on a
second look — the §6 instability, now seen on the judge.

One **extractor mislabel**: `hot_water_in_showers` came back `boolean_rule`
(no amenity twin existed, so no duplicate this time). This is the cost of the
design: a wrongly tagged term is searched against the wrong shelf and a
duplicate written where nothing will find it. Extraction noise otherwise as
usual on this model — `lighting` / `field_lighting`, `picnic_tables` /
`picnic_tables_and_benches`, `shade_sails_on_beach` / `shade_screens`,
`mattresses` / `mattresses_for_rent` all landed as separate subjects (missed
merges, the tolerable direction), and פלטות became `planks_for_rent`.

**Decision.** Pending the user's call. What the run shows: the split removes
exactly the class of merge it targets, saves ~20% of judge calls, and shifts
the risk to extractor mislabels; it does nothing for same-kind over-merges,
which the judge still gets wrong on occasion.

### 8. The split in production code, re-run on the same two pages (smoke run)

**Question.** §7 was accepted. Does the ported code — `SubjectCategory` with
three members, migration `030`, the rewritten extractor prompt with the
numeric-range example, the run report file — behave the same on the same
pages, and does the range example land?

**Setup.** `python -m source.scraper.rules_ingest.ingest --limit 2` with
`DATABASE_URL` carrying `search_path=category_split_experiment,public` (schema
reset first), `RULES_REPORT_DIR=temp/reports`, `SCRAPE_COST_LOG` pointed at
`temp/`. Production tables 64 / 145 before and after. 20 extract / 27 judge /
73 embed, $0.0278, 447 s. Report: `temp/reports/2026-09-04_191431.md`.

**Result.** Categories 31 amenity / 13 boolean / 21 numeric. The two §7 target
merges stayed gone: `late_check_out_end_time` and `early_arrival_fee_percent`
are separate numeric subjects with their 17:00 and 50% on all three campsite
rows. The range example works at the extractor: `(30-80 לנים)` came out as
`family_and_friends_group_min_occupancy 30` + `_max_occupancy 80`, and `מעל 80`
as `group_min_occupancy 80`. Then the judge merged the family-and-friends min
into `group_min_occupancy` (single candidate, −0.874) and the 30 was dropped as
CONFLICTING on all three rows — the same pair merged in §7 and was kept apart
in the production run; two of three runs now. A second same-shelf merge:
`late_check_out_on_saturday_evening_allowed` into `late_check_out_allowed`
(−0.917, one of two candidates; dropped as a duplicate since both are true, so
the Saturday variant is simply gone as a subject). Both pairs are the
narrowing rule the judge prompt states with near-identical worked examples
(`min_weekend_nights` vs `min_nights`, `late_check_out_saturday_allowed` vs
`late_check_out_allowed`). Extractor: `hot_water_in_showers` tagged
`boolean_rule` again (2/2), `showers_women_count` / `showers_men_count`
emitted again, this time as `numeric_rule`.

**Decision.** Split confirmed in production code (design.md "Category: three
shelves"). Open: the 235B judge accepts a narrower name as the broader subject
often enough to matter, on the same shelf, with the counter-example in its
prompt; the split cannot reach that. Candidates: repeat the §5 grid with these
two pairs, or an experiment on candidate presentation (one candidate at a time,
or showing the qualifier word explicitly).

### 9. Without the ("in", "out") antonym pair, does the judge reject what it blocked?

**Question.** In today's runs `opposed()` rejected
`late_check_out_in_accommodation_units_*` as candidates for the other
`late_check_out_*` subjects only by accident — through the ("in", "out") pair,
which fires whenever one name has `in` and the other `out`, even when both have
both. Is the accident load-bearing, or would the judge get those right anyway?

**Setup.** `temp/in_out_guard_probe.py`: `pick_match` on the 235B with the
run's Hebrew contexts, offering exactly the candidates the judge would have
seen without that pair; three calls per case; no writes. 12 judge calls,
$0.004.

**Result.**

| term | candidates | judge ×3 |
|---|---|---|
| `late_check_out_in_accommodation_units_allowed` | `late_check_out_allowed` (−0.951) | merged 3/3 |
| `late_check_out_in_accommodation_units_fee_required` | `…_units_allowed` (−0.907), `late_check_out_allowed` | merged into `…_units_allowed` 3/3 |
| `late_check_out_on_saturday_evening_allowed` | `late_check_out_allowed`, `…_units_allowed` | null 3/3 |
| `check_out_time` | `check_in_end_time` (−0.896), `check_in_start_time` | null 3/3 |

The first two are exactly what the guard has been preventing: a scoped variant
folded into its parent, and a `fee_required` folded into an `allowed` on the
same shelf — the judge said yes every time, with the narrowing rule and the
"different kinds of question" rule both in its prompt. The Saturday variant,
which the judge merged once in a live run, was rejected all three times here.
`check_in` / `check_out` was kept apart.

**Decision.** Keep the ("in", "out") pair. Its accidental reach is doing real
work, and the guard can only over-split (design.md "Antonyms are decided in
code"). Noted that `opposed()` also fires when both names contain both words;
left as is, since every such pair seen so far should indeed stay apart.
