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
