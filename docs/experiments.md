# Experiments log

Every design choice made on the strength of an experiment points here, and
`docs/design.md` records the choice with its reason. Dates newest first; within
a date, in the order the experiments were run. An entry is never edited after
the fact — a re-run is a new entry. Each one says what question it answered,
how production was kept untouched, what came out, what it cost, and what was
decided.

## 2026-09-07

### 7. Does loosening the amenity gate −0.8 → −0.7 catch electricity without tent-as-desert?

**Question.** `electric_hookup` sits at −0.704, outside −0.8, so with no
claim the listing never enters `fits`. Is it trailer-only? Does top 5
already contain it? How much noise does −0.7 add on other queries?

**Setup.** No LLM, no writes. All `campsite_rules` whose subject name
has electric/hookup/outlet/power, with unit name and `evidence_span`.
Then `search_stated_amenities` (limit 80) and `search_site_amenities`
(limit 40) for 11 queries; count types/sites at −0.8 vs extra in
(−0.8, −0.7]. Output `temp/amenity_gate_minus07.json`.

**Result.** `electric_hookup` is **not only trailers**:

| scope | sentence | sites |
|---|---|---|
| `עמדת חניה לקרוואן פרטי` | קיים חיבור חשמל ומים | Horshat Tal, Maayan Harod, Yarkon, Besor, Tel Arad, Mamshit |
| `מתחם PITCH` | …וחיבור חשמל | Yehiam, Maayan Harod |
| site-wide | נקודות חשמל (sometimes לקבוצות בלבד / shade shelters) | Nahal Amud, Kochav Hayarden, Gan Hashlosha, Masada, Tel Arad, Khan Be'erot, Hai-Bar, Yehiam |

Query `"electricity"`, unit ranking (ungated): **top 5 does reach it.**
Ranks 1–3 `electric_outlet` bungalow/tukul **−0.750**; ranks 4–5
`electric_hookup` PITCH then caravan bay **−0.704**. All fail −0.8; all
pass −0.7. `electric_stove` (staff kitchen) is −0.703, just behind.

Noise at −0.7 (unit extras unless noted):

| query | @−0.8 | extra @−0.7 | what the extras are |
|---|---|---|---|
| electricity | 0 | **14** types + 8 sites | outlets, hookups, **stoves** |
| running water | 0 unit / 1 site | 14 types + 17 sites | caravan water hookup; **every** `drinking_water_fountain` at −0.788 |
| desert / in the desert | 0 | **25** tents+huts at −0.719/−0.73 | lodging, not location |
| quiet | 0 | **32** tents/huts/rooms | same |
| near the north | 0 | 5 חושה as `hut` −0.706 | false location |
| fridge | 18 | 3 | AC rooms as fridge |
| hot showers | 14 | 3 | shared_shower / heating |
| AC in room | 17 | 0 | — |
| near the sea | 0 | 0 | still miss |

The judge cannot veto a stated amenity, so desert→tent at −0.719 would
put every tent pitch in `fits`.

**Decision.** Keep amenity gate **−0.8**. −0.7 recovers electricity
synonyms and also makes `tent` satisfy desert/quiet. Caravan-bay hookup
is one of three `electric_hookup` senses; bungalow `electric_outlet` is
the closer electricity hit. design.md amenity gate.

### 6. Does gold sit past rank 5 at gate −0.6 (K=5 vs 10 vs 20)?

**Question.** §5 packed top 5 claims and top 5 rules. If gold evidence
sometimes ranks lower, does K=10 or 20 recover it at the same −0.6
gate? Claims and rules may need different distance cutoffs.

**Setup.** Same 35 (query, site) packs as §5 (any claim `≤ −0.6`). Fetch
the nearest **20** claims and **20** rules with no K cap, then score gold
recall at K=5/10/20, both among gated hits and among the raw 20. Gold
claims = §5 relevant set; gold rules = `dogs_allowed` / `pets_allowed`
for pet-friendly, `electric_hookup` / `electricity` / `electric_outlet`
for electricity, `near_a_desert` / `desert` for desert queries. Distances
on every row. No LLM, no writes. `temp/claim_rule_topk.json`.

**Result.** **Labeled gold is always rank 1.** K=10 and K=20 add no gold.

| query | claim gold d / rank | rule gold d / rank |
|---|---|---|
| desert | Masada atmosphere **−0.750** r1 | none in top 20 |
| in the desert | Masada **−0.767** r1; Mamshit despite **−0.637** r1 | none in top 20 |
| pet friendly | Horshat Tal forbid **−0.700** r1; Ashkelon dogs **−0.615** r1 | `dogs_allowed` false **−0.897** r1 (12/12) |
| electricity | Yarkon **−0.656** r1 | `electric_hookup` true **−0.704** r1 |

Sites with **6–7** gated claims exist (Nahal Amud, Khan Be'erot, Castel,
Tel Arad) — all gold=0; K=5 already dropped only noise. Rules at −0.6
are flooded: Masada desert **20/20** pass, all tent/room/shower; no
location subject in 20. Tent rules sit at **−0.719**, which is *closer*
than `electric_hookup` (−0.704). A shared −0.8 rule gate would drop
tents **and** the electricity rule.

**Decision.** Keep claim K=5 at −0.6; raising K does not recover gold on
this set. Do not use the claim gate as the rule gate: −0.6 over-recalls
lodging subjects, −0.8 would miss `electric_hookup`. Rule threshold
still open. design.md "Planner claim/rule judge".

### 5. Can one 235B call return relevant claims *and* satisfies, given rules too?

**Question.** The planner needs two decisions on a −0.6 top-5 claim pack:
which claims are about the request (evidence for the recommender, including
forbiddens) and whether the guest would get what they asked for (filter).
Official `campsite_rules` must be in view (`dogs_allowed` polarity false,
`electric_hookup`). Does one JSON object (`relevant_claims` + `satisfies`)
hold, or do the jobs interfere so we need two calls?

**Setup.** Same 35 (query, site) packs as §4, but claims capped at top 5
per site and nearest 5 `campsite_rules` (all categories, including polarity
false) attached. Gold satisfies = §4 (Masada desert, Mamshit despite,
Yarkon electricity; pet-friendly none). Gold relevant = only claims actually
about the request, including “Pets are not allowed” / “not allowed to bring
dogs”. Campfires, staff-friendly, desert-animals-on-the-drive, tent/cabin
rules are noise. Combined prompt with polarity few-shots plus a relevant-
but-does-not-satisfy pet example. 235B, temperature 0. No writes. Dump
`temp/claim_rule_judge_packs.json`; output
`temp/claim_rule_judge_combined.json`.

**Result.** **35/35 satisfies and 35/35 relevant-exact.** fp=0 fn=0. 42 s,
~$0.01. Horshat Tal / Ashkelon: relevant = the forbid claim, satisfies =
false (official `dogs_allowed` false). Khan Be'erot / Hai-Bar animals: not
relevant, not satisfies. Yarkon `satisfy_by=both` (claim + `electric_hookup`).
Nearest rules for `"desert"` were tents/cabins at ~−0.72; the judge ignored
them.

**Decision.** One call. Do not split relevant vs satisfies. Claim retrieve
gate −0.6, top 5; planner runs the judge and keeps all relevant claims as
evidence. experiments.md this entry; design.md "Planner claim/rule judge";
claims.md.

### 1. Do date-intent few-shots fix הקרוב vs הבא and בעוד שבועיים?

**Question.** Five live camping queries had the 30B emit a `date_intent` that
`resolve_dates` then computed correctly — but two intents were wrong:
`שישי הקרוב` → `when=next` (should be `this`), and `סוף השבוע בעוד שבועיים`
dropped `weeks_from_now` and used `when=next`. The prompt already stated both
rules and had no few-shots. Do three examples plus a schema/precedence tweak
hold, or does date extraction need the 235B / its own call?

**Setup.** Production extractor prompt only. `today` frozen to Monday 2026-09-07
in the test (prompt + `resolve_dates`). Five prompts × five trials at
temperature 0 on the 30B (`planner_model`). No planner, no writes.

| prompt | expected stay |
|---|---|
| בשישי הקרוב | Fri 11 Sep (`when=this`) |
| בשישי הבא | Fri 18 Sep (`when=next`) |
| סוף השבוע בעוד שבועיים | Fri 25 Sep (`weeks_from_now=2`) |
| יש מקום בחורשת טל לזוג בשישי הקרוב עד 400 שקל ללילה? | Fri 11 Sep |
| משהו שקט במדבר לזוג, אפשר להביא כלב, סוף השבוע בעוד שבועיים | Fri 25 Sep |

**Result.** **25/25.** Bare phrases and the two original queries. 58 s. No
writes.

**Decision.** Keep the 30B. Do not split date intent into a second call.
design.md "Query extractor: date_intent".

### 2. Does a despite-X/Y few-shot unglue desert from tent cleanliness?

**Question.** Mamshit claim 470 was one row: "The shared sleeping tent is
clean and not very dusty despite being in the desert with winds." Query
`"desert"` sat at −0.578, outside the −0.7 claim gate; only Masada's short
"The desert atmosphere is perfect." passed. The 235B splitter had glued a
concessive setting (`בכל זאת מדבר ורוחות`) onto the cleanliness fact. Does
one despite-X → [X, Y] example split them, including on the full review?

**Setup.** Prompt-only change on `SPLIT_SYSTEM`. No DB writes. 235B
`split_one_review`, temperature 0, three inputs × five trials:

| input | what was glued before |
|---|---|
| English one-liner (the few-shot) | n/a (new) |
| Hebrew parenthetical `האוהל עצמו נקי… (בכל זאת מדבר ורוחות)` | the stored claim |
| Truncated Mamshit review containing that span | claim 470 |

Pass = a desert claim that does not also say clean/dust, plus a clean/dust
claim.

**Result.** **15/15.** 44 s. No writes. Stored rows unchanged until
`just populate-claims` after `just clear-claims`.

**Decision.** Keep the few-shot. Do not add GIN yet. claims.md split rules.

### 3. Can a 235B judge salvage a −0.6 claim gate?

**Question.** At −0.6 the claim lane adds many false sites (campfires /
seashore for `"desert"`). If we keep the loose gate for recall and show the
235B *all* claims that passed, telling it most are irrelevant, does it
still only verify a site when a claim actually states the amenity?

**Setup.** No writes. Query vector, then every claim with
`embedding <#> ≤ −0.6` (no top-5 cap), grouped by campsite. One
`Qwen3-235B` call per (query, site), temperature 0. Prompt: loose
retrieval, do not infer from distance or site name, concessive asides
count, passing mentions do not.

| query | sites at −0.6 | gold verifies | judge yes |
|---|---|---|---|
| desert | 7 | 1 (Masada atmosphere) | 1 Masada |
| in the desert | 15 | 2 (Masada, Mamshit glued despite) | 2 same |
| pet friendly | 12 | 0 (none say pets *allowed*) | 2 (Horshat Tal / Ashkelon: pets *forbidden*) |
| electricity | 1 | 1 (Yarkon, limited hookup) | 1 same |

**Result.** Location: **22/22** vs gold (7+15). Khan Be'erot correctly
rejected (reviews say במדבר, stored claims do not). Hai-Bar “desert
animals on the drive” rejected. Mamshit concessive accepted. Electricity
1/1. Pet-friendly: the judge treated a negative policy as verifying the
*topic* (“pets are not allowed” → yes). 35 calls, 382 s, ~$0.004. No
writes.

**Decision.** A post-retrieve 235B judge can eat the −0.6 location noise.
It must be told that *verifies* means the guest gets what they asked for
(`is_positive` / same polarity), not that the topic is mentioned. Not
wired into the planner until that line is in the prompt and re-checked.
claims.md open.

### 4. Does polarity + few-shot stop “pets not allowed” counting as pet-friendly?

**Question.** §3's 235B judge recovered desert sites at gate −0.6 but said
yes on Horshat Tal and Ashkelon for `"pet friendly"` because a claim
*mentioned* pets (`Pets are not allowed`). The recommender needs
*satisfies the request*, not topic overlap. Do an explicit polarity rule
and four few-shots fix that without losing Masada / Mamshit / electricity?

**Setup.** Same retrieve as §3 (every claim `<#> ≤ −0.6` per site). Same
35 (query, site) pairs. New system prompt: `is_positive` must match what
the guest wants; `"Pets are not allowed"` ↛ pet-friendly; concessive
desert aside still yes; electricity with limited coverage still yes.
235B, temperature 0. No writes. Output
`temp/claim_verify_judge_polarity.json`.

**Result.** **35/35 vs gold.** Pet-friendly 12/12 no (Horshat Tal and
Ashkelon: “mentions pets but forbids them”). Desert 7/7 (Masada only).
In-the-desert 15/15 (Masada + Mamshit despite). Electricity 1/1. 51 s.

**Decision.** Keep gate −0.6 and this judge prompt for the recommender
once wired. Not in production until the user is satisfied. claims.md.

## 2026-09-06

### 1. Does the lodging panel parse structurally, or does it need a model?

**Question.** `אפשרויות לינה` is AJAX-only and had never been fetched. Is it a
block of prose that needs an LLM to segment, or does its markup separate units
from rules on its own?

**Setup.** `temp/lodging_panel_testbed.py --panels`: two plain GETs per site
(the page for `body[data-id]`, the panel's `data-cnt` and the inline
`my_repeater_field_nonce`; then `ajax-handler-wp-loadmore.php`), cached under
`temp/panels/`, parsed by `parse_lodging_blocks`. All 18 sites. No LLM, no
database.

**Result.** 18/18 fetched. **68 units, 27 with an inventory count, 1 textless,
8 rule paragraphs.** The catalog is structural: a non-empty `<h4>` is a unit,
its `(N)` is how many the site has, a `<p>` is description. Two bugs the sweep
paid for itself with:

| what | cost of getting it wrong |
|---|---|
| `<b>` is editorial bolding, not a count marker | reading the count out of it and dropping the rest lost the unit **name on 7 sites** and collapsed three distinct Metsada staff rooms onto one name — a `(hotel_id, name)` collision |
| `<h3>` must not open a unit | Akhziv's subcamp headings `חניון צפוני` / `חניון דרומי` are `<h3>`; treating them as units invented two phantom types |

The one thing structure cannot decide: a **second** `<p>` after a heading is
sometimes the unit's description continuing and sometimes a rule about every
unit, in identical markup. 8 such paragraphs; "the heading claims only its
first paragraph" gets 7 of 8, failing on Khān Be'erot's room 5. Formatting does
not separate them either — the unit-specific accessibility intro is `<strong>`
and so is the site-wide rule, while both room specs are `font-weight: 400`.

**Decision.** Catalog parsed structurally, no model. One segmentation call per
panel for paragraph attribution and name normalisation only. design.md
"One tooltip, one pipeline".

### 2. Two passes over a unit: subtract what the first pass read, or not?

**Question.** Pass 1 takes beds and occupancy into columns; pass 2 reads the
same text for amenities and rules. To stop one fact landing twice, should pass 1
return the spans it consumed so pass 2 reads the remainder?

**Setup.** `--run` over 3 sites into `experiments.lodging_*`, `consumed_spans`
on `AccommodationExtract` and a `residual_text()` doing exact-substring removal.
Then the same 3 sites with the subtraction removed and the exclusion moved into
`unit_prompt` instead. 186 and 413 per-unit spans respectively.

**Result.** Subtracting mutilates the evidence. **89 of 186 spans (48%) quoted
text that had never appeared on the page** — the page says
`בכל חדר: 4 מיטות (מתוכם: מיטה זוגית…) מזרנים, כריות…`, the stored span said
`בכל חדר: מזרנים, כריות…`. An `evidence_span` exists so a row can be checked
against its source; half of them could not be. Reading the paragraph as written
and naming the excluded facts in the prompt instead: **8 of 421 (1.9%)**, and
every one of those is a model tic, not a design fault — the 235B
part-translating a word it was told to copy (`מיקרוגל` → `מיקרוwave`,
`בונגלו` → `בóngלו`, `מוגבלות` → `מогבלות`).

**Cost.** 3 sites $0.038; 18 sites $0.097–$0.100, 0 failures on the first full
run.

**Decision.** No subtraction. `consumed_spans` and `residual_text` removed;
`unit_prompt` names bed counts, sleeping capacity and joined-room counts as not
its own. design.md "One tooltip, one pipeline".

### 3. Does the extractor keep to its own predicate contract?

**Question.** The prompt says a rule name's last part is one of exactly twelve
predicates and "never coin another". Nothing checks. Does it hold?

**Setup.** The 18-site run above; every subject grouped by category and name.

**Result.** **19 of 431 rows broke it**, two shapes: `tent_setup` ×14 (a
perfectly good *amenity* name mislabelled `boolean_rule`) and `room_assignment`
×5, which asserts nothing and **disagreed with itself** — `false` at Hurshat
Tal, Mamshit and Khān Be'erot, `true` at Akhziv and Mishmar HaCarmel, from one
identical sentence. `numeric_rule` violations: 0. After adding both sentences to
the prompt as worked examples: **3 rows**, and neither original shape recurs.

**Decision.** Prompt examples, plus `miscategorised_rule()` clearing the
*category* rather than dropping the statement, so a good amenity name mislabelled
a rule survives and is placed on the evidence. The predicate list lives in code,
which `llm-decides-semantics` normally forbids; the user agreed on the grounds
that it checks an output contract the prompt states exhaustively rather than
judging meaning. Reported in the run report, both terminal and Markdown.

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

### 10. Show the judge what each side states (values), not only the sentences

**Question.** The three bad merges in the 19:55 five-page run —
`family_and_friends_group_min_occupancy` (30) → `group_min_occupancy` (80),
`electric_hookup` (site list, "נקודות חשמל") → `electric_hookup_in_caravan_pitch`,
`visitor_service_center_summer_friday_opening_end_time` (16:00) → the weekday
`…_summer_opening_end_time` (17:00) — all happened with both original
sentences already in front of the judge. Does adding what each side *states*
(the term's polarity/qualifier, the candidate's existing rows) change its
answer, without breaking true merges whose numbers legitimately differ?

**Setup.** `temp/judge_values_probe.py`: the judge's exact message format plus
one `states:` line per side; three variants — today (names + contexts),
`+values`, `+values+rule` (one added system-prompt sentence: two statements
from one page stating different numbers or opposite polarities are two facts;
across campsites a different number is normal). 7 cases × 3 variants × 3
calls = 63 judge calls on the 235B, $0.021, no writes. Controls: `freezer (1)`
→ `freezers (2)`, `mattress (16)` → `mattresses (100)`, `rental_equipment` →
`equipment_rental`, `rental_equipment_signature_required` → the commitment-letter
subject (two candidates offered).

**Result.**

| variant | bad merges rejected / 9 | controls merged / 12 |
|---|---|---|
| today | 0 | 12 |
| +values | 8 | 12 |
| +values+rule | 9 | 12 |

Today's presentation merged all three bad pairs 3/3 each — the instability seen
earlier was not luck, these are systematic. With values the judge rejected the
group minimum and the hookup 3/3 and the Friday time 2/3; with the one-line
rule 9/9. No control moved: differing counts across campsites did not stop a
single true merge, with or without the rule.

**Decision.** Pending the user's call. Cost of adopting: the term's value is
already in hand in `_resolve_statements`; a candidate's values need one query
on `campsite_rules` per offered candidate (cache-and-alias misses only), with
"same page" marked by campsite id.

### 11. An accessibility rule in the extractor prompt

**Question.** `הונגשו בחניון הלילה: חניה, שירותים, מקלחות, …` was extracted as
bare `parking`, `toilets`, `showers` — the property the sentence is about was
dropped, the bare names alias-hit the counted amenities and were refused at the
upsert. Does a rule with a worked example fix it, and does it disturb the
sections next to it?

**Setup.** `temp/accessibility_prompt_probe.py`: `SYSTEM_PROMPT` with one bullet
added after the part-of-container one — a property stated about a whole list
belongs in every name; `הונגשו X, Y, Z` → `accessible_x`, `accessible_y`,
`accessible_z`, never the bare nouns — plus the four-line example. Old vs new
prompt on five sections: both Akhziv accessibility clauses, the Hurshat Tal
amenity head (lighting, showers, toilets, `שירותי נכים (2)`, field kitchen),
the group-booking section, the caravan pitch line. 10 extract calls, $0.007.

**Result.** Both accessibility clauses: every subject now `accessible_*`
(parking, toilets, showers, picnic_area, path_to_tent_area, trails), none
bare. The other three sections are identical old vs new except `lighting` →
`area_lighting` (naming drift the judge already merges). Counts, categories,
the min/max pair and the caravan-pitch parts unchanged. One persistent misread
unrelated to the rule: `שתי חושות` (two huts) became `drinking_water_fountain`
under the old prompt and `accessible_sensory_trails` under the new — the model
reads חושות as senses; the huts never come out right.

**Decision.** Pending the user's call.

### 12. Judge confidence: does it track correctness?

**Question.** If the judge is asked for a `confidence` alongside `match`, is it
lower on the answers we know are wrong — and does asking change the answers?

**Setup.** `temp/judge_confidence_probe.py`: §10's 7 cases × 3 variants × 3
calls, schema `{"match": …, "confidence": 0..1}`. 63 judge calls, $0.022, no
writes.

**Result.** Confidence separates right from wrong completely on this sample:

| answers | n | confidence |
|---|---|---|
| right (match or null) | 52 | 0.95 every time |
| wrong (all were merges) | 11 | mean 0.45, range 0.30–0.85 |

A gate of "accept a match only at ≥ 0.9" would have rejected every wrong merge
and kept every right one. Values are coarse — the model emits 0.3, 0.8, 0.85
or 0.95 — and no wrong *null* occurred, so nothing is known about confidence
on missed merges.

Asking for confidence also *moved* the decisions relative to §10, same prompt
otherwise: the Friday closing time was rejected 3/3 under "today" (merged 3/3
in §10), while `electric_hookup` → `electric_hookup_in_caravan_pitch` merged
2/3 with values and 3/3 with values + rule (rejected 3/3 in §10) — at
confidence 0.3 each time. Any change to the output schema is a prompt change
and reshuffles the borderline cases; the confidence is what stayed honest
about them.

**Decision.** Pending. Proposal: adopt values (§10) *and* the confidence field,
and treat a match below 0.9 as null — in this project a missed merge is the
tolerable failure, so a gate that only ever turns merges into inserts is the
safe direction. Before fixing the threshold, run the field over the 24 merges
of the 19:55 report to confirm true merges sit at 0.95.

### 13. Extractor confidence is saturated

**Question.** Same for the extractor: it already emits `confidence` per
statement — does it dip on the statements we know are wrong?

**Setup.** `temp/accessibility_confidence_probe.py`: §11's five sections, old
and new prompt, twice each. 20 extract calls, $0.014, no writes.

**Result.** 96 statements, every one at confidence 1.0 — including
`drinking_water_fountain` / `accessible_drinking_fountains` read out of
`שתי חושות` (two huts) both times. The accessibility rule held on both re-runs
(all `accessible_*`, neighbours unchanged; `שירותי נכים (2)` came out
`disabled_toilets` under the old prompt this time and `accessible_toilets`
under the new — the drift the judge merges). Earlier live runs showed a 0.95
and a 0.9 on ~300 statements; the signal is effectively constant.

**Decision.** The extractor's confidence carries no information and should not
be used for anything. The חושות misread needs a different cure — a glossary
line in the prompt, since it is a translation error, not a judgement.

### 14. The 24 real merges under the proposed judge presentation, gated at 0.9

**Question.** §10 + §12 on the cases we chose. On every merge the 19:55 run
actually made (21 right, 3 wrong), does "values + rule + confidence" keep the
right ones above 0.9 and drop the wrong ones below?

**Setup.** `temp/merge_confidence_24.py`: merges parsed from the run report;
each term's value taken from the `campsite_rules` row it wrote on its page
(the three CONFLICTING ones hard-coded from the report), each candidate's
values from its rows on other campsites; one call each. 24 judge calls,
$0.009, no writes.

**Result.**

| | kept (match ≥ 0.9) | not kept |
|---|---|---|
| 21 true merges | 20 | 1 — `picnic_table` (17) → `picnic_tables_and_benches` (200), match at 0.8 |
| 3 bad merges | 0 | 3 — group minimum null 0.95, Friday time null 0.95, `electric_hookup` *match* at 0.3 |

23 of 24 right. The gate did real work on one of the three: the caravan-pitch
hookup was still answered "match", at 0.3, and only the threshold turned it
into an insert. The one loss is a missed merge of a plural-vs-compound name,
the tolerable direction. Counts that differ across campsites (1 vs 4 carts,
3 vs 11 refrigerators, 16 vs 100 mattresses) did not disturb a single true
merge.

**Decision.** Pending the user's call; the evidence is now 87 calls with no
true merge lost at ≥ 0.9 except the picnic table, and no wrong merge kept.

### 15. Can the 235B explain a collision? (not implemented; a probe)

**Question.** Every upsert collision in a run report marks something that was
extracted or merged wrongly. If the two colliding statements — names, sections,
values, sentences, resolver outcome — are handed to a model, does it diagnose
the cause well enough to be worth wiring in?

**Setup.** `temp/conflict_explainer_probe.py`: all 16 collision blocks from
the 19:55 report to the 235B (the highest Qwen in the rate table), asked for a
cause out of {extractor_wrong_name, extractor_wrong_value,
extractor_hallucination, judge_over_merge, true_duplicate, other}, which side
is right, an explanation and a fix. 16 calls, $0.003, no writes.

**Result, against our own diagnosis.**

| collisions | our cause | model | verdict |
|---|---|---|---|
| `group_min_occupancy` × 6 | judge over-merge | judge_over_merge, keep both, name the family-and-friends subject separately | right, all 6 |
| Friday closing time | judge over-merge | judge_over_merge, two closing times | right |
| `mattresses` (campsite 4) | extractor: unit-specific tent-rental line read as the site count | extractor_wrong_name, "mattresses included in family tent" | right — we had not spotted it |
| `toilets` × 5 | extractor dropped "accessible" | judge_over_merge × 4 (no judge was involved: alias hit), extractor_wrong_name × 1; fixes propose `toilets_count` / `toilets_accessible` | half — sees two facts, misreads the mechanism, invents a `_count` subject the shape rules forbid |
| `showers` × 2 | same | true_duplicate, "deduplicate" | wrong — the accessibility fact is lost |
| `drinking_water_fountain` | hallucinated from `שתי חושות` | judge_over_merge, but notes the misattribution; fix suggests `sensory_stations` | half — spots the misread, then misreads חושות itself |

9 of 16 right, 3 half, 4 wrong. Everything on the judge side was diagnosed
correctly, including one we had missed. Everything on the extractor side was
weak: the model does not know that "alias hit" means no judge ran, does not
know the naming shape, and does not know הונגשו means "made accessible" —
all of which the pipeline knows and could tell it.

**Decision.** Pending. Promising for judge-side collisions as they stand; for
extractor-side ones it would need the resolver mechanics, the subject shape
rules and the accessibility rule in its prompt before its fixes can be
trusted. Not implemented.

## 2026-09-05

### 16. A conflict resolver: explain, then choose an action (probe, nothing applied)

**Question.** On top of explaining a collision (§15, now in production), can the
235B choose what to DO about it from a closed set — drop the new statement,
give it its own subject (overriding the merge), rename the old subject when at
most 3 rows cite it, or add a detail to the kept row — and does it choose well?

**Setup.** New module `rules_ingest/resolve_conflicts.py`: the explainer's
pipeline mechanics (factored out as `explain.PIPELINE_MECHANICS`) plus the
action set and the old subject's facts (name, aliases, rows citing it); code
validates the answer — `rename_old` refused above 3 citing rows, names
normalised, missing names fall back to the extractor's term, missing details
turn `enrich_kept` into `drop_new`. Test set: every distinct collision in the
last three run reports (16 + 2 + 25 collisions, 14 distinct once per-campsite
repeats fold), each with a hand-decided expected action; two calls per case.
`temp/conflict_resolver_probe.py`, 26 calls, $0.010, no writes. Reading the
set showed a fifth action was needed: `reassign_kept` — the kept ROW is the
misfiled one (a "leave by 23:00" stored as `check_out_time`; urinals stored as
`accessible_toilets` 4) while the subject is right for other pages.

**Result.** 16/26 on the strict scoring; 19/26 after two corrections that
were ours, not the model's — the first `validate` refused a `new_name` that
was already an alias of the old subject, which is exactly what overriding a
merge looks like (fixed, tested); and `accessible_huts` for the hallucinated
fountain is a better answer than the `drop_new` we expected.

| case | expected | chosen (×2) | verdict |
|---|---|---|---|
| family-and-friends 30 vs groups 80 | rename_new | rename_new, no name → (old validate) drop_new | ours; right after fix |
| bare `toilets` / `showers` from `הונגשו` | rename_new → `accessible_x` | rename_new → `toilets_accessible` / `showers_accessible` | action right, name wrong shape |
| Friday 16:00 vs weekday 17:00 | rename_new | rename_new, right name | right |
| Saturday-evening late checkout | rename_new | rename_new, right name | right |
| tent-rental mattresses 4 vs 16 | drop_new / rename_new | rename_new → `mattresses_in_family_tent` | right |
| `_friday_eve_hours` 8 vs `_end_time` 16 (summer, winter) | rename_old / rename_new | rename_new, right name | right; `rename_old` never chosen though 1 row cites it |
| counted `accessible_toilets` vs uncounted | drop_new | once `…_in_overnight_area`, once enrich→drop_new | 1/2 |
| fountain vs `שתי חושות` (two variants) | drop_new | `accessible_chosot`, drop_new, `drinking_water_fountains_accessible`, `accessible_huts` | 2/4 |
| `check_out_time` 23 vs 9 | reassign_kept | rename_new → `check_out_tents_by_time` | wrong: it kept the misfiled 23:00 |
| `accessible_toilets` 4 (urinals) vs `disabled_toilets` 2 | reassign_kept | rename_new → `disabled_toilets_count` | wrong, and a forbidden `_count` name |

Two patterns. **The model always reaches for `rename_new`**: 22 of 26
answers; `reassign_kept` 0 of 4 chances and `rename_old` 0 of 4, even where its
own explanation says the kept row is the wrong one ("the sentence describes
urinals"). **Names are the weak output**: 6 of the 22 new names break the
shape the prompt states — a property as a suffix (`toilets_accessible`), a
`_count` suffix, a transliteration (`accessible_chosot`), a predicate inside
the topic (`check_out_time_of_tents_required`). Causes and explanations were
right in 25 of 26.

**Decision.** Pending. The action set is right (the fifth action earned its
place); the prompt needs worked examples for `reassign_kept` and for the two
naming shapes it broke, and a name check against the shape before an action
is applied — or the extractor's own naming pass. Applying actions to the
database is not built; the resolution is a proposal.

### 17. Resolver confidence does not track correctness (unlike the judge's)

**Question.** §12 showed the judge's self-reported confidence separating right
from wrong answers completely. Does the resolver's?

**Setup.** `resolve_conflicts.py` output schema gains `confidence` (kept in the
module: the trace prints it, callers may gate on it). Same 14 cases × 2 as §16,
after the `validate` fix. 26 calls, $0.010, no writes.

**Result.**

| answers | n | confidence |
|---|---|---|
| action right | 16 | 0.95 every time |
| action wrong | 10 | 0.95 × 8, 0.85 × 2 |

No usable gate: 8 of 10 wrong actions came back at the same 0.95 as every right
one. The two 0.85s were `campsite_accessible_toilets` (for the uncounted
duplicate) and one of the two `disabled_toilets_count`. The judge's wrong answers
were borderline calls it half-knew were borderline; the resolver's wrong
answers are confident misreadings — it is sure the 09:00 tent deadline is the
special case and the 23:00 "leave by" is the check-out, sure that
`toilets_accessible` and `disabled_toilets_count` are well-formed names.
Consistency across the two calls was high: 12 of 14 cases got the same action
and name twice; the fountain-vs-huts case gave `accessible_chosot` once and
`accessible_huts` once.

Same distribution as §16 otherwise: `rename_new` 24 of 26, `reassign_kept`
and `rename_old` never; the group-minimum case is now right 2/2 with the
`validate` fix; `accessible_huts` (scored wrong against our `drop_new`) is the
better answer. Accepting it: 17/26.

**Decision.** Keep the field (it costs nothing and the trace shows it) but do
not gate on it. The lever is the prompt: worked examples for `reassign_kept`
and for the naming shape, then a shape check on any proposed name.

---

## 18. The listing matcher on room-numbered names (2026-09-06)

**Question.** After the never-refuse change, the all-sites prices run flagged 12
of 44 matches. Is the remainder ordinary wording distance, or is 30B actually
picking the wrong candidate?

**Setup.** `just scrape-prices` over all 18 sites, `reports/prices_rerun.log`,
Qwen3-30B-A3B. 93 rows written, $0.0048 for 44 `listing_match` calls. Read back
against `info_website_names` and `list_prices`.

**Result.** Wrong, and on the easiest signal available. Khan Be'erot (site 17)
has seven lodging products, three of them room-numbered:

| id | listing |
|---|---|
| 62 | חדר צוות מאובזר כפול חדר מספר 1-2 |
| 63 | חדרי צוות חדרים 3-4 |
| 64 | חדר צוות מאובזר ומונגש חדר מספר 5 |
| 65 | חדר צוות מאובזר חדר מספר 6 |

The rate card states the same numbers, and the model still missed:

| rate-card label | correct | picked | conf |
|---|---|---|---|
| חדר צוות כפול אמצע שבוע (חדרים 1-2) | 62 | **62** | ok |
| חדר צוות כפול סופי שבוע וחגים (חדרים 1-2) | 62 | refused → 59 (tents) | 0.00 |
| חדר צוות קטן אמצע שבוע (חדרים 3 ו- 4) | 63 | 62 | 0.60 |
| חדר צוות קטן סופי שבוע וחגים (חדרים 3 ו- 4) | 63 | 62 | 0.60 |
| חדר צוות גדול אמצע שבוע (חדרים 5 ו-6) | 64/65 | 62 | 0.60 |
| חדר צוות גדול סופי שבוע וחגים (חדרים 5 ו-6) | 64/65 | 62 | 0.60 |

Four of six wrong, all onto the same candidate, and the two `חדרים 1-2` rows --
the one pair the model could have got by copying digits -- split, one right and
one refused outright. The failures are not near-misses in wording; the model is
not reading the room numbers as identifying at all. That is the prompt's doing:
it says a unit or room number does **not** change the product
(`בונגלו עם מזגן מספר 42 = בונגלו עם מזגן`), which is right when the panel names
a product once and wrong here, where the number is the only thing telling four
products apart.

**Second-order damage.** `list_prices_unique_rate` is
`(info_website_name_id, guest_type, rate_period, rate_class)`, so four labels
landing on listing 62 collapse: the run reported 93 stored, the table holds 87,
and Be'erot kept 8 of 11. Never-refuse moved the loss from the matcher, which
reported it, to the unique index, which does not.

**Decision.** Not a model-tier question yet -- the prompt tells it to ignore the
signal, so 235B would obey the same instruction. First the prompt learns that a
room number identifies a product *when more than one candidate carries one*,
with Be'erot as the worked example; then re-measure before spending a tier.
The run report now prints the full prompt for every flagged match
(`print_flagged_prompts`), which is what made this readable at all.

---

## 19. The matcher never saw the room numbers; 30B vs 235B (2026-09-06)

**Corrects §18.** That entry read the room numbers out of the `UNCERTAIN` log
lines and concluded the model had them and ignored them. It did not have them.
The line prints `row.raw_label`, but `snapshot_list_prices` passes
`row.accommodation_type` -- the classifier's normalised type -- to the matcher.
The final summary in the same log prints that field, and it says what was really
sent: `חדר צוות`, `חדר צוות גדול`, `חדר צוות כפול`. No numbers, and in one case
not even `קטן`. §18's "the model is not reading the room numbers as identifying
at all" is wrong; the prompt clause it blames is not what caused this run's
failures. This is the mistake the prompt appendix exists to prevent, found by
the appendix's first user on its first reading.

**Question.** Two then: does the full label fix it, and does 235B do better?

**Setup.** Khan Be'erot's seven real `info_website_names` as candidates. Three
labels the run got wrong, each sent twice -- as the run sent it, and as the full
rate-card label -- to `Qwen3-30B-A3B-Instruct-2507` and
`Qwen3-235B-A22B-Instruct-2507`. Temperature 0, unchanged `SYSTEM_PROMPT`.
12 calls, $0.0020, no writes.

**Result.**

| name sent | 30B | 235B |
|---|---|---|
| `חדר צוות כפול` (as run) | ✅ 0.85 | ✅ 0.85 |
| `חדר צוות כפול סופי שבוע וחגים (חדרים 1-2)` | ✅ 0.85 | ✅ 0.95 |
| `חדר צוות` (as run) | ❌ `חדר מספר 1-2` 0.60 | ✅ `חדרי צוות חדרים 3-4` 0.80 |
| `חדר צוות קטן אמצע שבוע (חדרים 3 ו- 4)` | ❌ `חדר מספר 1-2` 0.40 | ✅ **1.00** |
| `חדר צוות גדול` (as run) | ❌ `חדר מספר 1-2` 0.60 | ❌ `מאהל גדול קבוע` 0.40 |
| `חדר צוות גדול אמצע שבוע (חדרים 5 ו-6)` | ✅ `חדר מספר 6` 0.60 | ❌ `חדר מספר 1-2` 0.40 |

Three readings.

- **The numbers matter, and they are being thrown away before the model sees
  them.** 30B goes 1/3 → 2/3 and 235B reaches 1.00 on a case it answered at 0.80
  without them. The one thing that would help most costs no tokens at all.
- **235B is better where 30B is worst.** On `(חדרים 3 ו- 4)` 30B still picks
  `חדר מספר 1-2` even with the numbers in front of it -- it does not equate
  `3 ו- 4` with `3-4` -- while 235B does, at full confidence. This was the
  question asked of it, and 235B answers it.
- **`חדרים 5 ו-6` has no right answer.** One label spans listings 64 and 65, so
  no single pick is correct and `list_prices_unique_rate` could not hold both
  anyway. Scored ❌ above for whichever it picked; it is really a data-shape
  problem, not a model one.

Both models refuse or wander when handed a bare `חדר צוות גדול` against a list
where four candidates are staff rooms -- 235B's `מאהל גדול קבוע` (a tent
structure) is the worse answer of the two.

**Decision.** Pass the full rate-card label, not the normalised type: it is free
and it helps both models. Re-measure after that before buying a tier -- §18's
"fix the prompt first" was reasoning from a premise that turned out to be false,
and the prompt's room-number clause has not actually been shown to cost anything
yet. The `חדרים 5 ו-6` shape needs a decision of its own: one price legitimately
covers two products, and the schema has no way to say so.

---

## 20. Bracket order decides the 30B's answer; the 235B does not care (2026-09-06)

**Question.** §19 sampled once per cell. Repeated, is the listing match stable?
And does the bracket shape in `(חדרים 3 ו- 4)` matter?

**Setup.** Khan Be'erot's seven real `info_website_names` as candidates, the
`SYSTEM_PROMPT` unchanged, temperature 0. First `חדר צוות קטן אמצע שבוע
(חדרים 3 ו- 4)` five times per model per bracket form, pooled with the three
earlier single-shot runs of a literal verified byte-identical across all three
scripts (`U+0028 … U+0029`, the order every `raw_label` in the database uses).
Then, on the 235B only, four real Be'erot labels three times each with and
without brackets. 44 calls, $0.0091, no writes.

**Result — the same string, the same model, temperature 0:**

| | correct | the answer when wrong |
|---|---|---|
| 30B `(חדרים 3 ו- 4)` | **1/7** | `חדר צוות מאובזר כפול חדר מספר 1-2` @ 0.40 |
| 30B `)חדרים 3 ו- 4(` | **6/6** | — (0.85 every time) |
| 235B, either form | 12/12 | — (1.00 every time) |

Two stable attractors rather than noise: every wrong answer is the same pick at
the same 0.40, and the single right one came back at 0.85, the confidence the
flipped form gives every time. The flip is not a fix — it is evidence that the
30B is deciding this on something that carries no meaning.

**Result — removing the brackets, 235B, n=3 per cell:**

| label | with | without |
|---|---|---|
| `… (חדרים 3 ו- 4)` | 3/3 @ 1.00 | 3/3 @ 1.00 |
| `… (חדרים 1-2)` | 3/3 @ 0.95 | 3/3 @ 0.95 |
| `… (חדרים 5 ו-6)` | 0/3 @ 0.40 | 1/3 |
| `… (עד 4 לנים)` | 3/3 @ 1.00 | 3/3 @ 1.00 |

Identical picks and identical confidences on three of four. The fourth is the
label that spans listings 64 and 65 and has no single right answer; 0/3 → 1/3 is
noise at this n and is not a regression.

**Decision.** `listing_match` moves to `Qwen3-235B-A22B-Instruct-2507`, and
`strip_brackets` removes `(` and `)` from the name before sending — inert on the
235B, and it takes away a lever that should never have moved an answer. The
candidates are not stripped: the pick has to be a string on the list. The
accommodation-type matcher in `populate_availability.py` stays on the 30B,
pinned explicitly, since none of this was measured on its prompt.
design.md "The rate-card listing match runs on the 235B, and never sees a
bracket".

**Cost.** 44 calls at $0.0091 here. In a full run the matcher made 44 calls at
$0.0048 on the 30B; the same run on the 235B is roughly $0.012 — about a cent
per all-sites scrape.

**Closed since.** §19's "pass the full rate-card label, not the normalised
type" is now implemented: `match_info_website_name` takes `full_label` and the
exact test keeps using the normalised name, so the free path is unchanged and
only the model's view widens. The report appendix prints the user message alone.

---

## 21. The full prices run on the 235B (2026-09-06)

**Question.** §20 changed three things at once -- the model, the brackets, and
the string the model is shown. What do they do to a real all-sites run?

**Setup.** `just scrape-prices`, all 18 sites, `reports/prices_235b.log`.
Compared against `reports/prices_rerun.log`, the same recipe on the 30B with the
normalised type and the brackets intact.

**Result.**

| | 30B run | 235B run |
|---|---|---|
| matches flagged | 12 | **2** |
| refusals forced onto a candidate | 3 | **0** |
| rows written / rows kept | 93 / 87 | 93 / **91** |
| `listing_match` | 44 calls, $0.0048 | 43 calls, $0.0094 |
| whole run | $0.0099 | $0.0145 |

Khan Be'erot, the site that motivated all of this, went from 8 rows to 10 and
from three wrong picks plus one price filed under tent camping to zero of
either. `(חדרים 3 ו- 4)` now lands on `חדרי צוות חדרים 3-4` for both its rates,
and both `(חדרים 1-2)` rates land on `חדר מספר 1-2`.

Both remaining flags are the same label in its two rate periods:
`חדר צוות גדול ... (חדרים 5 ו-6)`, which names two products. The model picked
`חדר מספר 1-2` at 0.40 for one and `חדר מספר 5` at 0.60 for the other -- and
those two answers are what cost the run its two lost rows, since one of them
collided on `list_prices_unique_rate` with a row already filed under
`חדר מספר 1-2`. Four cents of model spend fixed everything except the case that
is not a matching problem.

**Decision.** Keep the change. The open item is unchanged and is now the only
one left at this site: one rate legitimately covers two products, and neither
the prompt nor the schema can express that.

**Noticed.** The `matched no lodging product` summary lines still print
`row.accommodation_type` (`חדר צוות גדול`) while the model is now shown
`row.raw_label`. Harmless but misleading; the prompt appendix below them is
correct.

---

## 22. The rescue pass, on a cleared table (2026-09-06)

**Question.** A first answer below `UNCERTAIN_BELOW` is asked again under a
prompt that permits several names. Does it find the split, and does it invent
ones that are not there?

**Setup.** `TRUNCATE list_prices`, then `just scrape-prices` over all 18 sites.
`reports/prices_rescue.log`.

**Result.** It fired exactly twice, on exactly the label that needed it.

| | 30B | 235B | 235B + rescue |
|---|---|---|---|
| rate-card lines matched | 93 | 93 | 93 |
| rows in `list_prices` | 87 | 91 | **94** |
| flagged | 12 | 2 | 2 uncertain + 2 splits |
| refusals | 3 | 0 | 0 |
| run cost | $0.0099 | $0.0145 | $0.0150 |

Both `חדר צוות גדול ... (חדרים 5 ו-6)` rates came back from the first pass as
`חדר מספר 5` at 0.60, and the rescue returned rooms 5 **and** 6 at 0.90 and
0.95. Khan Be'erot now holds 13 rows against 11 rate-card lines, and every one
is right: `(חדרים 3 ו- 4)` on the single catalog entry that covers both rooms,
`(חדרים 1-2)` on room 1-2, and `(חדרים 5 ו-6)` on both rooms at 480 midweek and
680 at weekends. Two calls, $0.0003.

Nothing else split. The prompt's second worked example -- one candidate already
covering several rooms must not be broken up -- is what `(חדרים 3 ו- 4)` needed,
and it stayed single.

**Decision.** Keep it. Two extra calls a run is not a cost worth optimising.

**One row is still lost, and the rescue cannot see it.** Tel Arad prices two
Canaanite structures:

```
מאהל גדול קבוע מבנה כנעני (עד 10 לנים)      860
מאהל גדול קבוע מבנה כנעני כפול (עד 36)     3080
```

against a catalog holding both `מאהל גדול קבוע (מבנה כנעני)` and
`מתחם כפול בתוך מבנה החאן הכנעני`. Both lines matched the first listing
**confidently**, so no flag and no rescue, and the second insert overwrote the
first on `list_prices_unique_rate`. 3080 survives, 860 is gone, and
`מתחם כפול` has no price at all. This is the opposite failure to Be'erot's: not
one label naming two products, but two labels collapsing onto one.

Worth noting the shape of the fix, unbuilt: two rate lines resolving to the same
listing at the same `(guest_type, rate_period, rate_class)` is a contradiction
the code can detect without asking anyone -- the card is pricing two things.
Today that collision is silent, which is how a run reports 93 and stores 94.

---

## 23. Collision resolution: two labels, two products (2026-09-06)

**Question.** §22 left one row lost at Tel Arad, to two labels matching one
product *confidently*. Told that the clash happened, can the model separate
them?

**Setup.** The pair prompt prototyped on Tel Arad's five candidates alone, 5
runs at temperature 0, then wired in and run over all 18 sites on a cleared
table. `reports/prices_collision.log`.

**Result — the prototype, 5/5 identical at 0.95:**

```
Label A: מאהל גדול קבוע מבנה כנעני עד 10 לנים   -> מאהל גדול קבוע (מבנה כנעני)
Label B: מאהל גדול קבוע מבנה כנעני כפול עד 36   -> מתחם כפול בתוך מבנה החאן הכנעני
```

Asked one at a time the same model put both on the first candidate, at 1.00 and
0.80. Knowing the two must differ is the whole difference: it makes `כפול` --
one word of six -- outweigh the five the wrong candidate shares. Stripping the
brackets from the candidates as well as the label changed nothing either way
(3/3 identical picks and confidences), so the brackets were never the problem
here.

**Result — the full run:**

| | 235B | + rescue | + collision |
|---|---|---|---|
| rows in `list_prices` | 91 | 94 | **95** |
| rows lost to a silent overwrite | 3 | 1 | **0** |
| run cost | $0.0145 | $0.0150 | $0.0153 |

One collision call in the whole run, $0.0001, and Tel Arad now keeps 860 on
`מאהל גדול קבוע (מבנה כנעני)` and 3080 on `מתחם כפול בתוך מבנה החאן הכנעני`.
93 rate-card lines, 95 rows, nothing dropped at any of the 18 sites.

The rescue fired three times rather than two: Metsada's `חדר צוות גדול אמצע
שבוע` came back at 0.60 and the second pass confirmed one product rather than
splitting it, which is the behaviour the prompt's second worked example is for.

**Decision.** Keep it. Detection is pure and needs neither a model nor a
confidence -- `colliding_rows` groups by the unique key itself -- so it fires on
the fact of the clash, which is what confidence could never catch here. Only a
pair is asked about; three labels on one product is reported and left alone,
since that shape is more likely a catalog missing an entry than one bad match.
An answer is refused unless both names are on the list and differ, and a refused
answer leaves the rows untouched and says so.
