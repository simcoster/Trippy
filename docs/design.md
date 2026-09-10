# Subject vectors and campsite rules — provisional decisions

Landed 2026-09-02 with migrations `023_subject_vectors` / `024_campsite_rules` and
the `source/scraper/rules_ingest` + `source/scraper/subjects` packages.

Everything below is a decision we expect to revisit. Read it before extending the
schema, so you know which parts are load-bearing and which are scaffolding.

## Locked

| Choice | Decision |
|--------|----------|
| Dictionary table | `amenities` → `subject_vectors`: one row per subject, `name` canonical, `aliases[1] = name`, `category` 1 amenity / 2 boolean rule / 3 numeric rule (migration `030`) |
| Subject naming | snake_case English, carries the predicate, **always positively phrased** — `dogs_allowed`, never `dogs_not_allowed`. Enforced by the extractor prompt and backstopped by `subjects/naming.to_positive_subject` |
| Polarity | nullable `BOOLEAN` on `campsite_rules`: `True` allowed/provided, `False` forbidden/not provided, `NULL` a pure quantity |
| Qualifier | `NUMERIC` + `qualifier_unit SMALLINT`. Direction lives in the name (`min_weekend_nights`, `max_occupancy`), matching the extractor's existing `policy_rules` keys. Times of day are decimal hours: 20:30 → 20.5 |
| Alias resolution | exact `aliases @>` hit → 5-NN by `<#>` + 30B adjudicator → 30B classify + insert. Only misses cost an LLM call |
| Embedding | the **canonical name**, not the surface form, so a row's vector does not drift as aliases accrue |
| Ingest scope | site-level only, static HTML only |
| Vector op | `subject_vectors_embedding_idx` is `hnsw (embedding vector_ip_ops)` |

## Provisional — expect these to change

### Category: three shelves, decided by what answers the statement

`category` was amenity / rule until 2026-09-04. Rules now split by what answers
them:

| | answered by | predicates |
|---|---|---|
| 1 `amenity` | `polarity` | none — a bare noun |
| 2 `boolean_rule` | `polarity` | `allowed`, `required` |
| 3 `numeric_rule` | `qualifier` | `time`, `fee_ils`, `fee_percent`, `min_age`/`max_age`, `min_nights`/`max_nights`, `min_occupancy`/`max_occupancy`, `count` |

The extractor tags every statement (it has the sentence; a classifier shown one
word does not), and the resolver searches only the term's own shelf, so a
permission is never a merge candidate for a deadline on the same topic. The live
run that forced this had the judge merge `late_check_out_end_time` into
`late_check_out_allowed` and `early_arrival_fee_required` into
`early_check_in_fee_percent` — both pairs sit in its prompt as "null" examples —
and every campsite then lost the 17:00 and the 50%. Re-run on the same two pages
in an isolated schema with three categories: both merges gone, judge calls
32 → 25, no new cross-kind merge (experiments.md 2026-09-04 §7).

What the split does not do. Same-shelf over-merges are still the judge's to get
right (`family_and_friends_group_stay_min_occupancy` went into
`group_stay_min_occupancy` in that run). And the tag is trusted: a term the
extractor mislabels — `hot_water_in_showers` came back `boolean_rule` once — is
searched on the wrong shelf and, if a twin exists, duplicated where nothing will
find it. The amenity / rule line is still fuzzy (`barbecue_allowed` and
`barbecue_equipment` come from one sentence); `category` remains a search
convenience, not an ontology. Migration `030` widens the CHECK and adds the
third partial HNSW index but does **not** reclassify existing category-2 rows —
that would be a suffix list — so the vocabulary is rebuilt:
`scripts/clear_rules.py --subjects`, then `just scrape-rules`.

### `campsite_rules` is the only home for an amenity (migration `027`)

Amenities used to live in two places: `campsite_rules`, and four JSONB arrays of
`subject_vectors` ids — `amenities` / `not_included_amenities` on both `campsites`
and `accommodation_types`. The planner's two lanes read the JSONB, so
`rules_ingest.db.sync_campsite_amenity_ids` mirrored site-level rows into
`campsites.amenities` after every ingest.

`027_drop_amenities_jsonb` ended that. Both lanes in `source/agent/search.py` now
join `campsite_rules`, the mirror function is gone, and the four columns and the
two `*_with_amenity_names` views are dropped.

The two arrays only ever carried one bit between them — provided, or explicitly
not provided — which is `campsite_rules.polarity`. The equivalent of "in the
`amenities` array" is `polarity IS DISTINCT FROM false`: a NULL polarity is a
bare quantity ("6 fountains"), which still means the thing is there, and only an
explicit `false` was ever a negative.

The two sides were **not** symmetrical, which is the part worth remembering:

| | before `027` |
|---|---|
| `campsites.amenities` | a derived mirror; safe to drop |
| `accommodation_types.amenities` | the **only** copy of per-unit amenities — 12 types held 98 ids, and `campsite_rules` had 0 rows with an `accommodation_type_id` |

So the migration backfills before it drops. Verified first in
`temp/drop_amenities_jsonb_testbed.py`: every one of the 105 subject vectors was
used in turn as a query, and old (JSONB) and new (`campsite_rules`) SQL returned
identical top-5 results — 105/105 on both lanes. Then
`temp/verify_027_roundtrip.py` ran upgrade *and* downgrade against the real rows
inside a rolled-back transaction: all 98 ids survived, and rebuilding the arrays
from the rows reproduced them exactly, 30/30 rows identical. That round trip is
the actual proof that nothing lived in the JSONB the rows cannot hold.

### `accommodation_type_id` is filled by the availability scrape

It used to be NULL on every row. Migration `027` backfilled per-unit amenities as
`campsite_rules` rows carrying the type's id, and the availability scrape writes
them from there on. The info-page ingest still only writes site-level rows: the
info page's `אפשרויות לינה` panel is a second, weaker source for per-unit data, so
`sections.parse_sections` drops it on purpose. The rate-card notes
(`הערות למחירון`) are parsed but parked at the ingest (`PARKED_SECTION_TITLES`):
every note is about one rate, the extractor drops the rate label, and the facts
landed as campsite-wide rules — `child_min_age 5`, `weekend_min_nights 2`,
`mattresses 4`. They come back when statements carry a referent that can route
them to an accommodation type (PLAN 2026-09-05 "Referent field").

### The rate-card listing match runs on the 235B, and never sees a bracket

A rate-card label is resolved to an `info_website_names` row by exact name, else
one LLM pick over that site's lodging products. That pick is the **235B**, not
the 30B the rest of the info-site scrape uses.

The 30B does not merely make mistakes here; it makes them on distinctions that
should not exist. On `חדר צוות קטן אמצע שבוע (חדרים 3 ו- 4)` against Khan
Be'erot's seven products it answered `חדר צוות מאובזר כפול חדר מספר 1-2` six
times in seven, always at 0.40 — and answered correctly six times in six when
the same bytes were sent with `(` and `)` swapped. Nothing about a bracket's
direction should decide which room a price belongs to. The 235B was right twelve
times in twelve across both forms, at 1.00 (experiments.md §20).

So both were changed. The model moved up, and `strip_brackets` removes `(` and
`)` from the name before it is sent — measured inert on the 235B, same picks and
same confidences on three of four real Be'erot labels and no worse on the
fourth, which spans two products and has no single right answer. The candidates
are never stripped: the answer has to come back as a string on the list.

**The two stages read different strings.** The exact test uses the classifier's
normalised type, because only that can equal a catalog name and cost nothing;
the model is shown the raw rate-card label, because normalisation is what drops
`(חדרים 3 ו- 4)` and those numbers are the only thing separating four staff
rooms at Khan Be'erot. The run report prints the user message for every flagged
match, so what the model was actually asked is on the page rather than inferred
from the summary line.

**A doubted answer is asked again; a clash is settled last.** Below
`UNCERTAIN_BELOW` -- or on a refusal -- `pick_names` may name several products,
because one rate line can price two rooms (`חדרים 5 ו-6`) and a single pick has
to be wrong about one of them. Separately, and with no model involved,
`colliding_rows` finds two rate lines resolving to the same product at the same
`(guest_type, rate_period, rate_class)`: that is a contradiction on the unique
key, so one match is wrong however confident it was. Such a pair gets one
`pick_pair` call that must answer with two *different* candidates. Both are
measured (experiments.md §22, §23); together they took a full run from 87 stored
rows to 95, with nothing silently overwritten.

The accommodation-type matcher in `populate_availability.py` shares the class
but keeps the 30B, pinned explicitly. It runs a different prompt for a different
job, and none of the above was measured on it. The booking name and the
candidate list are lowercased for that call (`מתחם pitch` vs catalog
`מתחם PITCH` is the same product; a pick that is not an exact candidate
string comes back null). The original casing is restored before the alias
is stored.

### One tooltip, one pipeline

A unit's tooltip is a section like any other, so `rules_ingest.units` hands it to
the same `RuleExtractorLLMClient`, `resolve_subject` and `upsert_campsite_rules`
the info page's sections go through — `scrape-availability` now files conflict
cases and writes a run report on the same terms `scrape-rules` does. Only the
scope differs: rows carry the type's `accommodation_type_id`, and `unit_prompt`
countermands the production prompt's "this section describes the CAMPSITE AS A
WHOLE" line, which is exactly backwards for a tooltip.

Before this, `amenity_enrichment` had its own amenity path: `ExtractorLLMClient`
returned `amenities` / `not_included` as bare strings and `write_unit_amenities`
wrote `(subject_id, polarity)` and nothing else. Every per-unit row therefore had
a NULL `evidence_span`, `source_url`, `confidence` and `qualifier` — a site rule
could be checked against the sentence it came from and a unit rule could not —
the merge judge was called without `states=` or `campsite_id=`, so it decided
sameness with neither side's assertion in view, and collisions were resolved by
letting the later write win instead of being filed for review. Counts the tooltip
states ("4 מיטות") had nowhere to go at all.

`ExtractorLLMClient` still reads the same tooltip for what belongs in the type's
own columns: beds, occupancy, `room_count`, check-in/out times and the
`policy_rules` JSONB. Those remain the older shape, and unifying them is the next
step — `campsite_rules` can express every one of them, and the unit prompt already
extracts the numbers alongside.

Named-place expansion (`מכתש רמון` → the place, `near_a_crater`, `near_a_desert`)
moved into `unit_prompt` with the rest; it used to live in the amenity extractor's
prompt, with a `PlaceEnrichmentLLMClient` second pass that nothing called.

A private-caravan bay's חיבור חשמל ומים is **not** `electric_hookup` /
`water_hookup`. Retrieve matches the subject name, not the unit type on
the row, so those bare names look like a guest socket and satisfy an
`"electricity"` query for a tent stay. The unit prompt makes one
exception to "never name the unit in a subject": that listing emits
`caravan_bay_electric_hookup` and `caravan_bay_water_hookup`. A PITCH
tent or bungalow with חיבור חשמל stays `electric_hookup` /
`electric_outlet`. experiments.md 2026-09-08 §1. Stored rows stay the
old names until the unit is re-extracted.

### `NULLS NOT DISTINCT` is load-bearing

Every row this pipeline writes has a NULL `accommodation_type_id`. Standard SQL
treats each NULL as distinct, so a plain unique constraint would let every re-ingest
insert a fresh copy of every site rule. `campsite_rules_scope_subject_key` is
declared `UNIQUE NULLS NOT DISTINCT` (Postgres 15+) and is the `ON CONFLICT` target.
Do not "simplify" it.

### Over-merging is the expensive failure, so splitting wins ties

The first live Hurshat Tal run merged `barbecue_allowed` onto `barbecue` **and**
`barbecue_equipment_included` onto `barbecue`. All three are near-neighbours in the
vector space and the 30B adjudicator accepted both. Because the unique key allows
one row per subject per scope, the second statement was then dropped at upsert and
the "bring your own grill" fact vanished silently. Merging too eagerly destroys
data; splitting too eagerly only leaves a slightly redundant dictionary whose
vectors sit close together anyway, so RAG recall barely notices.

Three defences, in order:

1. **Category, from the extractor.** `RuleStatement.category` is read off the
   sentence — the extractor knows `ניתן להדליק מנגל בציוד עצמי` states a permission
   *and* a missing provision, where the classifier only ever sees the bare word
   `barbecue`. It arrives before resolution at no extra cost (a field in a reply
   the model already sends), and candidates of the other category are dropped.
   This matters because the vector space genuinely ranks the wrong thing first:
   from `barbecue`, the rule `barbecue_allowed` sits at −0.891 while the correct
   amenity `barbecue_equipment_included` is further away at −0.854.
   A statement the extractor labels `amenity` pins its search the same way.

   The filter is **in the SQL**, not applied afterwards. Post-filtering spends all
   five nearest-neighbour slots on the wrong category and can leave nothing at all
   to consider — the first traced run offered zero candidates for
   `barbecue_allowed` because its five nearest were every one of them amenities,
   so it never saw a single existing rule. The partial indexes from migration
   `025` serve exactly this query. Note this applies to *resolution* only: the
   planner's lanes stay unfiltered and see both categories.
2. **Predicates, in the prompt — no longer in code.** `barbecue_allowed` (may I
   grill?) and `barbecue_equipment_included` (is a grill provided?) are different
   kinds of question about one noun and must stay apart; `late_check_out_fee`
   vs `late_check_out_available` likewise. A suffix-list filter
   (`naming.same_predicate`) used to enforce this before the judge saw the pair.
   It was removed after it fragmented `late_check_out_*` into nine subjects —
   see "Predicates are the judge's call" below. The distinction is now stated in
   `ADJUDICATE_SYSTEM_PROMPT` with the synonym groups that *do* merge.
3. **Prompt, for the semantics.** Narrowing qualifiers (`accessible_toilets` vs
   `toilets`, `mattresses_for_rent` vs `mattress`) are judgement calls, not
   structure, so they live in `ADJUDICATE_SYSTEM_PROMPT` with worked counter-examples.
   `test_subject_adjudication_llm.py` pins both directions.

A deterministic "one name is the other plus a word ⇒ not a match" rule was tried and
rejected: it also splits `mattress_pickup_time` from `mattress_rental_pickup_time`,
which really are one subject.

**Every resolution is traced.** `resolve_subject` prints one line saying what it
considered and why, because an over-merge is otherwise invisible:

```
'barbecue_equipment_included' from extractor (amenity). no alias match. ran NN,
top 5: [barbecue -0.914[predicate], barbecue_pit -0.882[predicate], ...].
considered 0. ADJUDICATOR rejected all. classified as amenity
'barbecue_equipment_included'. INSERTED.
```

`verbose=False` silences it and `trace_sink=[]` collects `ResolutionTrace` objects
instead — that is how the unit tests assert on decisions rather than on stdout.

**Watch this over time.** The other health check is the alias list — a subject accumulating
unrelated aliases is an over-merge that is already costing rows:

```sql
SELECT name, category, aliases FROM subject_vectors
WHERE array_length(aliases, 1) > 1 ORDER BY name;
```

### Why one table and not two

Rules and amenities share `subject_vectors` rather than living in
`amenity_vectors` / `rule_vectors`. Two tables would make a cross-category merge
structurally impossible, which is appealing, but:

- Two partial HNSW indexes (`WHERE category = 1` / `= 2`, migration `025`) give the
  **same query plan** — a `UNION ALL` of two `ORDER BY … LIMIT` branches is two
  `Index Scan`s under an `Append`, one round trip. There is no performance argument
  either way.
- `campsite_rules.subject_id` would become polymorphic: two nullable FKs plus a
  CHECK, or no referential integrity on the hot table.
- The category would have to be known *before* the search, on every miss. It now is
  — but from the extractor, not from a classify-first call, and a wrong category
  with two tables silently writes a duplicate into the table nothing will search.
- Cross-category merges were only half the problem anyway:
  `barbecue_equipment_included` → `barbecue` is amenity-to-amenity, and separate
  tables do nothing about it.

Revisit if the category filter turns out to leak.

### A range needs two subjects

`campsite_rules` holds one row per (scope, subject) and one `qualifier` column, so
a range cannot fit in a single statement. The first traced run lost half of
`חלוקת מזרנים בין השעות 15:00-20:00, החזרת מזרנים בין השעות 11:00-8:00` — the
extractor emitted both bounds under one subject and the second was discarded at
upsert. The convention (same as `check_in_time` / `latest_arrival_time` for the
arrival window) is two subjects with the bound in the name:

```
mattress_pickup_start_time  15  hour_of_day
mattress_pickup_end_time    20  hour_of_day
```

The prompt states this and `test_rules_extraction.test_no_subject_is_stated_twice`
guards it. If ranges become common, the alternative is a `qualifier_max` column.

### Experiments live in the `experiments` schema

Production schema is not changed to try something out. `just
setup-experiments copy` rebuilds `experiments` as a copy of `public`
(tables, rows, views; FKs stay inside the schema). `--empty table,…`
truncates after the copy. Scrapes, search and the planner then run with
`TRIPPY_SCHEMA=experiments`, which makes `db.connect.connect` set
`search_path=experiments,extensions`:

```
just setup-experiments copy
just on-experiments scrape-info -- --site 2
```

Pytest still clones an **empty subset** of tables in the fixture
(`experiments_schema.TABLES`); that path does not copy production rows.

**No foreign key crosses between the two schemas, in either direction.** That is
the part doing the work. While the testbeds lived in `public` under a `test_*`
prefix, the separation was a naming convention, and it had already failed twice:
one testbed table held a real FK to `accommodation_types`, which put it inside
production's cascade paths, so `just clear-data` emptied it (it is now
`just clear-availability`, which does not); and every "what is
in this database?" query returned testbed tables mixed in with real ones.
Whatever production table a testbed needs is **copied** into the schema instead —
`campsites` and `accommodation_types` are there for that reason, with their own
sequences, because a shared sequence is a dependency as real as a foreign key.

A read that genuinely wants production data names it: `public.subject_vectors`
seeds the vocabulary so merges are judged against a realistic dictionary. Every
unqualified name resolves inside `experiments`, so a forgotten prefix cannot
quietly hit a production table. `temp/isolate_experiment_schema.py` performs the
separation and refuses to commit if any FK still crosses.

The only production change this needs is dependency injection:

- `SubjectStore(table=..., has_context=...)` — where subjects live, and whether the
  table carries the `context` column. The table name is interpolated into SQL, so
  `SubjectStore` rejects anything that is not a plain identifier. (Which is also
  why the search path, not a schema-qualified name, is what points a testbed at
  its own tables.)
- `upsert_campsite_rules(table=...)` and `ingest_site(store=, rules_table=)`.

A writer that names its table in a SQL literal cannot be tested against a copy at
all: `sync_campsite_amenity_ids` hardcoded `campsites` in its `UPDATE`, so
`temp/split_campsites_testbed.py` had to duplicate its SQL. That function is gone
with migration `027`, but the lesson stands.

### Context: what a subject was first read from

`context` is the sentence a subject came from — `"מה בחניון?: שירותים (15 תאי
שירותי נשים ו- 15 תאי שירותי גברים)"` for a site list, `"חדר צוות: בכל חדר ...
שירותים, מקלחת מים חמים"` for a room tooltip. It is stored on the subject and shown
to both small LLMs:

- the **sameness judge** sees the term's context and each candidate's, which is the
  only way to tell a 30-stall communal block from a room's own bathroom — the names
  `toilets` and `bathroom` alone give it nothing to work with;
- the **classifier** sees it when naming a brand-new subject.

Promoted to production in migration `026_subject_context` after the testbed
showed it working: `toilets` stayed a separate subject from `bathroom` at −0.868
with four candidates offered, where it had merged before. `DEFAULT_STORE` now
carries `has_context=True`; a store built with `has_context=False` still resolves
and simply drops the context. Rows created before `026` have NULL — then the judge
decides on the names alone, as it did before.

On the unit path the context is `_statement_context`'s section title plus the
statement's own evidence span — the unit name and the sentence — so a room's own
`bathroom` arrives with what distinguishes it from the site's communal `toilets`.
`ensure_amenities(contexts=...)` used to do this from the tooltip as a whole; it
is no longer on the path.

### The classifier renames, but only to fix a real problem

`pick_match` (the sameness judge) does the matching. `classify` runs only when
`pick_match` finds nothing, and gives the new subject its canonical name and
category. It used to rewrite freely, which was non-deterministic and sometimes
destructive:

| extractor said | classifier stored |
|---|---|
| `coolers` | `cooler` in one run, `cooler_included` in another |
| `child_max_age` | `max_child_age` — its sibling `child_min_age` kept the other shape |
| `picnic_tables_benches` | `picnic_tables_included` — invented a predicate absent from the Hebrew |
| `suitable_for_shabbat_observers` | `shabbat_observers_suitable` |

Aliases only converge if a term maps to the same canonical name every time;
non-determinism here means the *next* campsite creates a duplicate subject rather
than aliasing onto the first.

**The classifier no longer names anything.** Asking it for a name distinct from
the neighbours the judge had rejected was measured over 40 terms: it reordered
`weekend_min_nights` → `min_weekend_nights`, dropped words
(`stay_min_nights` → `min_nights`), and invented a direction — `dogs_entry_time`,
whose context said "from 16:00", became `last_dogs_entry_time`. A wrong name on a
real fact is as damaging as a wrong merge, and reordered names fork the alias
vocabulary. So the extractor's term — normalized, negation moved to polarity — is
the canonical name, stored with the probe embedding (one embed call, not two), and
the classifier is consulted only for a category the extractor did not supply. Name
quality is the extractor prompt's job (the canonical shape above); duplicates from
naming drift are the tolerable failure, surfaced by the per-site report and the
`ALIAS OVERFLOW` lines (subjects with more than 20 aliases) rather than repaired
by a model.

It is also told that a trailing `included` / `provided` / `available` always marks
an **amenity**, and that it must never append `_included` to a bare noun.

`test_subject_adjudication_llm.py` pins both directions, and asserts the property
that actually matters: classifying the same term twice gives the same name.

### Antonyms are decided in code, never asked about

The first production run with the section split merged all four mattress-window
bounds into `mattress_pickup_start_time` and `child_max_age` into
`child_min_age`. The judge sees `child_min_age` and `child_max_age` — near
identical strings, same predicate, same category, both plausibly one subject —
and says yes. It is wrong every time, so `naming.opposed` decides it:

```
min/max   minimum/maximum   start/end   first/last   early/late
earliest/latest   in/out   entry/exit   pickup/return
arrival/departure   open/close   before/after
```

Two names taking opposite sides of any pair are never offered to the judge. False
positives only over-split, which is the safe direction.

That failure was caught by the CONFLICTING message on the upsert, not by a test —
a repeat statement whose polarity or qualifier disagrees with the row already
written is logged loudly precisely because a silent skip hides an over-merge:

```
dropping CONFLICTING statement for subject 291
    (kept polarity=None qualifier=15, dropped polarity=None qualifier=20)
```

### Predicates are the judge's call, not a suffix list's

`same_predicate` used to compare the trailing token of two names against a fixed
tuple (`allowed`, `required`, `time`, `fee`, …) and refuse to offer the judge any
pair whose suffixes differed. It was removed. Any suffix missing from the list
read as "bare noun", so `late_check_out_end_time` / `late_check_out_available_until`
and `late_check_out_fee_required` / `late_check_out_fee_applies` were never
compared and became separate subjects, while `late_check_out_fee_percent` and
`late_check_out_fee_applies` (both "unknown") were judged the same predicate and a
boolean was merged into a numeric — site 20 holds a `fee_percent` row whose value
is `True`. One section of one page produced nine `late_check_out_*` subjects. The
gate over-split where its vocabulary was missing and over-merged where it was
blind, and both failures were silent.

The rule now lives in `ADJUDICATE_SYSTEM_PROMPT`, applied to the actual pair with
both contexts in view: different *kinds* of question about a noun (permission,
obligation, time, price, count, limit) are different subjects however alike the
words; different *words* for the same kind (`_available` / `_allowed`,
`_until` / `_end_time`, `_applies` / `_required`, `_included` / bare noun) are one
subject. The judge is also told that identical contexts are not evidence of
sameness: one list item can say a field kitchen exists *and* that it has no gas,
and a reproduced run showed the judge merging `gas_stove_in_field_kitchen` into
`field_kitchen` on exactly that signal, with the counter-example already in its
prompt. The prompt carries that case now.

When the judge rejects every offered neighbour, those neighbours go to the
classifier (`classify(near=…)`), which is asked to choose a canonical name that
stays clear of them by adding the one distinguishing facet the context supports.
The original term is kept as an alias either way, so later alias hits still land.

The repo rule behind this is `.cursor/rules/llm-decides-semantics.mdc`: do not
decide meaning by comparing strings to a constant list. `opposed()` is the one
sanctioned exception, kept on evidence: with it off, the 30B judge merged 4 of
6 direction pairs (`child_max_age` → `child_min_age`, `gate_close` → `gate_open`,
…) even with contexts stating the direction; the 235B merged none. The guard can
only over-split, which is the tolerable failure, so it stays in front of the
judge. A `direction` column remains the structural fix if the name is ever to
stop carrying min/max.

**The judge runs on the 235B, not the 30B.** On 13 pinned cases the 30B merged
4 of 6 direction pairs with the current prompt and 1 of 6 with the added
direction/actor block; the 235B merged none on either prompt and missed no true
synonym. It costs 2× per token (~$0.32 vs ~$0.16 per 1000 judge calls), and judge
calls grow with the vocabulary, not with ingest volume, so a full re-population
is cents. The block was added as well. (experiments.md 2026-09-04 §5; the guard
decision above is §4.)

**The judge sees what each side states, and its match is gated on its own
confidence.** With both original sentences already in view, the 235B merged a
30-person group minimum into an 80-person one, a site-wide `electric_hookup`
into the caravan-pitch hookup, and a Friday 16:00 closing time into the weekday
17:00 — three for three, systematic. Shown one `states:` line per side (the
term's polarity or number from its statement; a candidate's from its existing
`campsite_rules` rows, the current page marked "same page") plus one prompt
sentence — two statements from one page that state different numbers are two
facts; across campsites a different number is normal — it rejected all three,
and none of the true merges with differing counts (1 vs 4 carts, 16 vs 100
mattresses) moved. Asked for a `confidence` as well, every right answer in 87
probed calls came back at 0.95 and every wrong merge at 0.30–0.85, so
`pick_match` returns a match only at `MATCH_MIN_CONFIDENCE` (0.9) or above; the
one true merge below it, `picnic_table` → `picnic_tables_and_benches` at 0.80,
is the missed merge this project tolerates. A reply without a confidence is
accepted as before. The trace records the confidence on a merge, and a match
the gate refused reads `ADJUDICATOR said 'x' at confidence 0.30 < 0.9: rejected`.
(experiments.md 2026-09-04 §10, §12, §14; `test_subject_judge_confidence.py`.)

**Every upsert collision is explained, advisorily.** A collision means the
extractor or the judge got something wrong; `rules_ingest/explain.py` hands both
sides — names, sections, values, sentences, resolver outcome — to the 235B and
asks for the cause (extractor wrong name / wrong value / hallucination, judge
over-merge, true duplicate), which side is right and the fix. Probed cold it
got every judge-side collision right and one of eight extractor-side ones,
because it did not know that an alias hit involves no judge, the naming shape,
or that הונגשו means "made accessible"; its prompt now states all three. The
answer is printed under the collision in the terminal summary and in the run
report, tagged `conflict_explainer` in the cost log, and nothing acts on it.
(experiments.md 2026-09-04 §15; `test_rules_conflict_explainer.py`.)

**The classifier stays on the 30B.** Moving it with the judge broke a pinned
test: asked for the category of a bare `dogs_allowed`, the 235B answered
"amenity" 9 times in 10 on one run and "rule" 4 of 4 on another — unstable, not
merely wrong — where the 30B was 20/20 "rule". The two models are therefore
separate constants (`MODEL` for `pick_match`, `CLASSIFY_MODEL` for `classify`).
The classifier's category is only consulted when a caller passes none, and every
production caller passes one, so today this is a test-suite guarantee rather than
a production path. Temperature 0 on Nebius is not a determinism guarantee for
either model; live-model tests should expect occasional flakes and the judge
grid in §5 should be repeated before its 0/6 is relied on. (experiments.md
2026-09-04 §6.)

**Live Nebius clients are off in pytest.** Scraper/ingest code requests an
`OpenAI` client from one factory, `make_nebius_openai_client` — that is the
only `OpenAI()` constructor in that tree. The factory calls `ensure_live_llm()`:
allowed in production, denied in tests unless `@pytest.mark.llm`, in which
case it raises `LiveLlmDisabled`. LangGraph chat models are a separate path
(`ChatOpenAI` via `make_agent_chat_model`) and are not gated here.

The per-site report printed by the rules ingest lists every subject a page
touched, which term reached it by which path (alias / merged / existing /
inserted), and every upsert collision with both phrasings — the evidence the
above was diagnosed from. Since 2026-09-04 each `scrape-rules` run also writes
the same evidence as one Markdown file, `reports/rules_ingest/<start time>.md`
(`RULES_REPORT_DIR` overrides the folder; git-ignored like the cost log): per
page, every merge with the sentence each side was read from, the new subjects
and their sentences, the upsert collisions, the resolver drops, and the run's
cost by role and model (`rules_ingest/report.py`). The terminal scrolls past;
the file is what you re-read when a subject looks wrong a week later.

A qualifying word **anywhere** in the name is narrowing, not just a prefix. A live
run collapsed `gas_in_field_kitchen` into `field_kitchen` (losing "no gas"),
`early_arrival_parking_allowed` into `early_arrival_allowed` (opposite polarities,
so the surviving row claimed the opposite of the source), and two
`late_check_out_*_allowed` variants into one. The judge is now given those exact
cases, and a repeat whose polarity or qualifier disagrees with the row already
written is logged as CONFLICTING rather than as a quiet duplicate.

Word order is likewise not a new subject: `child_min_age` and `min_child_age`
merge.

### Conflicts are filed, and one action is automatic

An upsert collision -- two statements from one page on one subject -- always
marks an upstream error, and the run report alone let it scroll past. Every
collision is now diagnosed by the 235B and written to `conflict_cases`
(migration `031`): both sides verbatim (names, sections, values, sentences, how
the resolver placed each), the cause, which side is right, the explanation, and
the action taken. `status` is `open` until a person marks it `reviewed`.

The resolver may take exactly one action on its own, `rename_new`: the new
statement was a different fact folded into the old subject -- a judge
over-merge, or an alias hit on a bare name missing its qualifying word -- so it
gets its own subject, the alias the merge created is released from the old one
(never its canonical `aliases[1]`), and the statement is written. The old
subject and the kept row are never touched; if a subject of the new name already
exists the statement joins it. Everything else is `none`: a true duplicate, a
hallucination, a rate-specific line, or a kept row that is itself wrong -- filed,
not fixed.

Why only that one. Probed on the 14 distinct collisions of three runs
(experiments.md 2026-09-05 §16-§17), the model diagnosed 25 of 26 correctly and,
whenever the collision really was two facts, chose `rename_new` with a usable
name; offered four other actions (drop, rename the old subject, move the kept
row, add a count) it never chose the two that recover a misfiled kept row and
named badly in 6 of 22 cases. Its confidence said 0.95 on right and wrong alike,
so nothing gates on it; the field is recorded. `validate` allows a name that is
already an alias of the old subject (that is the merge being undone) and refuses
only the old subject's own name. Resolution runs after the page's own commit,
in its own transaction. `test_rules_conflict_resolver.py`,
`test_conflict_cases_db.py`.

### A property stated about a list goes into every name on it

`הונגשו בחניון הלילה: חניה, שירותים, מקלחות, …` says these were made
accessible. The extractor used to emit the bare nouns; each alias-hit the counted
amenity from the list above it and was refused at the upsert, and the
accessibility fact was gone. The prompt now carries the rule with a worked
example — `accessible_parking`, `accessible_toilets`, `accessible_showers` … as
amenities, never the bare noun — and held on two re-runs with the neighbouring
sections unchanged (experiments.md 2026-09-04 §11, §13).
`test_rules_extraction_accessibility.py` pins it. The same sentence's
`שתי חושות` (two huts) was misread as a fountain and as senses — a translation
error, not a naming one — so the prompt carries a glossary line: חושה is a hut,
an accommodation unit, and `שתי חושות` in an accessibility list is
`accessible_huts`.

### Every statement must assert something

A statement with neither a polarity nor a qualifier is discarded before
resolution. `שעות פתיחה: על פי צורך` used to land as
`service_center_on_demand` with `polarity=NULL, qualifier=NULL` — a permanent
subject no query can use, and one later terms could be merged into. The prompt
now says a fact with no number must be named as something answerable and given a
polarity:

```
visitor_service_center        amenity  true      the centre exists
service_center_regular_hours  rule     false     its hours are "as needed"
```

The drop is printed, so a fact the schema cannot hold is visible rather than
silently stored as an empty row. A sentence that only points at hours published
elsewhere is the same kind of non-statement, even when the model gives it a
polarity (`entry_allowed` true). The prompt now says emit nothing:

```
ניתן להגיע לחניון הלילה בהתאם לשעות הכניסה המפורסמות באתר  -> nothing
you may only enter during visiting hours                    -> nothing
```

### Natural setting is an amenity

The unit tooltip prompt already generalised a named place (`מול הכינרת` →
`near_the_kineret` + `near_a_lake` + `near_water`). The site-level extractor did
not, so Akhziv's `בקרבת הים` was skipped as brochure (experiments.md 2026-09-08
§4). Sea, desert and forest are amenities like any other — retrieve already
searches that shelf — not a fourth category. The site prompt now carries the
same keep-and-generalise rule.

### `שעות כניסה ויציאה` is cut in half before extraction

Handed the whole 674-character paragraph the model summarises it: 7 facts, and it
skips the early-arrival sentence entirely in 4 runs out of 5. Cut between arrival
and departure it returns 11 facts *every* run, with every number attached.
Measured over 5 interleaved runs per strategy:

| strategy | statements/run | facts in all 5 | numbers/run |
|---|---|---|---|
| whole, 674 chars | 7, 10, 7, 7, 7 | 4 | 5 |
| **arrival \| departure**, 178+495 | **11 ×5** | **8** | **6** |
| char midpoint, 325+348 | 10 ×5 | 8 | 3 + a duplicate |

The midpoint cut is the cautionary one: it scores *higher* on a naive stability
metric while losing the 50% late fee and the 17:00 cut-off, because it falls
between "vacate by 12:00" and the sentence that qualifies staying past it. **Where
you cut decides what survives**, so `_split_topics` finds the boundary by topic —
the first line mentioning יציאה / עזיבה / לפנות / להישאר — not by length.

The apparent 47% instability of the winning split is entirely naming: group the
subject names by the fact they describe and every group is 5/5. That is the alias
layer's job, not the chunker's.

Two caveats worth keeping: the same section measured 22%, 88%, 33% and 29% stable
across four probes on identical input, so trust the ordering and not the absolute
figures; and the first comparison ran the strategies sequentially and pointed the
*wrong* way — between-session drift exceeds the effect, so the strategies must be
interleaved round-robin. `temp/section_split_probe.py` does that.

### One site can be two campsites, and config says which

Akhziv's amenity section lists `חניון צפוני` and `חניון דרומי` separately, each
with its own counts. `campsite_rules` is keyed
`(campsite_id, accommodation_type_id, subject_id)` with no room for a subcamp, so
the southern list collides with the northern one and is dropped as CONFLICTING —
today every stored Akhziv count is a northern one and the southern list is gone.

**Which sites are split is configuration, not detection.** `config.json` carries
a `subcamps` block keyed by campsite URL:

```json
"subcamps": {
  "<akhziv url>": [
    {"heading": "חניון צפוני", "aliases": ["חניון הצפוני", "אכזיב צפון"],
     "unit_name_contains": ["חניון צפוני"]},
    {"heading": "חניון דרומי", "aliases": ["חניון הדרומי", "אכזיב דרום"],
     "unit_name_contains": ["חניון דרומי"], "default_units": true}
  ]
}
```

Keyed by URL because ids are assigned at discovery and do not survive a wipe,
while the URL is the site's identity and already `discover_sites`' upsert key.
Being in the repo, it also survives repopulating an empty cloud database, which
derived data would not.

Detecting it instead was measured and rejected. The candidate signal — does the
`אפשרויות לינה` panel carry subcamp headings — works on all 18 sites today
(`temp/subcamp_detect_probe.py`: Akhziv splits into exactly two, nothing else
splits, including the riskiest case). It is still the wrong mechanism, because
**it answers a different question**. Nahal Amud has three named sub-areas —
`מתחם ראשי / משני / הדס` — and must *not* be split: its counts are inline in one
list, nothing collides, nothing is lost. Akhziv must be split because it has two
complete parallel lists. That difference is about how a site is run; that it
currently shows up as "panel headings vs parenthesised counts" is an authoring
coincidence. The costs are also asymmetric — a false negative is the status quo,
while a false positive puts phantom campsites in search results and partitions a
real site's rules onto subcamps that do not exist.

So the detector stays as a **warning** during discovery: a site that looks split
but is absent from the config, or is in the config but no longer looks split,
gets printed. It never restructures anything.

`temp/check_subcamp_config.py` validates the block against the database and the
live page — the URL must match exactly one campsite, every heading must still
appear in the lodging panel, and every alias must occur somewhere on the page.
That last check earned itself immediately: two aliases carried over from the
experiment (`החניון הצפוני`, `החניון הדרומי`) appear nowhere, and were redundant
besides — any text containing them contains the shorter form. A config that is
quietly wrong produces no error anywhere; the site simply never splits.

`unit_name_contains` routes booking unit types to a subcamp by name, and exactly
one subcamp carries `default_units` for the rest. Akhziv's two tent pitches name
their subcamp; its four `חושה` types name none, and no prompt can place them
because their booking data never mentions one. The info page's accessibility line
does — `אכזיב דרום: … שתי חושות …` — so they default south. That is a judgement
recorded as data, which is the point of it living here.

`unit_owner` (`rules_ingest/subcamps.py`) applies it: a booking unit whose name
contains one of a subcamp's `unit_name_contains` strings belongs to that subcamp,
anything else to the `default_units` one, and on an unsplit site to the site
itself. Everything downstream of that follows the owner rather than the site —
the `accommodation_types` row, its `campsite_rules`, its image urls, the
`availability` row, and the delete that replaces a night's snapshot, which now
spans the parent and every child (`site_ids`, not `site_id`).

**Flagged as provisional.** Substring-matching a booking unit name is the weakest
part of the subcamp design: it works because Akhziv's operator happens to put the
subcamp in two unit names, it will not survive a rename, it does not generalise to
a second split site distinguished some other way, and it fails *silently* — a
config that stops matching routes everything to the default and looks fine. The
replacement worth building is a reconciliation against the info page's per-subcamp
lodging panel, which states the split in its own right; until then the strings at
least live in `campsites.subcamp` (seeded from `config.json`) rather than in code,
and `test_subcamp_unit_routing.py` pins the split at two units north, four south.
There is a `FIXME(subcamp-routing)` on the function.

It also produced the first truncation: ~30 amenities with Hebrew evidence spans
overran `MAX_TOKENS = 2500`, and the cut-off JSON surfaced as a parse error, so
the per-section `except` dropped Akhziv's entire amenity list as if the section
had been empty. The cap is now 8000 and a `finish_reason == "length"` raises a
message that says truncation, because the parse error sent us looking in the
wrong place.

### The `מידע למבקר` accordion

wrapUseInfo already carries hours, the booking iframe and the dog icon.
The accordion tab of the same name is a different source: a bullet list
of site-wide visitor rules (no generators, no glass, max consecutive
nights, reservation only, …) that is **not** in the static HTML.

It is fetched the same way as `אפשרויות לינה` — `fetch_panel(...,
title="מידע למבקר")` reads that tab's own `data-cnt` — and parsed into
one `Section`, same shape as `מה בחניון?`, then the ordinary extract →
resolve → upsert path. Measured on Akhziv after the prompt and naming
fixes: 36 stored statements, 28/29 gold lines, $0.009
(experiments.md 2026-09-08 §5; first run was 26/29, $0.014, §4).
The visiting-hours pointer is emit-nothing. Reservation is
`entry_by_reservation_only`. The 18:00 surcharge cutoff is
`late_check_in_end_time` — the 235B judge was offered `check_in_end_time`
and said no (few-shot in `ADJUDICATE_SYSTEM_PROMPT`). `cant_` / `cannot_`
in a subject name rewrite to `can_` with polarity false rather than
dropping the statement (`naming.to_positive_subject`). Open: 50% and
100% share `late_check_in_fee_percent` so the 100 is CONFLICTING;
southern-only Shabbat still merges into the site-wide row; there is
no kilogram or calendar-date qualifier unit.

A full `scrape-info --site 2` into `experiments`, then planner query 3
("near the sea" ∧ electricity, next Thursday), put Akhziv's tent
pitches in the slot list: `sea` retrieved as a site amenity. They
still fail electricity (`missing_stated_amenity`; charging points
are not a hookup). Powered inland sites still fail the sea judge.
experiments.md 2026-09-08 §6.

`test_visitor_info_panel.py` pins the parse against a captured panel.

### Breadcrumb regions are claims without a review

`just scrape-info` ends with `scrape-breadcrumbs`. The static page's
`#breadcrumbs` trail is `בית > צפון > גליל עליון > <park> > <overnight>`.
The two area links are enough: `area:north` from `/area-north/`,
`region:upper-galilee` from `/area-north/an-upper-galilee/` (the `an-` /
`as-` / `ac-` prefix is the parent area code, stripped so the embed
is `region:upper-galilee` not `an-upper-galilee`). Park and overnight
crumbs are skipped.

The review splitter emitted `{"claims": []}` on `the site is on
region:צפון at:גליל עליון`. Straight embed of the slugs against
`"not far from north"` was −0.787 / −0.605, and the claim judge said
yes (experiments.md 2026-09-09 §1). Stored claims are `area:north` and
`region:upper-galilee`. They go in `claims` with `review_id` NULL,
`is_positive` true, `confidence` 1.0, and
`notes='no review, region by breadcrumbs'`. Retrieve left-joins
`reviews` so these rows are not dropped. `clear-claims` and
`clear-reviews` leave them; `clear-info` deletes `review_id IS NULL`
claims. This is the quick ingest; it may change.

### Sources not yet ingested

The `נהלים, טפסים ומידע כללי` and `מידע לקבוצות` panels link PDFs that hold real
rules — quiet hours (`מדיניות חניונים שקטים`), group conduct, cancellation policy.
They need a PDF text dependency, which the project does not have, and the documents
are Hebrew RTL so extraction quality is unproven.

`אפשרויות לינה` and `מידע למבקר` are AJAX-loaded. Recorded here so nobody has
to rediscover it:

```
GET /ajax-handler-wp-loadmore.php
    ?action=my_repeater_show_more
    &post_id=<body[data-id], via info_site.parse.parse_wp_post_id>
    &offset=<the tab's own data-cnt, never assumed>
    &nonce=<my_repeater_field_nonce, from an inline page script>
Response: {"content": "<html>"}
```

### Known inconsistencies left alone

- `claims.embedding` and `notices.embedding` still carry `vector_cosine_ops` HNSW
  indexes while every query ranks with `<#>` (negative inner product), so those two
  indexes are never used. `subject_vectors` is fixed; the others are not.
- `rules_ingest/fetch.py` re-implements `fetch_page_html`. The reason it had to is
  gone — `info_site/scrape.py` now imports through `source.scraper.*` and runs as
  `python -m source.scraper.info_site.scrape` — but the two copies have not been
  hoisted into one place yet. That is also the sixth copy of the same
  `_ssl_context` helper in the repo; hoisting them is a separate cleanup.

## Query extractor: date_intent

The query extractor (`extractor_node`) emits a `date_intent` only;
`resolve_dates` turns it into ISO stay windows. It does not invent
calendars. That node is the **235B** (`extractor_model`, temperature 0).
`planner_node` is vacancies + amenity SQL, not a chat model.

Hebrew clocks that used to be mislabelled, and the intents they must emit:

| user said | intent |
|---|---|
| הקרוב / הזה / coming | `when=this` (this ISO week, if that weekday is still ahead) |
| הבא / next | `when=next` (next ISO week, not this week's upcoming day) |
| בעוד N שבועות / in N weeks | `weeks_from_now=N`, and no `when` |

The 30B mapped `שישי הקרוב` to `when=next` and dropped `weeks_from_now` on
`סוף השבוע בעוד שבועיים` because the prompt stated those rules and never
demonstrated them. Three few-shots plus `weeks_from_now: N` in the schema
held 25/25 at temperature 0 on Monday 7 Sep, including the Horshat Tal
query (experiments.md 2026-09-07 §1). Re-measured Tuesday 8 Sep: the
bare phrase is still 5/5, but the same 30B full prompt is **2/5** on
buried `בשישי הקרוב` (`when=next` the other three). A dates-only 30B
prompt and the 235B full prompt are both 30/30 (experiments.md
2026-09-08 §2). Extractor moved to 235B: p50 4.0s → 2.4s, ~$0.00025 →
~$0.00050 per search (experiments.md 2026-09-08 §3).

## Named campsite lookup

The extractor may emit English (`Achziv`, `Horashat Tal`) or a Hebrew
fragment (`חורשת טל`, `אכזיב`). `campsites.name` is the parks.org.il
title. `campsites.english_name` is a title from
`https://en.parks.org.il/camping/`. `scrape-sites` crawls those names
and asks the **235B once** to match each Hebrew `campsites.name` to
that closed list (subcamps append North/South). No alias table.

`lookup_campsite_by_name` ranks the query with **pg_trgm** already in
`extensions`: `GREATEST(similarity, word_similarity)` against `name`
and `english_name`, threshold 0.4, top 5. `similarity` covers
same-length typos (`Horashat` / `Horshat`); `word_similarity` covers a
short query inside a long title (`אכזיב` in the southern subcamp name).
That is character trigrams, not embeddings.

## Planner claim/rule judge

The claim retrieve gate is **−0.6** (`CLAIM_MATCH_MAX_DISTANCE`), top 5
hits per site. That is recall: campfires match `"desert"`, “pets not
allowed” matches `"pet friendly"`. Precision is a 235B call per
(query, campsite) in `planner_node` (`source/agent/claim_judge.py`).

The planner retrieve (one query embedding) loads the top-5 claims
**and** the nearest official `campsite_rules` (all categories,
including polarity false) onto each fit. The 235B judge in
`planner_node` (`source/agent/claim_judge.py`) only scores that
payload — it does not embed or search. One JSON object returns both
decisions (experiments.md 2026-09-07 §5, 35/35; a split into two
calls was not needed):

- `relevant_claims` — every claim actually about the request, including
  forbiddens. Those are the only review claims the recommender sees.
  Default: compact JSON (`TRIPPY_JUDGE_COMPACT=1`; Streamlit and
  `just run-eval` inherit this). The model returns `relevant` as
  0-based indices into the in-memory claims plus a 4–5 word `reason`.
  The planner maps those indices back to the stored claim rows before
  the recommender sees them. On E02, compact cut judge completion
  tokens 824→376 and judge wall 20.4s→13.4s; prompt grew ~3.5k from
  the compact suffix (experiments.md 2026-09-10 §6). Full eval
  `2026-09-10_131406` was compact ×5 (23/26). Quoted output is
  `TRIPPY_JUDGE_COMPACT=0` / `--no-judge-compact`. Live judge calls
  run **5 at a time** (`TRIPPY_JUDGE_CONCURRENCY`, default 5).
- `satisfies` — true iff a relevant claim says yes **or** a granting
  rule exists. A no does not veto: fridge complaints do not drop
  `refrigerator` true; "no electricity at the tent" does not drop
  `electric_hookup` true. Those nos stay in `relevant_claims` for the
  recommender. `dogs_allowed` false is not a yes for pet-friendly, but
  it also does not veto a granting claim. experiments.md 2026-09-08 §9.

`area:*` / `region:*` breadcrumb claims satisfy a request for that area
or region by name (`area:north` → north, `region:negev` → desert,
`region:dead-sea` → Dead Sea). They do not satisfy `"near the sea"` /
ליד הים because the slug contains sea/ים. Dead Sea and Kinneret are
named places, not the Mediterranean or Red Sea coast; a beach-access
review still grants. experiments.md 2026-09-10 §2 (8/8 on 235B).

Claim-only fits the judge rejects are dropped. Listing hits at amenity
**−0.7** are recall as well: the same judge runs on amenity-only fits and
drops a site when `satisfies` is false (tent-as-desert, stove-as-
electricity, caravan-bay-only hookup). experiments.md 2026-09-07 §8,
59/60. Ranking of the survivors is still the recency-weighted claim
score; a later pass can rank on the judge.

Nearest-20 retrieve at the same −0.6 gate (experiments.md 2026-09-07 §6)
put every labeled gold claim and gold rule at **rank 1**, so K=10/20 do
not recover missed gold. Rules cannot share that gate: tent subjects at
−0.719 pass −0.6 in bulk and are closer than `electric_hookup` (−0.704);
−0.8 would drop the hookup. Claim K stays 5; the rule cutoff is still
open.

The listing amenity gate is **−0.7** (`AMENITY_MATCH_MAX_DISTANCE`).
That recovers `"electricity"` → `electric_outlet` (−0.750) and
`electric_hookup` (−0.704), both in the ungated top 5, and also matches
`tent` (−0.719) for `"desert"` / `"quiet"`. The judge sifts those listing
rows (experiments.md 2026-09-07 §8).

For a site-locus ask, the claim lane and the site-amenity lane
**always both retrieve**. A slot matches if either hits (or a unit
listing). A claim must not skip `search_site_amenities`: Akhziv's
communal `מקררים (3)` was listed, but fridge complaints already
matched retrieve, so the official row never entered `why` and the
judge never granted it (experiments.md 2026-09-08 §7).

`search_campsite_rules` and `search_site_amenities` for a subcamp read
**the child's rows and the parent's, never a sister's**. Ingest writes
visitor-info onto the subcamp (Akhziv 37/38); reviews stay on the
parent. Looking only at `COALESCE(parent_id, id)` made the judge search
parent 2, which has no fridge rule. Pulling every child of that parent
would mix אכזיב דרום with אכזיב צפון.

Planner benchmark v1 (`evals/planner_v1.json`) is **26 queries**
(14 easy / 12 hard) covering dates, no-dates, prices, capacity,
amenities and rules. Amenity/rule gold is the parks.org.il page,
not `campsite_rules`. Occupancy is `experiments.availability_frozen`
(snapshot of `public.availability` on 2026-09-08, nights 7–19 Sep).
`TRIPPY_AVAILABILITY_TABLE=availability_frozen` and
`TRIPPY_TODAY=2026-09-08` so a later availability scrape cannot
move who is vacant, and `יום חמישי הבא` stays 17 Sep.
`just run-eval` first copies `public` into `experiments` except
`availability` (that table is cloned empty; occupancy is
`availability_frozen`). Then extractor then planner on every case and
writes `reports/evals/` (per-query seconds in the table, then a
**Cases** section with the extractor JSON, the planner RAG queries,
retrieved claims/rules, and the judge verdict). Each case prints
token in/out (extractor + judge) before the report is written; the
markdown has a **Tokens** table. `--model 30B` runs
extractor+judge on the 30B; `--judge-concurrency N` overrides the
default of 5 parallel live judge calls; `--no-judge-compact` quotes
claim text instead of indices. `--no-copy` skips the
refresh. Case FAIL rows are scored in the report; the process still
exits 0 unless a setup or runtime error stops the run.
Gold is mostly campsite ids (`must_include_sites` /
`must_exclude_sites`); a few cases also substring-match fit
`accommodation_type` names (`must_include_types` /
`must_exclude_types`). Every query states a party, with mixed
phrasing (אדם אחד, זוג, שלושה חברים, 4 חברים, ארבעה זוגות,
משפחה של 6). E03 keeps `לאדם` as a per-person **price** unit next
to an explicit `אדם אחד`.

`electric_hookup` on stored rows is
still three listings (caravan-bay water+power, PITCH tent power, site-wide
נקודות חשמל). After re-extract the caravan bay is
`caravan_bay_electric_hookup` / `caravan_bay_water_hookup` instead
(experiments.md 2026-09-08 §1); PITCH and site-wide keep the generic
names.

