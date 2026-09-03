# Subject vectors and campsite rules — provisional decisions

Landed 2026-09-02 with migrations `023_subject_vectors` / `024_campsite_rules` and
the `source/scraper/rules_ingest` + `source/scraper/subjects` packages.

Everything below is a decision we expect to revisit. Read it before extending the
schema, so you know which parts are load-bearing and which are scaffolding.

## Locked

| Choice | Decision |
|--------|----------|
| Dictionary table | `amenities` → `subject_vectors`: one row per subject, `name` canonical, `aliases[1] = name`, `category` 1 amenity / 2 rule |
| Subject naming | snake_case English, carries the predicate, **always positively phrased** — `dogs_allowed`, never `dogs_not_allowed`. Enforced by the extractor prompt and backstopped by `subjects/naming.to_positive_subject` |
| Polarity | nullable `BOOLEAN` on `campsite_rules`: `True` allowed/provided, `False` forbidden/not provided, `NULL` a pure quantity |
| Qualifier | `NUMERIC` + `qualifier_unit SMALLINT`. Direction lives in the name (`min_weekend_nights`, `max_occupancy`), matching the extractor's existing `policy_rules` keys. Times of day are decimal hours: 20:30 → 20.5 |
| Alias resolution | exact `aliases @>` hit → 5-NN by `<#>` + 30B adjudicator → 30B classify + insert. Only misses cost an LLM call |
| Embedding | the **canonical name**, not the surface form, so a row's vector does not drift as aliases accrue |
| Ingest scope | site-level only, static HTML only |
| Vector op | `subject_vectors_embedding_idx` is `hnsw (embedding vector_ip_ops)` |

## Provisional — expect these to change

### Category is not a clean split

Amenities come in "provided / not provided". Rules come in "allowed / forbidden"
**and** in "only after 18". The line between them is fuzzy: `barbecue_allowed` is a
rule, `barbecue_equipment_included` is an amenity, and they are extracted from the
same sentence (`ניתן להדליק מצלה (מנגל) בציוד עצמי`). `category` is a search
convenience — it lets the planner ask only about amenities — not an ontology. If it
starts costing more than it saves, collapse it.

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

It used to be NULL on every row. `amenity_enrichment.db.write_unit_amenities` now
writes per-unit amenities as `campsite_rules` rows carrying the type's id, which
is what migration `027` backfilled and what the enrichment path writes from here
on. The info-page ingest still only writes site-level rows: the info page's
`אפשרויות לינה` panel is a second, weaker source for per-unit data, so
`sections.parse_sections` drops it on purpose.

A subject the extractor puts in both the included and not-included list for one
unit collides on `campsite_rules_scope_subject_key`. `write_unit_amenities`
writes the not-included pass second, so the stricter reading survives.

Still in the older shape, and the next thing to unify:
`accommodation_types.check_in_time`, `check_out_time` and the `policy_rules`
JSONB (`min_weekend_nights`, `pets_allowed`, …), all of which `campsite_rules`
can express better.

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
   `ensure_amenities` pins its own path to `AMENITY` for the same reason.

   The filter is **in the SQL**, not applied afterwards. Post-filtering spends all
   five nearest-neighbour slots on the wrong category and can leave nothing at all
   to consider — the first traced run offered zero candidates for
   `barbecue_allowed` because its five nearest were every one of them amenities,
   so it never saw a single existing rule. The partial indexes from migration
   `025` serve exactly this query. Note this applies to *resolution* only: the
   planner's lanes stay unfiltered and see both categories.
2. **`naming.same_predicate`, in code.** The trailing token of a subject name *is*
   its predicate, so `barbecue_allowed` (may I grill?) and
   `barbecue_equipment_included` (is a grill provided?) cannot be the same subject
   whatever the model thinks. Candidates with a different predicate suffix are
   filtered out before the adjudicator ever sees them — which also saves the call.
   Same rule kills `late_check_out_fee` → `late_check_out_available` and
   `equipment_rental_deposit_required` → `rental_equipment_available`.
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

Production schema is not changed to try something out. The testbeds create their
tables in a separate `experiments` schema — never in `public`, and never via an
Alembic migration — and their connections set
`options="-csearch_path=experiments"`:

```
uv run python -m temp.rules_ingest_testbed --reset --rules --amenities --report
uv run python -m temp.split_campsites_testbed --reset --guards --ingest --report
```

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

`ensure_amenities(contexts=...)` carries the room tooltip down the accommodation
path, so a room's own `bathroom` arrives with the sentence that distinguishes it.

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

Bypassing the classifier when the extractor supplied a category was tried and
reverted — it removed the one place that cleans a genuinely bad name. Instead the
prompt is strict: **return the term unchanged unless there is a real problem**
(a misspelling, a plural, a negation, or a name stating no predicate). Reordering,
adding and dropping words for style are forbidden. `picnic_tables_and_benches` →
`picnic_tables` still happens; `child_max_age` and
`suitable_for_shabbat_observers` now survive untouched.

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

### A suffix means different things on a rule and on an amenity

`towels_included` and `towels` are one subject — whether it is provided is what
`polarity` records. The same for `_provided` and `_available`. `same_predicate`
strips those before comparing, so the pair reaches the judge instead of being
blocked in code, and the judge is told the rule explicitly. Before this,
`electric_hookup` and `electric_hookup_included` were two rows for one thing.

But `available` is not one word. On an amenity it means *supplied*; on a rule it
means *permitted*, and `late_checkout_allowed` / `late_check_out_available` are one
rule. Stripping it there produced a false split instead — the trace caught it:

```
'late_checkout_allowed' (rule). NN: [late_check_out_available -0.788[predicate], ...].
   considered 0. INSERTED.
```

So `same_predicate` takes the category. On a rule, `allowed` / `permitted` /
`available` all collapse to one `permission` predicate; on anything else the
provision words are stripped. `available` sits in both lists on purpose.

The cost: `barbecue_equipment_included` vs `barbecue` is no longer blocked
structurally and now depends on the judge reading the nouns. It is pinned in
`test_subject_adjudication_llm.py` in both directions.

A qualifying word **anywhere** in the name is narrowing, not just a prefix. A live
run collapsed `gas_in_field_kitchen` into `field_kitchen` (losing "no gas"),
`early_arrival_parking_allowed` into `early_arrival_allowed` (opposite polarities,
so the surviving row claimed the opposite of the source), and two
`late_check_out_*_allowed` variants into one. The judge is now given those exact
cases, and a repeat whose polarity or qualifier disagrees with the row already
written is logged as CONFLICTING rather than as a quiet duplicate.

Word order is likewise not a new subject: `child_min_age` and `min_child_age`
merge.

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
silently stored as an empty row.

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

### Sources not yet ingested

The `נהלים, טפסים ומידע כללי` and `מידע לקבוצות` panels link PDFs that hold real
rules — quiet hours (`מדיניות חניונים שקטים`), group conduct, cancellation policy.
They need a PDF text dependency, which the project does not have, and the documents
are Hebrew RTL so extraction quality is unproven.

Both those panels and `אפשרויות לינה` are AJAX-loaded. Recorded here so nobody has
to rediscover it:

```
GET /ajax-handler-wp-loadmore.php
    ?action=my_repeater_show_more
    &post_id=<body[data-id], via info_site.parse.parse_wp_post_id>
    &offset=<0-based panel index>
    &nonce=<my_repeater_field_nonce, from an inline page script>
Response: {"content": "<html>"}
```

Everything currently in scope is server-rendered, which is why
`rules_ingest/fetch.py` is a plain GET.

### Known inconsistencies left alone

- `claims.embedding` and `notices.embedding` still carry `vector_cosine_ops` HNSW
  indexes while every query ranks with `<#>` (negative inner product), so those two
  indexes are never used. `subject_vectors` is fixed; the others are not.
- `rules_ingest/fetch.py` re-implements `fetch_page_html` because
  `source/scraper/info_site/scrape.py` still uses bare `from info_site...` /
  `from amenity_enrichment...` imports that only resolve under the justfile's
  `PYTHONPATH`, so it cannot be imported by module path. That is the sixth copy of
  the same `_ssl_context` helper in the repo; hoisting them is a separate cleanup.
