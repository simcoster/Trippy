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

### Amenities live in two places

`campsite_rules` and the `campsites.amenities` / `accommodation_types.amenities`
JSONB arrays both hold `subject_vectors` ids. The planner still reads the JSONB
arrays (`source/agent/search.py` `search_stated_amenities` / `search_site_amenities`),
so `rules_ingest.db.sync_campsite_amenity_ids` mirrors site-level amenity rows into
`campsites.amenities` after each ingest. That is what finally gives the site lane
something to search — it had no writer at all before (`docs/PLAN.md`).

The duplication is deliberate and temporary. The follow-up is to point the two
search functions at `campsite_rules` and drop the four JSONB columns. Until then,
**`campsite_rules` is the source of truth for site-level amenities** and the JSONB
array is a derived cache; do not write it by hand.

### `accommodation_type_id` is always NULL today

The column exists and is indexed, but the info-page ingest never fills it: per-unit
amenities already come from the availability scrape
(`source/scraper/populate_availability.py` → `amenity_enrichment`), which reads the
booking-site tooltips. The info page's `אפשרויות לינה` panel is a second, weaker
source for the same data, so `sections.parse_sections` drops it on purpose.

That leaves per-unit rules in the older shape: `accommodation_types.check_in_time`,
`check_out_time` and the `policy_rules` JSONB (`min_weekend_nights`, `pets_allowed`,
…) — all of which `campsite_rules` can express better. Unifying them means teaching
`amenity_enrichment` to write `campsite_rules` with a non-NULL
`accommodation_type_id`, and is the natural next step.

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

### Experiments run against `test_*` tables

Production schema is not changed to try something out. `temp/rules_ingest_testbed.py`
creates `test_subject_vectors` / `test_campsite_rules` / `test_accommodation_amenities`,
seeds the vocabulary from production, and drives both ingest jobs against them:

```
uv run python -m temp.rules_ingest_testbed --reset --rules --amenities --report
```

The only production change this needs is dependency injection:

- `SubjectStore(table=..., has_context=...)` — where subjects live, and whether the
  table carries the `context` column. The table name is interpolated into SQL, so
  `SubjectStore` rejects anything that is not a plain identifier.
- `upsert_campsite_rules(table=...)`, `sync_campsite_amenity_ids(rules_table=,
  subjects_table=)`, `ingest_site(store=, rules_table=, mirror_amenities=)`.

`mirror_amenities=False` is what keeps a test run out of `campsites.amenities`,
which is production *data* even though the schema is untouched. The room-amenity
job likewise writes `test_accommodation_amenities` rather than
`accommodation_types`, whose JSONB ids would otherwise point at test subjects.

### Context: what a subject was first read from

`context` is the sentence a subject came from — `"מה בחניון?: שירותים (15 תאי
שירותי נשים ו- 15 תאי שירותי גברים)"` for a site list, `"חדר צוות: בכל חדר ...
שירותים, מקלחת מים חמים"` for a room tooltip. It is stored on the subject and shown
to both small LLMs:

- the **sameness judge** sees the term's context and each candidate's, which is the
  only way to tell a 30-stall communal block from a room's own bathroom — the names
  `toilets` and `bathroom` alone give it nothing to work with;
- the **classifier** sees it when naming a brand-new subject.

It lives only on the test table for now (production `subject_vectors` has no such
column, and the resolver omits it unless `store.has_context`). Promote it with a
migration once the testbed shows it earns its place.

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
