# Planner benchmark v1

27 Hebrew queries (15 easy / 12 hard) against a **frozen occupancy
snapshot**. Ingest and retrieve can change; who is vacant that night
does not.

Gold for amenities and rules is the **parks.org.il camping page**
(fetched 2026-09-08), not `campsite_rules`. Occupancy and “is this night
bookable” come from the booking dump frozen at the same time — the info
page has no calendar.

## Freeze

```text
just setup-experiments freeze-availability
```

Copies `public.availability` → `experiments.availability_frozen`
(226 nights, 2026-09-07 … 2026-09-19, 14 vacant parks). `copy` does
not drop this table.

## Run

```text
just run-eval
just run-eval -- --ids E01,H02
just run-eval -- --limit 2
just run-eval -- --no-copy
just run-eval -- --model 30B
just run-eval -- --judge-concurrency 1
just run-eval -- --no-judge-compact
just run-eval -- --recommender
just run-eval -- --recommender --from-planner reports/evals/<stamp>.json
uv run python -m source.eval.run --ids E01,H02
```

Judge defaults match production: compact JSON and 5 parallel live
calls. `--judge-concurrency` / `--no-judge-compact` override.

First copies `public` → `experiments` except `availability` (occupancy
stays `availability_frozen`). `--no-copy` skips that refresh.

Pins `TRIPPY_SCHEMA=experiments`, `TRIPPY_AVAILABILITY_TABLE=availability_frozen`,
and `TRIPPY_TODAY=2026-09-08` from the JSON. Each case goes through
**extractor then planner** (not the light cleaner). Writes
`reports/evals/<timestamp>.md` plus a JSON dump. The markdown table
has per-query seconds; **Cases** lists extractor constraints, planner
queries, RAG claims/rules, and judge verdicts. `--recommender` also runs
the Kimi-K3 picker after the planner and dumps 1–2 cited recs (not scored).
Each rec also records first-chunk and first-spoken TTFT (`ttft_chunk` /
`ttft_spoken`), printed next to `recommend=` on the CLI.
Every dump also stores a compact recommender `pack` per case (query,
extractor JSON, compact fits with why / review_claims / rules /
claim_judge). `--recommender --from-planner reports/evals/<stamp>.json`
replays only the picker from those packs — no extract, judge, copy, or
DB. `--from-json reports/evals/<stamp>.json` rebuilds the markdown (recommendation
and cost tables) without re-running. A mixed score is still
exit 0; only a setup or runtime error fails the recipe.

Gold is campsite ids (`must_include_sites` / `must_exclude_sites`) on
every dated case. Six cases also substring-match fit type names:
`must_include_types` on E10, E11, H03, H05; `must_exclude_types` on
H07, H08. Every query states a party. E03 `לאדם` is per-person
price, not party (`אדם אחד` is).

The full 27 is tens of 235B judge calls (tens of minutes). `--ids` picks
named cases; `--limit 2` is the first 2 easy and first 2 hard.

Requires `just setup-experiments freeze-availability` once so the frozen
table exists. `copy` does not drop it.

## What the set covers

| Dimension | Easy | Hard |
|---|---|---|
| Dates | E01, E09, E13 | H05 buried Friday |
| No dates | E14 | H11 |
| Prices | E03, E12 | H04 sea + ≤200 for 3 |
| Capacity | E02, E10, E15 | H08 hut occ=4 vs 6 |
| Amenities | E04 sea, E05 desert, E06 fridge, E08 showers, E11 AC | H01 sea∧power, H02 sea∧fridge, H03 room fridge, H06 OR, H07 tent power not caravan |
| Rules | — | H09 weekend min-2, H10 dogs forbidden, H12 south Shabbat only |

Queries are in `evals/planner_v1.json`.
