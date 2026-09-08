# Planner benchmark v1

26 Hebrew queries (14 easy / 12 hard) against a **frozen occupancy
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
uv run python -m source.eval.run --ids E01,H02
```

Pins `TRIPPY_SCHEMA=experiments`, `TRIPPY_AVAILABILITY_TABLE=availability_frozen`,
and `TRIPPY_TODAY=2026-09-08` from the JSON. Each case goes through
**extractor then planner** (not the light cleaner). Writes
`reports/evals/<timestamp>.md` plus a JSON dump.

The full 26 is tens of 235B judge calls (tens of minutes). `--ids` is the
smoke path.

Requires `just setup-experiments freeze-availability` first.

## What the set covers

| Dimension | Easy | Hard |
|---|---|---|
| Dates | E01, E09, E13 | H05 buried Friday |
| No dates | E14 | H11 |
| Prices | E03, E12 | H04 sea + ≤200 for 3 |
| Capacity | E02, E10 | H08 hut occ=4 vs 6 |
| Amenities | E04 sea, E05 desert, E06 fridge, E08 showers, E11 AC | H01 sea∧power, H02 sea∧fridge, H03 room fridge, H06 OR, H07 tent power not caravan |
| Rules | — | H09 weekend min-2, H10 dogs forbidden, H12 south Shabbat only |

Queries are in `evals/planner_v1.json`.
