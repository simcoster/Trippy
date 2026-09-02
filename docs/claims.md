# Review claims — splitter experiments and ingest decisions

Probe tables (anonymized): `temp/split_reviews.md`, `temp/split_claims_table_235b.md` (5 reviews / 1 call), `temp/split_claims_table_235b_single.md` (1 review / call). Hurshat Tal, five legacy Places `most_relevant` Google reviews.

## Locked (2026-09-01)

| Choice | Decision |
|--------|----------|
| Split model | Qwen3-235B-A22B-Instruct-2507 (same as amenity extract) |
| Batch size | **One review per chat call** |
| Embed | After split, one `embed([])` of all kept `text_en` (Qwen3-Embedding-8B, 1536) |
| Generic judgments | Skip in the prompt; also drop `confidence < 0.5` |
| Aspect / locus | **Not stored.** Revisit if we need structured filters or amenity-key alignment |
| Stars / author / full text | `reviews` table only — claims do not copy rating |
| Claims columns | `review_id`, `campsite_id`, `claim` (standalone rewrite, usually EN), `evidence_span` (original language), `polarity`, `confidence`, `embedding` |
| Recency | `reviews.published_at`; search joins reviews |
| Visit gate | **30B** before split. Ads / brochure / history dumps **and hiking-trail writeups** → `reviews.skip_reason = not_personal`, `skip_note`, **no claims**. Guest reports of site conditions (streams dry, crowding, paid entry) **keep**, even if they rant. Mixed stay+trail still keep. Splitter and `confidence < 0.5` unchanged. Empty text: store only, no gate. Gold: `visit_gate.json`. |

Ingest: `source/scraper/populate_reviews_and_claims.py` → `populate_reviews_and_claims(campsite_id, reviews_dict)`. Places fetch is still a separate step.

## Why 1 review / call (not 5-in-1)

Same 235B prompt, Hurshat Tal 5 reviews, ~20 claims either way. Single cost ~1.75× batch (system prompt repeated). Quality was not uniformly better, but **batch dropped bungalow rental and mattress rental** on review 4; single kept them. Batch split stream vs pools (good); single glued those once. We would rather pay for the extra call than miss a site fact.

Review 3 (leashed-dog incident) stayed **one claim** in both modes with the v1 incident rule. 30B over-split that story; do not use 30B for split.

## Why a visit gate (not claim-level extra filter)

Yehiam-style ads and history lectures still emit high-confidence "site facts" (accessible parking, 1930s raids). `confidence < 0.5` does not catch them. **30B yes/no on the whole review** before 235B; keep the row, set `skip_reason`, insert no claims. Claim split rules stay as locked above. Revisit a claim keep/drop pass only if the splitter still leaks encyclopedia rows from reviews that passed the gate.

## Why drop confidence &lt; 0.5

Batch still emitted *The place is disappointing.* (`בקיצור אכזבה`) at **0.3** with `aspect=overall` even though the prompt says skip generic judgments. The model leaked a skip-rule violation as a low-confidence row. On that table, 0.5 only dropped that junk; keepers sat at 0.8–0.95 (single-call keepers can sit at 0.6–0.7, so do not raise the floor). Confidence is uncalibrated — a junk filter, not a ranking signal.

## Why not aspect / locus yet

Embeddings retrieve by topic; `text_en` is already a standalone sentence. Aspect/locus were canonical slots for grouping, dedup, and “no water” → shower vs stream vs tap. If the splitter always names the place in `text_en`, they are optional. **Add later** if we need SQL filters, amenity-key joins, or we still get underspecified water/place claims. Polarity stays: opposite claims embed nearby.

## Split rules (kept)

- Faithful `text_en` of `evidence_span`; this review only for pronouns/feature. Do not generalize (`שירותים` ≠ facilities; `תאים` = stalls; `סוגרים` = latches).
- JSON still sends `place` (campsite name) as splitter context. Do **not** put the park name in `text_en` (hurts topic embeddings; site is `campsite_id`).
- Skip generic “place is great”; “excellent except X” → only X. Do not wrap facts in rants.
- Direct experience vs speculation: keep “staff were rude”; drop “management should open the streams”.
- Dedup same fact in one review; crowding at two spots ≠ streams dry.
- One visitor incident = one claim (dog + leash refusal + cashier + U-turn → one pets/gate claim).
- Keep specific rentable features (bungalow, mats).
- Opposite sentiments are separate rows.

## Schema

```
reviews  (campsite_id, source, author, rating, text, published_at, review_uid, skip_reason, skip_note)
claims   (review_id, campsite_id, …)  — no stars; date via join
```

Migration `014_reviews_and_claims` drops the old `claims` table (text `campsite_id`, author/date/source on the claim) and recreates it with FKs.
