# TODO — 2026-09-07

## 1. Reviews by relevancy

Done in code: `just scrape-reviews` stores `[5 newest] + [5 most_relevant]`
on `reviews` only (overlap dropped; `--most-relevant` is a no-op).
`just populate-claims` classifies `is_relevant IS NULL` and writes claims.
`just clear-claims` wipes claims and nulls `is_relevant`, not review rows.
Apply `just update-tables` then run scrape/populate when you want data.

## 2. Pricing search

Planner already quotes `price_per_night` and honours `price_per_night <= N` (the barbecue ≤₪250 query kept two cheap slots). Verify end-to-end:

- Extractor turns a real budget into a numeric constraint (not a semantic “cheap”).
- Quoted rate matches the stay’s weekday vs weekend/holiday period.
- “Not too expensive” must not invent a cap (it did: ≤₪500).

## 3. Disqualify caravan parking unless the user has a caravan

`עמדת חניה לקרוואן פרטי` was offered to families who asked for a campsite night, a wheelchair room, or a cabin. A caravan bay requires the guest to bring a caravan.

Do not solve this with a hardcoded name list — the planner/extractor should treat “needs own caravan” as a product constraint and drop those types unless the query says they have one.

## 4. Recommender node

Graph already has `recommender_node` (235B) after the planner. Today’s eval stopped at the planner. Turn it on for family/group searches and check:

- It uses `fits` / `why` / review claims, not invented amenities.
- It does not recommend caravan bays or tent pitches when the user asked for a room.
- Empty planner results get an honest “nothing this weekend”, not a hallucination.

Local harness: `uv run streamlit run scripts/streamlit_chat.py` with heavy path through recommender.

## 5. Cloud — jobs vs bot (phase 1: Streamlit)

Two process types, not one box:

| Process | Role |
|---------|------|
| **Bot** | Streamlit chat → LangGraph. Not Telegram unless wiring it is trivial (`main.py` webhook already exists). |
| **Jobs** | `just scrape-*` CLIs in a second Compose service, triggered from Streamlit. |

Phase 1 is Streamlit + a jobs worker + **self-hosted** Postgres on Nebius
(`docs/deploy.md` runbook, `docs/deploy_design.md` why). Telegram can wait.
`docs/scaling.md` is the later target.

## 6. README with design choices

Root `README.md` is empty. Write a short one that a new clone can follow: what Trippy is, how to run (Compose, `just`, Streamlit), and the load-bearing choices (with pointers into `docs/design.md` / `docs/claims.md` / `docs/scaling.md` — do not duplicate those files).
