# ReAct vs the graph agent

Logged 2026-09-08 from an architecture discussion. **Not shipped. No
spike run.** Ingest, Postgres, availability, and query-time RAG stay as
they are in every option below. This is only about the *agent loop*
that sits on top of them.

The product is a chatbot, not one-shot search → recommend. The turns
that have to work:

- “why did you pick this?”
- “wait, what was that first one again?”
- “that’s too expensive, anything cheaper?”

PLAN.md §4 already parked conversation state (`last_recommendations`,
structured prefs vs full transcript) as after the request→recommendation
path. This note is the agent-loop choice that path implies.

## What does not change

Scrapers, `subject_vectors` / `campsite_rules` / `claims` / `availability`
/ `list_prices`, Qwen embeddings, amenity gate **−0.7**, claim gate
**−0.6**, `quote_night`, `resolve_dates`. Query-time RAG is not optional
in any of the serious options.

Amenity and claim search are **not** `ILIKE`. They embed the user phrase
with Qwen3-Embedding-8B, then rank with `<#>`. Prices are not a column
on `availability`; `quote_night` prefers a per-unit `any` rate, else
adult×N, weekday vs weekend. A generic SQL tool will get both wrong.

## The current graph

`source/agent/graph.py`:

```text
light (keep/drop, trivial short-circuit)
  → extractor (235B, temperature 0) → date_intent JSON
  → planner (no chat model): search_open_slots + amenity/claim RAG + claim_judge
  → recommender (235B) writes Hebrew from fits JSON
```

Tools are invoked **imperatively** in `planner_node`. They are not bound
on the chat models: binding them made Qwen return empty `content` plus
`tool_calls`, which surfaced as blank agent replies.

`search_open_slots` returns up to **80 slots per date window**. A slot
is one `(campsite, accommodation_type)` that has a one-night vacancy for
**every** night of the stay (`GROUP BY` + `HAVING COUNT(DISTINCT
start_date)`), plus a quoted `price_per_night`. Not 80 campsites and not
80 nights. Up to four date windows (`MAX_DATE_WINDOWS`) are concatenated,
so the amenity filter can see more than 80 rows.

The recommender is told to pick only from `fits`. It is passed all
`HumanMessage`s and all planner `ChatMessage`s, and **drops** spoken
recommendations (`AIMessage`). Every user turn re-runs extractor →
planner → recommender. There is no `last_recommendations` list, no
merge of “cheaper” onto existing constraints, and no route that answers
“why?” without a new search.

Streamlit keeps `graph_messages` in session; Telegram keeps an in-memory
list per `chat_id`. Neither is the structured state PLAN.md sketched.

## Options considered

### A. Stuff visitor-info pages into the recommender after availability

Skip query-time embeddings, retrieve, and the claim/rule judge. After
open slots, dump each candidate’s official page plus the user query.

**Would execute** for site-level visitor-info *text* (Akhziv’s `מידע
למבקר` panel is ~1k tokens). Qwen3-235B-A22B-Instruct-2507 is **262k**
native context; 25 panels (~25k) plus 80 slot summaries (~4–8k) fit.
Raw parks.org.il HTML does not (hundreds of k of nav/scripts).

**Does not replace RAG.** Per-unit amenities live on booking tooltips,
not that panel. “Quiet”, “good for kids”, “in the desert” live in
review claims. Notices are a third source.

**Lost-in-the-middle is a quality cliff, not a window overflow.** Qwen’s
RULER needle scores stay high past 32k; that is “find a unique sentence.”
Comparing dozens of near-identical Hebrew rule lists is the opposite.
~5–8 structured cards are fine; ~15+ similar pages start mixing
attributions (wrong park cited, a prohibition in the middle missed);
80 pages make “the first one” undefined. The 262k window is still
happy. Cost at `$0.20`/MTok input is small next to that failure.

Rejected as a RAG replacement. Fine only as extra *evidence* on an
already-short `fits` list.

### B. Hosted Claude Code + Postgres MCP

Keep ingest and the DB. Replace the graph with a hosted Claude Code
session, skills for follow-ups, and MCP so it can query Postgres
(including availability and the RAG tables).

**The chatbot shape is right** (session memory, “why?” without a new
search, merge “cheaper”). **The runtime and the MCP surface are wrong.**

- Claude Code is a repo coding session, not a Telegram webhook. Cold
  start re-discovers the schema; one session ≠ N concurrent `chat_id`s;
  latency is tens of seconds to minutes.
- Generic Postgres MCP cannot embed a query or run `quote_night`. The
  MCP to expose is `search.py` + `resolve_dates`, read-only.
- Skills can quote the judge prompt and the date rules; they cannot
  stop an agent from writing a plausible vacancy SQL that skips the
  every-night `HAVING`.

Useful as how we would *write* skills. Not how users talk to the bot.

### C. ReAct agent, Qwen 235B backbone, existing search as tools

Same split as B, with the model we already pay for, inside LangGraph
(or any tool loop). **ReAct vs LangGraph is a false choice** — LangGraph
can host a ReAct node. Today’s graph is a DAG; ReAct is “the 235B
decides whether to call a tool or just answer.”

Typed tools, not SQL:

- `resolve_dates`
- `search_open_slots`
- `search_stated_amenities` / `search_site_amenities` / `search_review_claims`
  (embed + gates inside the tool)
- optional `judge_site_request`
- read-only, no `execute_sql`

Checkpointer per conversation; `last_recommendations` in state.

This is the only option that matches “chatbot on top of the current
DB” without changing products. It is also the one that re-opens
measured failures unless the loop is constrained.

## The buried-date bug (why ReAct cannot own calendars casually)

Date extract is not “the 235B is big, it will get Friday.” It is a
narrow, flaky Hebrew clock, measured twice.

**Rules that must land in `date_intent` before `resolve_dates`:**

| user said | intent |
|---|---|
| הקרוב / הזה / coming | `when=this` (this ISO week, if that weekday is still ahead) |
| הבא / next | `when=next` (next ISO week) |
| בעוד N שבועות | `weeks_from_now=N`, and no `when` |
| weekend / סופ״ש | Friday night only (`nights=1`) |

**2026-09-07 §1.** 30B full extractor, temperature 0, three few-shots:
**25/25**, including Horshat Tal `בשישי הקרוב` → Friday 11 Sep
(`when=this`). Decision at the time: keep the 30B, do not split dates
into a second call.

**2026-09-08 §2.** Same 30B full prompt, today frozen Tuesday 8 Sep, 6
prompts × 5 trials. **27/30.** The only misses are Q2 — the Horshat Tal
query with `בשישי הקרוב` buried next to a named park and a ₪400 cap:
**2/5** `when=this` (11 Sep), **3/5** `when=next` (18 Sep). Bare
`בשישי הקרוב` on the same model is **5/5**. 235B full prompt **30/30**.
30B dates-only prompt **30/30**.

**2026-09-08 §3.** Extractor shipped to 235B. p50 4.0s → 2.4s,
~$0.00025 → ~$0.00050 per extract. `planner_node` stays SQL.

ReAct at temperature 0.7, mid-conversation, with amenities and prices
in the same prompt, is the **30B-full** failure mode: the clock is
buried. A skill that restates the table is what the 7 Sep prompt
already did, and 8 Sep still flipped הקרוב → הבא when the phrase sat
inside a longer ask.

**Constraint on option C:** keep a dedicated extract (current 235B node,
or a dates-only tool the agent **must** call before `search_open_slots`).
Do not let the ReAct backbone emit ISO dates or invent `when`.

## Other measured constraints that apply to ReAct

**Claim/rule judge (2026-09-07 §5, §8).** Retrieve is recall: campfires
match `"desert"`, “Pets are not allowed” matches `"pet friendly"`,
`tent` at −0.719 matches desert/quiet, caravan-bay hookup matches
`"electricity"`. Precision is one 235B JSON call per (query, campsite):
`relevant_claims` + `satisfies`. 35/35 on the claim pack; **59/60**
when the same judge sifts amenity −0.7 listing hits (the miss was gold
string-match, not the model). Letting ReAct “reason” over raw hits
re-rolls that 59/60 in an uncontrolled loop. Keep `judge_site_request`
as a tool, or keep the current planner pass.

**`bind_tools` empty replies.** Tools bound on extractor/recommender
made Qwen emit `tool_calls` and empty `content`. A ReAct node is
*allowed* to emit tool calls; the **final** turn must still be text.
Guard: if the last model message has no content, retry once with tools
unbound. Do not bind tools on light / extractor / recommender if those
nodes stay.

**Follow-up routing is not free with ReAct.** Without an explicit
policy, the loop will re-search “the first one.” Load-bearing prompt:

- new trip: dates tool → open slots → amenity/claim search → judge → answer
- cheaper / different weekend: same tools, merged constraints
- why / which / say that again: **no tools**, index `last_recommendations`

Cap iterations (e.g. 6). “Why?” should be 0 tool calls.

**Cost.** Current first ask: 1 extract + SQL + N judge calls + 1
recommender. Naive ReAct: a 235B round trip per tool (dates, slots,
several amenity queries, claims, judge, answer). Fine if capped; not
fine if it wanders. `$0.20` in / `$0.60` out per MTok on the 235B.

## How the three follow-ups should work (any loop)

| User says | Search? | State |
|---|---|---|
| why this? | no | last recs: `why`, official vs guest claim, price, dates |
| the first one again? | no | `last_recommendations[0]` — the list that was *spoken* |
| cheaper? | yes, delta | keep dates/party/amenities, tighten `price_per_night`, re-filter or re-run open slots |

Stuffing 80 pages makes “the first one” undefined (dump order vs spoken
order vs new retrieve). A 10-item `last_recommendations` array answers
all three. The current recommender cannot, because it never sees the
spoken list.

## If we spike option C

Streamlit, three scripted turns (first ask, why, cheaper), tool-call
log, empty-content guard, iteration cap. Success:

1. First ask still goes through `resolve_dates` (buried הקרוב stays
   `when=this` on the Horshat Tal query — replay experiments.md
   2026-09-08 §2 Q2, five trials).
2. “Why?” is 0 tools and quotes `why` / `review_claims` from the last
   rec, not a new retrieve.
3. “Cheaper” calls `search_open_slots` with a tighter price and does
   not drop dates.
4. No blank reply when the model wanted a tool.

Failure on (1) means keep the extractor node and only ReAct *after*
constraints exist. Failure on (2)–(3) is a prompt/state bug, not a
reason to switch to Claude Code.

## Decision

**None.** Graph stays until a spike. Option A rejected as a RAG
replacement. Option B rejected as a production runtime; a *Claude API*
tool loop with the same search MCP would be the same experiment as C
with a different backbone. Option C is the one to try, with dates and
the judge kept as tools (or nodes), not absorbed into free-form ReAct.
