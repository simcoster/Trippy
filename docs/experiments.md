# Experiments log

Every design choice made on the strength of an experiment points here, and
`docs/design.md` records the choice with its reason. Dates newest first; within
a date, in the order the experiments were run. An entry is never edited after
the fact — a re-run is a new entry. Each one says what question it answered,
how production was kept untouched, what came out, what it cost, and what was
decided.

## 2026-09-12

### 1. Which model writes the least robotic Hebrew why?

**Question.** Super’s why echoed the query and calqued English
(`מרחביים`, `שמדליות`). On the same planner packs, do
Qwen3.5-397B-A17B, Qwen3-235B, Kimi-K3, or GLM-5.2 sound like a
person?

**Setup.** Replay only. Packs from eval `2026-09-12_104117`
(extractor 235B, GLM batch judge, frozen occupancy, `experiments`,
no `public` writes). Same prompt (match-first, no query recap).
Thinking off on 397B/GLM/Kimi. 27 cases × 4 = 108 recommend calls.
Dumps `reports/evals/2026-09-12_110039.json` (397B),
`_110326.json` (235B), `_110609.json` (Kimi), `_111634.json`
(GLM). ~$0.89. No DB writes.

**Result.**

| | 397B | 235B | Kimi-K3 | GLM-5.2 |
|---|---|---|---|---|
| wall | 113s | 124s | 574s | **76s** |
| $ | 0.113 | **0.030** | 0.506 | 0.241 |
| Latin in why | 4 (`specifically`, `comfortably`) | 2 (`איןoutlets`, `mentioned`) | **0** | **0** |
| query-echo template | 1 | 2 | **0** | **0** |
| coined `מרחביים`/`שמדליות` | 0 | 0 | 0 | 0 |
| extra empty (fits existed) | 0 | **E09, H08** | 0 | 0 |
| n=2 | 11 | 1 | 17 | 17 |

Kimi is the only one that reads like spoken Hebrew (H02 fridge
caveat with recency; H03 room mini-fridge vs communal). GLM is
close, faster, and cheaper; still says `ליסטינג` once. 397B
glues English into Hebrew. 235B H12 is broken (`נמטר במחוז
המערבי… השקל של הסערה`).

**Decision.** Phrasing: Kimi > GLM >> 397B > 235B. Super stays
the default until we pick; speed is secondary because the reply
streams. design.md "Recommender".

### 2. Re-read §1 with phrasing over Latin

**Question.** Same four dumps. Weight invented/weird Hebrew
over an occasional Latin token. Listing+review repeats
(“site says showers, guests say showers”) are a prompt fix,
not a model knockout.

**Setup.** No new calls. Same dumps as §1.

**Result.** 235B is out on Hebrew itself: `תאוצה` for occupancy,
`האתר מציעה`, `אווני שינה`, `גורשת טל`, H12 garbage, plus
empty E09/H08. 397B coins `מלונאית`, `לצנוע`, `אינם מקרים`
(for מקררים), `התארים`, and invents Mitzpe HaYamim on E10.
GLM is close but coins `משוערפים`, uses `מקררון`, and on E10
claims Masada is not a campsite. Kimi is ordinary spoken
Hebrew; one E10 mix-up (מצפה הימים); listing+review doubles
on E06/E08 like the others.

**Decision.** Recommender → Kimi-K3
(`moonshotai/Kimi-K3`, thinking off). GLM is the runner-up.
Repeat “listed and guests confirm the same yes” is a prompt
change, not a model change. design.md "Recommender".

## 2026-09-11


### 1. Is Streamlit slower than eval on the same occupancy?

**Question.** The live-occupancy AppTest of E15/E03/E04 was 67s / 77s /
300s timeout. Eval `2026-09-11_154623` on frozen nights was 13.3s /
19.5s / 17.6s. Same Streamlit graph, eval env
(`TRIPPY_SCHEMA=experiments`, `availability_frozen`,
`TRIPPY_TODAY=2026-09-08`), with `collect_stages` + claim_judge call
counts.

**Setup.** AppTest of `scripts/streamlit_chat.py`, recommender path.
No copy, no writes to `public`. Three queries. Dump
`reports/streamlit_evalenv_2026-09-11_130528.json`. 40 judge calls.

**Result.**

| query | live AppTest | eval | Streamlit+frozen |
|---|---|---|---|
| couple 17 Sep (E15) | 67s (planner 57s) | 13.3s judge 3.6s×10 | **16.3s** judge 10 / 3.7s×10 |
| tent ≤₪80 (E03) | 77s (planner 32s, light 34s) | 19.5s judge 6.6s×20 | **17.0s** judge 20 / 4.6s×20 |
| sea, 4 friends (E04) | 300s timeout | 17.6s judge 4.7s×10 | **12.1s** judge 10 / 2.6s×10 |

~58 s wall. Recommend 6–7s in all three.

**Decision.** Streamlit is not a slower runtime. The earlier minutes
were live vacancy fan-out / a cold Nebius (light 34s). Keep stage +
`judge_calls` on the Streamlit trace. design.md "Recommender".

### 2. Live occupancy Streamlit with judge call counts

**Question.** On `public.availability`, do the same three queries fan
out more judge jobs than frozen (10 / 20 / 10), and does that explain
the first live 67s / 77s / 300s timeout?

**Setup.** AppTest, recommender path, no `TRIPPY_SCHEMA` /
`availability_frozen` / `TRIPPY_TODAY`. Reads `public` only. Timeout
600s. Dump `reports/streamlit_live_2026-09-11_131703.json`. 40 judge
calls.

**Result.** Same job counts as frozen: 10 / 20 / 10. Walls **47.6s /
43.1s / 46.1s**. Judge wall 16.6s×10 / 6.9s×20 / 16.6s×10 (frozen was
3.7 / 4.6 / 2.6). Extract 4.7 / 10.8 / 9.0s. Sea finished; it did not
need extra sites. ~158 s wall.

**Decision.** Today's live night does not add judge jobs on these
asks. The first 300s timeout was a stall, not a larger fan-out.
Per-call 235B/extract is what stretched live vs frozen.

### 3. Is the 12s Streamlit light node the frozen vs live env?

**Question.** Live AppTest light was 12.8s then 10.6s (285–297
prompt tokens, 2 completion, KEEP). Frozen Streamlit light on the
same couple query was 2.0s. Light never reads occupancy. Is the
gap the three env pins, or Nebius?

**Setup.** Direct `light_node` (no Streamlit, no extractor/planner),
couple query E15, 5 interleaved pairs: frozen pins then live
(unset). Same 235B cleaner. 10 calls. Dump
`reports/light_frozen_vs_live_2026-09-11_133327.json`. No DB.

**Result.** Every call KEEP (`{}`). Frozen 0.35–1.81s, mean
**0.69s**. Live 0.31–1.37s, mean **0.73s**. First call 1.81s
(warmup); the rest overlap.

**Decision.** Frozen vs live env does not slow light. The 12s
Streamlit light was Nebius TTFT/queue in that process, not the
pins. design.md unchanged.

### 4. Is the ~47s live turn Streamlit/AppTest?

**Question.** Live AppTest E15 was 47.6s (light 12.8s, judge
16.6s×10, recommend 9.4s). Isolated `light_node` is ~0.7s. Does
the same full graph without Streamlit still take ~47s?

**Setup.** `build_graph(stop_after="recommender")` in-process,
`compiled.stream` with `collect_stages` + claim_judge counts. No
AppTest, no widget serialize. Five repeats, live
`public.availability`, E15. ~50 LLM calls. Dump
`reports/full_nostreamlit_2026-09-11_145919.json`.

**Result.** Mean **18.6s** (8.9–44.6). Same 23 fits except run 4
skipped RAG/judge (extractor emitted no amenity queries; planner
0.1s).

| n | wall | light | extract | planner | judge | recommend |
|---|---|---|---|---|---|---|
| 1 | **44.6s** | 4.2s | 0.9s | 6.3s | 3.3s×10 | **33.3s** |
| 2 | 10.1s | 0.7s | 1.1s | 3.6s | 2.2s×10 | 4.7s |
| 3 | 17.0s | 0.7s | 1.3s | 8.3s | 7.0s×10 | 6.7s |
| 4 | 8.9s | 0.8s | 2.0s | 0.1s | 0 | 6.0s |
| 5 | 12.6s | 0.6s | 1.4s | 4.7s | 2.9s×10 | 5.9s |

**Decision.** AppTest is not the 47s. First-turn Nebius is
(here recommend 33s, not light). After that, live full flow is
~10–17s, in the frozen Streamlit band. design.md unchanged.

### 5. Does a newer model make one batched judge call accurate?

**Question.** One-call-per-case batch still dropped E03 `tent` vs
`tent_pitch` on the 235B (2026-09-10 §3). Do Qwen3.5-397B-A17B,
DeepSeek V4 Pro, or GLM-5.2 match stored one-by-one 235B `satisfies`
on a compact batch, and is the wall worth it?

**Setup.** No planner rerun, no DB writes. Jobs + gold from eval
`2026-09-11_154623` (fits + rejected) for E03/H02/H07/E04/H10
(20/20/20/10/10). Same `CLAIM_JUDGE_SYSTEM` + compact suffix + a
`judgements[]` wrapper. Four Nebius models, thinking disabled when
the API accepted it, `max_tokens=8000`. 20 calls. Dump
`temp/judge_batch_models_2026-09-11_191112.json`. $0.36.

**Result.** Agree vs stored one-by-one 235B:

| model | all | E03 | H02 | H07 | E04 | H10 | wall | $ |
|---|---|---|---|---|---|---|---|---|
| Qwen3-235B (current) | 74/80 | **16/20** | 20/20 | 18/20 | 10/10 | 10/10 | **35.5s** | 0.009 |
| Qwen3.5-397B-A17B | **77/80** | **20/20** | 20/20 | 17/20 | 10/10 | 10/10 | 176s | 0.111 |
| DeepSeek V4 Pro | 68/80 | **11/20** | 20/20 | 17/20 | 10/10 | 10/10 | 22.5s | 0.079 |
| GLM-5.2 | 60/80 | **20/20** | 20/20 | **0/20** | 10/10 | 10/10 | 57s | 0.164 |

E03 tent misses on 235B are the same four sites as 2026-09-10 §3
(Yehudiya, Horshat, Akhziv N/S): batch treats `tent` polarity false
as no lodging and ignores `tent_pitch` true. 397B and GLM grant
those. DeepSeek drops more tents (11/20). H07 Akhziv `not caravan`
false-yes on 235B/397B/DeepSeek (gold false). GLM H07 wrote a
prose analysis until `max_tokens=8000` (no JSON). 397B/GLM billed
3–8k completion tokens/call on compact JSON (thinking still
counted). Current 5-wide 235B singles on E03 were 6.6s.

**Decision.** Stay on one-call-per-job 235B with concurrency 5.
397B fixes E03 tent but is slower and ~12× the batch 235B cost,
and still misses H07 caravan. GLM matches when it returns JSON
and dies when it does not. DeepSeek is fast and worse on tent.
design.md unchanged.

### 6. Does a 2000 completion cap make the batch models usable?

**Question.** §5 billed 3–8k completion tokens on 397B/GLM
(thinking or prose). Compact 20-job JSON on 235B was ~720 tokens.
Does `max_tokens=2000` (~2.5× that) keep JSON and cut cost/wall?

**Setup.** Same jobs, gold, prompt, and four models as §5. Only
change: `max_tokens` 8000→2000. 20 calls. Dump
`temp/judge_batch_models_2026-09-11_192628.json`. $0.25.

**Result.**

| model | all | E03 | H02 | H07 | E04 | H10 | wall | $ | finish |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-235B | 74/80 | 16/20 | 20/20 | 18/20 | 10/10 | 10/10 | 48s | 0.009 | all stop, ~350–750 out |
| Qwen3.5-397B | **0/80** | 0 | 0 | 0 | 0 | 0 | 72s | 0.058 | **all length**, 2000 out, no JSON |
| DeepSeek V4 Pro | 73/80 | 19/20 | 20/20 | 14/20 | 10/10 | 10/10 | 35s | 0.078 | all stop, ~380–950 out |
| GLM-5.2 | 10/80 | 0 | 0 | 0 | 0 | 10/10 | 24s | 0.100 | length except H10 stop |

397B and GLM spent the budget on thinking/prose and never reached
`judgements[]` (GLM E04 started JSON and truncated). 235B scores
match §5. DeepSeek E03 moved 11/20→19/20 at `stop` (under the
cap); H07 got worse 17/20→14/20 — run noise, not the cap.

**Decision.** A compact-sized cap does not tame 397B/GLM; it
deletes their JSON. Stay on per-job 235B. design.md "Planner
claim/rule judge".

### 7. Does turning thinking off make the 2000-token batch work?

**Question.** §6 failed because 397B/GLM spent the 2000-token cap
on thinking. Same cap, same jobs: if thinking is off
(`reasoning_effort=none`, `chat_template_kwargs.enable_thinking=false`,
`thinking.type=disabled`), do they emit compact JSON and match gold?

**Setup.** Same as §6 plus the no-think flags on every model
including 235B. `max_tokens=2000`. 20 calls. Dump
`temp/judge_batch_models_2026-09-11_193236.json`. $0.19.

**Result.** All `finish=stop`, `reasoning_tokens=0`, ~360–950 out.

| model | all | E03 | H02 | H07 | E04 | H10 | wall | $ |
|---|---|---|---|---|---|---|---|---|
| Qwen3-235B | 74/80 | **16/20** | 20/20 | 18/20 | 10/10 | 10/10 | 49s | 0.009 |
| Qwen3.5-397B | **78/80** | **20/20** | 20/20 | 18/20 | 10/10 | 10/10 | 27s | 0.034 |
| DeepSeek V4 Pro | 74/80 | **20/20** | 20/20 | 14/20 | 10/10 | 10/10 | 21s | 0.078 |
| GLM-5.2 | **78/80** | **20/20** | 20/20 | 18/20 | 10/10 | 10/10 | **8.4s** | 0.073 |

397B/GLM/235B share the same two H07 misses: Akhziv N/S `not
caravan` gold false, batch true via rule. DeepSeek E03 tent is
now 20/20; H07 is the noisy case (6 misses).

**Decision.** Thinking must be off for a compact batch. 397B and
GLM then beat 235B batch on E03 tent and are faster. Do not
switch production yet: five cases, and the Akhziv caravan miss is
still real. Stay per-job 235B until a full eval. design.md
"Planner claim/rule judge".

### 8. Does GLM-5.2 one-call (thinking off) match 235B on planner_v1?

**Question.** §7 GLM batch was 78/80 vs stored 235B singles on five
cases. On the full 27, with thinking off and one judge call per
case, does the planner score hold, and what does it cost?

**Setup.** `just run-eval -- --recommender --judge-batch
--judge-model glm`. Frozen occupancy, experiments schema copy
(skip availability). Extractor and recommender stay 235B. Judge
is GLM-5.2, compact, `max_tokens=2000`, thinking off. 18 judge
calls (cases with jobs). Dump
`reports/evals/2026-09-11_230420`. ~75 LLM calls, 322 s.

**Result.** **22/27** (easy 14/15, hard 8/12) — same count as
235B singles `2026-09-11_154623`, different misses.

| | 235B 5-wide | GLM batch |
|---|---|---|
| pass | 22/27 | 22/27 |
| judge | 70.8s×189 | **30.6s×18** |
| judge in/out | 447k / 6.2k | 134k / 7.0k |
| judge $ | ~0.09 | ~0.22 |
| wall | 335s | 322s |

E10 couple-tent at Masada **PASS** (was empty). E05 desert drops
Be'erot (17). H03/H06/H07 still fail. H12 now unexpected Akhziv
37 (was missing 38). Recommend still ~154s of the wall.

**Decision.** Same score, not a quality win. Batch is ~2.4× the
235B judge $ (not 10× — the system prompt is no longer repeated
189 times) and about half the judge wall. Stay on per-job 235B.
Flags: `TRIPPY_JUDGE_BATCH` / `--judge-batch`,
`TRIPPY_JUDGE_MODEL=glm`. design.md "Planner claim/rule judge".

### 9. Nemotron Lightning / Super as the recommender?

**Question.** Can NVIDIA Nemotron 3.5 Lightning or Nemotron 3
Super 120B-A12B replace Qwen 235B on the Hebrew `why`, on the
five longest recs from eval `2026-09-11_154623`?

**Setup.** Planner once per case (235B extract+judge, frozen
occupancy, `experiments`, no `public` writes). Same packed fits
then recommended by 235B, `nvidia/Nemotron-3_5-Lightning`
($0.06/$0.24), and `nvidia/nemotron-3-super-120b-a12b`
($0.30/$0.90). Thinking off (`/no_think` +
`chat_template_kwargs.enable_thinking=false`; Super rejects
`reasoning_effort=none`). Cases E03, E04, E06, E12, H03 (longest
recommend completions that actually picked a stay: 304 / 248 /
253 / 249 / 229 out). Dump
`temp/recommender_nemotron_2026-09-11_202946.json`. 15 recommend
calls + planner. 86 s recommend+planner wall.

**Result.** Same first stay as live 235B on E03/E04/E06. E12
Lightning+Super picked Yehudiya, 235B kept Mamshit. H03 Super
and Lightning picked `חושה`; 235B picked `חושה עם מזגן…`.

| | 235B | Lightning | Super |
|---|---|---|---|
| recommend s | 19.8 | **8.8** | 17.0 |
| recommend $ | 0.009 | **0.003** | 0.015 |
| Hebrew why | yes, `pitch`/`outlets` leaks | **English on 5/5** | yes, **0 leaks** |
| n=1 | 5/5 | 1/5 (else n=2) | 5/5 |

Lightning quotes subject keys (`mini_refrigerator`, tent pitch)
and ignores the one-language rule. Super’s Hebrew is cleaner
than 235B (E03 `שקע חשמל` not `איןoutlets`) but E06 invents
`טוקול` and H03 leads with fridge complaints.

**Decision.** Do not switch. Lightning cannot write Hebrew.
Super is the interesting Hebrew candidate, not cheaper, not a
clear quality win on five cases. Stay 235B. design.md
"Recommender".

### 10. Ship Super as the recommender anyway?

**Question.** §9 left Super as the Hebrew candidate: 0 Latin
leaks vs 235B `pitch`/`outlets`, same first stay on 3/5, not
faster (16.9s vs 19.8s), 1.5× $. Ship it?

**Setup.** No new calls. Same dump
`temp/recommender_nemotron_2026-09-11_202946.json`.

**Result.** The leak-free Hebrew is the product reason. Cost
and wall are not. Lightning stays out.

**Decision.** Recommender → Nemotron Super 120B-A12B, thinking
off (`/no_think` + `chat_template_kwargs.enable_thinking=false`).
Extractor, light, and judge stay 235B. `TRIPPY_RECOMMENDER_MODEL=235B`
opts back. design.md "Recommender".

## 2026-09-10

### 1. Does feeding the Hebrew breadcrumb label with the English slug stop Dead Sea from satisfying "near the sea"?

**Question.** Masada's stored claim is `region:dead-sea` (label `ארץ ים המלח`).
The 235B judge grants `near the sea` on the slug. Does adding the Hebrew
label, or using Hebrew alone, make it a no? Does Akhziv's actual
breadcrumb (`region:western-galilee` / `גליל מערבי`) grant sea?

**Setup.** No embed, no DB writes. Production `judge_site_request` (235B),
claims only, no rules. Request `near the sea`. Six calls: slug / hebrew /
both × Masada and Akhziv. Dump `temp/breadcrumb_sea_judge.json`.

**Result.** Masada: slug, Hebrew, and both all `satisfies=true`
(`satisfy_by=claim`; Dead Sea / ארץ ים המלח counts as the sea). Akhziv:
all three `satisfies=false` (Western Galilee is not about the sea).
6 calls, 38 s, $0.0019.

**Decision.** Do not dual-write Hebrew+slug as a judge fix. The Dead Sea
false positive is semantic, not a missing-label problem. Akhziv sea
does not live in breadcrumbs.

### 2. Does a judge-prompt principle stop Dead Sea from satisfying "near the sea"?

**Question.** 235B grants `near the sea` on `region:dead-sea` / `ארץ ים המלח`.
Does a principle — breadcrumb `area:*`/`region:*` satisfy that region by
name, not "near the sea" because the slug contains ים/sea — fix Masada
without dropping Akhziv beach, `area:north` → north, `region:negev` →
desert, or `region:dead-sea` → "Dead Sea"?

**Setup.** Prompt change in `CLAIM_JUDGE_SYSTEM`. No embed, no DB writes.
Eight production `judge_site_request` (235B) calls. Dump
`temp/breadcrumb_sea_judge_prompt.json`. Control: same Dead Sea feeds
were all `satisfies=true` in §1.

**Result.** 8/8. Masada slug / Hebrew / both → `satisfies=false`. Akhziv
beach → true. Western Galilee → false. `area:north` → "in the north"
true. `region:negev` → desert true. `region:dead-sea` → "Dead Sea" true.
8 calls, 225 s, $0.003.

**Decision.** Keep the principle and the Dead Sea / beach few-shots in
the judge prompt. design.md "Planner claim/rule judge".

### 3. Can one 235B call judge all (site, query) jobs for a planner case?

**Question.** Eval cases with 20 judge calls: is a single batched JSON
list as accurate as one call per (campsite, query), and how much wall
time does it save?

**Setup.** No planner rerun, no DB writes. Jobs from
`reports/evals/2026-09-10_102209.json` retrieved payloads for E03, H02,
H07 (20 jobs each). Same current `CLAIM_JUDGE_SYSTEM`. One-by-one
`judge_site_request` vs one batch call (`max_tokens=8000`) that returns
`judgements[]`. Dump `temp/judge_batch_compare.json`. 63 calls.

**Result.** Satisfies agree 16/20 (E03), 20/20 (H02), 20/20 (H07). The
four misses are all E03 `tent`: one-by-one grants `tent_pitch` polarity
true; batch treats `tent` polarity false as no tent lodging (Yehudiya,
Horshat, Akhziv N/S). Wall: 20.7s→13.3s, 16.7s→9.5s, 17.1s→7.6s.
Tokens: 60 singles 119k in / 3.6k out $0.026; 3 batches 27k in / 4.5k
out $0.008. Total $0.034.

**Decision.** Do not batch in production yet. H02/H07 matched; E03 tent
vs tent_pitch is a real quality drop. Parallel singles (`--judge-concurrency 4`)
already cut wall without changing the verdict schema.

### 4. Are batches of 5 or 10 (site, query) jobs more accurate than 20?

**Question.** Same E03/H02/H07 jobs and one-by-one verdicts as §3.
Does chunking the list to 5 or 10 recover the E03 tent/`tent_pitch`
misses?

**Setup.** No DB writes. Stored one-by-one from
`temp/judge_batch_compare.json`. New 235B batch calls of 5 and 10.
Dump `temp/judge_batch_chunks.json`. 18 calls, $0.021.

**Result.** Agree vs one-by-one (satisfies):

| size | E03 | H02 | H07 |
| 20 (§3) | 16/20 | 20/20 | 20/20 |
| 10 | 17/20 | 20/20 | 18/20 |
| 5 | 17/20 | 20/20 | 19/20 |

E03 still drops Horshat + Akhziv N/S `tent` (`tent` polarity false
overrides `tent_pitch`). Yehudiya recovered vs size 20. H07 picked up
a `no caravan` false no on Besor. Sequential chunk walls ~22–31s are
not faster than 20 one-by-one (~17–21s) because chunks run one after
another.

**Decision.** Smaller batches do not fix the tent_pitch error. Stay on
one-call-per-job; speed via concurrency, not list size.

### 5. Does the 30B judge match 235B on the three heaviest eval cases?

**Question.** Same E03/H02/H07 jobs as §3. One-by-one 30B vs stored
235B `satisfies`. Is it as accurate, and is it faster?

**Setup.** No DB writes. `TRIPPY_INSTRUCT_MODEL=30B`. 60
`judge_site_request` calls. Dump `temp/judge_30b_compare.json`.

**Result.** Satisfies agree 20/20, 20/20, 19/20. The one miss (Tel Arad
`no caravan`) is `max_tokens=600` truncation: the 30B JSON said
`satisfies: true` then was marked false as unparseable. Wall 19.8 / 17.0
/ 24.8s vs 235B 20.8 / 16.7 / 17.2s — no sequential speedup (RTT).
Cost $0.013 vs $0.026 for the 235B sixty.

**Decision.** Do not switch the production judge to 30B on this probe
alone. Accuracy is close; sequential latency is not the win. Concurrency
on 235B still owns wall time.

### 6. Does compact judge output cut tokens and wall on E02?

**Question.** E02 (`camping` × 10 sites). Compact (`relevant` indices +
4–5 word `reason`) vs quoted `relevant_claims`. Same 235B, sequential,
`--no-copy`. Does decode drop, and does wall follow?

**Setup.** `just run-eval -- --ids E02 --no-copy --model 235B` twice,
once `--judge-compact`. Experiments schema. Reports
`2026-09-10_121353` (compact) and `2026-09-10_121442` (quoted).
22 chat calls.

**Result.** Both PASS. Extractor identical (2185 in / 93 out).

| | judge in | judge out | judge wall | cost |
| quoted | 20186 | 824 | 20.4s×10 | $0.0045 |
| compact | 23736 | 376 | 13.4s×10 | $0.0050 |

Out ~half (82 → 38 tokens/call). In *up* ~3.5k from the compact
suffix few-shots, so judge cost is slightly higher. Reasons are
4–5 words (`tent pitch granted`) vs long quoted paragraphs. Both
kept `satisfies=true` on every site.

**Decision.** Keep the flag default off until a full eval. Compact
wins decode and this case's wall; shorten the compact suffix if
the extra prompt tokens matter.

### 7. First 235B recommender dump on planner_v1

**Question.** After rewriting `recommender_node` as a JSON picker
(query + extract + compact fits, relevant claims, unsifted retrieved
rules), what does it actually say on the 26-query set? Does it stay
inside `fits`, pick 1–2, and cite listing vs guests?

**Setup.** `just run-eval -- --recommender`. Experiments schema,
`availability_frozen`, `TRIPPY_TODAY=2026-09-08`, compact judge ×5,
235B. No writes to `public`. Recs are not scored. Report
`reports/evals/2026-09-10_185136.md`.

**Result.** Planner 19/26 (easy 11/14, hard 8/12) — same gold as
before; H07 still fails caravan bays (1113 s judge). Wall 1290 s.
Tokens in=574313 out=11162 (extract 56920/2392×26, claim_judge
409089/5606×173, recommend 108304/3164×26). ~$0.12 total, ~$0.024
of that on recommend.

15 cases emitted one rec; 11 emitted empty; **none emitted two**.
Empty was right when `fits` was empty (E14/H11 no dates; H10 dogs
forbidden; named-site misses). Picks stayed inside `fits` (E04/H02
Akhziv sea, E10 Masada couple tent, H12 south Shabbat plate, H07
Tel Arad family tent not a bay). Caveats showed up (E04 dirty beach,
H02 warm fridges).

Failure modes for the next pass: never used the second slot (H06
sea∨desert recommended only Akhziv); English/Chinese leaks in the
Hebrew `why` (H12 `accommodation`/`dank`, E10 `二人`, H02
`מקampינג`); E05 cited caravan stations as desert color; empty
replies sometimes echo the planner's English miss (`Horashat Tal`
on H08).

**Decision.** Keep the picker. Correct from this dump: ask for 2
when two loci or two sites fit; forbid Latin in Hebrew `why`.
design.md "Recommender".

### 8. Did “Hebrew only” stop Latin/CJK in recommender `why`?

**Question.** After the one-language prompt (no Latin/Chinese, not
`pitch` / `camping` / `Stay`), does the 235B still leak foreign
scripts into Hebrew `why`?

**Setup.** Same `--recommender` dump as production path; report
`reports/evals/2026-09-10_195239.md` (after the language prompt).
No new calls for this note.

**Result.** Leaks in 7 of the 15 one-rec Hebrew replies. Same
glitch forms: `ゲuests` (E02, E03, E06, E12, H03), `הospites`
(E08, E10), `.pitch` / `.pitch tent` (E03, E12), `בungalו` (E11),
`איןoutlets` (E12). E03 is the tent+₪80 case: `יש.pitch ל אוהלים`
plus `ゲuests דיווחו`.

**Decision.** The forbid-list was not enough. Packed input is
English (`review_claims.claim` is `text_en`, rules `subject` is
`tent_pitch`) and the prompt cued “say guests report”, which Qwen
emits as `ゲuests` / `ospites`. Prompt now: paraphrase claims,
use `evidence_span` not the subject key, Hebrew reviews as
אורחים מספרים. design.md "Recommender".

## 2026-09-09

### 1. Can breadcrumb slugs retrieve and satisfy a north query without the claim splitter?

**Question.** parks.org.il `#breadcrumbs` are `בית > צפון > גליל עליון > …`.
Can we store them as claims for "not far from north"?

**Setup.** Hurshat Tal page. No DB writes. (1) Phrase as `the site is
on region:צפון at:גליל עליון` (and English `north` / `upper galilee`)
through production `split_one_review` (235B). (2) Embed `area-north`
and `upper-galilee` with Qwen3-Embedding-8B, `<#>` to `not far from
north`. (3) Production `judge_site_request` on those slugs as
`is_positive` claims, no rules. Dumps `temp/breadcrumb_claims_probe.json`,
`temp/breadcrumb_embed_probe.json`, `temp/breadcrumb_judge_probe.json`.

**Result.** Splitter: `{"claims": []}` both phrasings (~$0.0005).
Embed: `area-north` −0.787, `upper-galilee` −0.605; both pass the −0.6
claim gate. Judge: `satisfies` true on both together and each alone,
`satisfy_by=claim` (~$0.0009). Nested URL slugs are `an-upper-galilee`
/ `as-dead-sea` / `ac-coastal-plain`; the two-letter area prefix is
stripped so the stored claim is `upper-galilee`.

**Decision.** Fourth `scrape-info` stage: parse `#breadcrumbs`, embed
the slugs, insert claims with `review_id` NULL and
`notes='no review, region by breadcrumbs'`. Not through the splitter.
design.md "Breadcrumb regions are claims without a review".

## 2026-09-08

### 1. Can the unit extractor name a caravan-bay hookup as not generic electricity?

**Question.** `"electricity"` retrieves `electric_hookup` at −0.704 for
עמדת חניה לקרוואן פרטי ("קיים חיבור חשמל ומים") and for מתחם PITCH
tent power. The judge already drops the caravan bay for a tent stay.
Can a small unit-prompt change make the extractor emit
`caravan_bay_electric_hookup` / `caravan_bay_water_hookup` instead of
bare `electric_hookup` / `water_hookup`, without renaming PITCH power?

**Setup.** No writes. `RuleExtractorLLMClient` (235B) on
`UNIT_PROMPT`. Listing:

```
עמדת חניה לקרוואן פרטי
כניסה לחניון לילה עם קרוואן פרטי
קיים חיבור חשמל ומים
הרכב ההזמנה: עד 6 לנים בהרכב.
```

Control: `מתחם PITCH` / "אוהל בשטח. חיבור חשמל. שירותים ומקלחות משותפים."
Current prompt vs the same prompt plus (1) a never-name-the-unit
exception for vehicle hookups and (2) a few-shot of that listing.
4 calls. Dump `temp/caravan_bay_hookup.json`.

**Result.** Current caravan: `trailer_parking`,
`campsite_entry_with_private_caravan`, **`electric_hookup`**,
**`water_hookup`** (occupancy skipped). Proposed caravan:
`trailer_parking`, **`caravan_bay_electric_hookup`**,
**`caravan_bay_water_hookup`**. PITCH stayed `tent_pitch` +
`electric_hookup` + toilets/showers both times. ~4–7 s/call, ~3500–3800
in / 220–380 out, ≈$0.0009 each, **$0.0036** total.

**Decision.** Ship the glossary and the never-name exception on
`UNIT_PROMPT`. Classifier one-liner aligned (`caravan_bay_electric_hookup`,
not `electric_hookup`) so resolve does not teach the old name.
design.md "One tooltip, one pipeline".

### 2. Is date_intent more consistent on 235B, or on a 30B dates-only call?

**Question.** The 7 Sep few-shots were 25/25 on the 30B, including
Horshat Tal `בשישי הקרוב`. The 8 Sep five-query replay emitted
`when=next` for that same query. Is that a flake of the 30B full
extractor, does the 235B hold, and does a 30B call that extracts only
`date_intent` hold?

**Setup.** No writes. Today frozen to **Tuesday 2026-09-08** in the
prompt and in `resolve_dates`. Temperature 0. 6 prompts × 5 trials × 3
setups = **90 calls**. Setups: production `EXTRACTOR_SYSTEM_PROMPT` on
30B; the same prompt on 235B; a dates-only 30B prompt (same date rules
and few-shots, no amenities / price / campsite). Gold is the compact
intent plus the resolved `start`. Dump
`temp/date_intent_consistency.json`.

| prompt | gold start |
|---|---|
| Q1 סופ״ש הבא + showers + north | 18 Sep, `when=next` weekend |
| Q2 חורשת טל בשישי הקרוב ≤₪400 | 11 Sep, `when=this` Friday |
| בשישי הקרוב (bare) | 11 Sep, `when=this` |
| Q3 חמישי הבא + sea + electricity | 17 Sep, `when=next` Thursday |
| Q4 סוף השבוע בעוד שבועיים | 25 Sep, `weeks_from_now=2` |
| Q5 מהיום ליומיים | 8 Sep, `on=today`, nights=2 |

**Result.** **30B full 27/30.** The only misses are Q2: **2/5**
`when=this` (11 Sep), **3/5** `when=next` (18 Sep). Bare הקרוב on the
same model is 5/5. **235B full 30/30**, one intent per prompt.
**30B dates-only 30/30**, one intent per prompt, ~1s/call, 23k in /
0.8k out. Wall 272s / 4 workers. Est. **$0.025** (30B full $0.008,
235B $0.015, dates-only $0.003).

**Decision.** Not shipped. The 30B full extractor is inconsistent on
buried הקרוב; a second dates-only 30B call matches the 235B without
moving amenity extract. Wait for a product choice. design.md
"Query extractor: date_intent".

### 3. Ship the 235B for query extract?

**Question.** §2: 235B full 30/30, dates-only 30B 30/30, 30B full 27/30
on buried הקרוב. Latency and $ vs the current 30B extractor hop?

**Setup.** Same 90-call dump as §2. Per-call p50 from those 30 trials
each. Now = 30B full. Options: replace with 235B full, or add a
dates-only 30B call and keep 30B full for amenities.

**Result.** Extractor hop only (judge/planner dwarf it):

| | now 30B full | 235B full | 30B dates + 30B full |
|---|---|---|---|
| gold | 27/30 | 30/30 | 30/30 if dates-only wins |
| p50 latency | 4.0 s | 2.4 s | sequential 5.0 s / parallel ~4.0 s |
| $ / search | $0.00025 | $0.00050 | $0.00034 |

**Decision.** Extractor → 235B (`extractor_model`). Simpler than a
second call; 2× token cost, faster on this run. `planner_node` stays
SQL — the old name `planner_model` was the extract chat client.
design.md "Query extractor: date_intent".

### 4. Does `מידע למבקר` extract on the same pipeline as `מה בחניון?`?

**Question.** The accordion tab is AJAX-loaded and was not in
`parse_sections`. wrapUseInfo already has hours, booking and the dog
icon; is the tab a richer visitor-rules list, and does one extract
call on the bullet list recover the facts?

**Setup.** Akhziv only
(`https://www.parks.org.il/camping/…אכזיב…`). `experiments` schema
reset (`clone_tables` of campsites / subject_vectors / campsite_rules),
parent campsite 2 copied from `public`, vocabulary seeded (234
subjects). Production not written. One `RuleExtractorLLMClient` call
on the panel body, then resolve + upsert. Gold = the 28 `<li>` plus
the caravan paragraph. 35 chat / 41 embed, 193 s, **$0.014**. Report:
`reports/visitor_info_ingest/2026-09-08_142435.md`.

**Result.** 47 statements, 46 stored, 1 dropped. 26/29 gold lines
covered. The three "misses": caravan-closed **was** stored (matcher
false negative on `<strong>` whitespace); reservation-required
dropped (`entry_without_reservation_allowed` failed the positive-
phrasing guard); "in the national park by the sea" correctly skipped.
Wrong merge: `late_entry_exit_end_time` 18:00 → `check_in_end_time`
(20:30 on the hours section). `gas_balloons_max_weight` 10 **meters**;
lifeguard `15/10` → 15.1 days. Southern-only Shabbat collapsed onto
site-wide `shabbat_observance_suitable_allowed` true.

**Decision.** Fetch + parse shipped into `ingest_site` (same path as
`מה בחניון?`). Open: reservation drop, 18:00→check-in over-merge
(will CONFLICTING against wrapUseInfo hours), no kg/date units.
design.md "The `מידע למבקר` accordion".

### 5. After the prompt and naming fixes, does Akhziv visitor-info still drop reservation and merge 18:00 into check-in?

**Question.** The first run (§4) dropped `entry_without_reservation_allowed`
on infix `_without_`, skipped the sea line as brochure, and the 235B
judge merged `late_entry_exit_end_time` 18:00 into `check_in_end_time`.
Prompts now keep sea/desert/forest as amenities, treat visiting-hours
pointers as emit-nothing, and keep infix `_without_`. Naming rewrites
`cant_` / `cannot_` → `can_` (polarity false) instead of dropping.
The judge prompt has a few-shot that a late-arrival surcharge hour is
not check-in. Does a re-extract recover reservation and sea, skip the
hours pointer, and keep 18:00 off `check_in_end_time`?

**Setup.** Same as §4: Akhziv only, `experiments` schema reset
(`clone_tables` of campsites / subject_vectors / campsite_rules),
campsite 2 copied, 234 subjects seeded from `public`. Production not
written. One `RuleExtractorLLMClient` (235B) call on the panel, then
resolve (`SubjectAdjudicatorLLMClient` judge 235B / classify 30B) +
upsert. 21 chat / 33 embed, 70 s, **$0.009**. Report:
`reports/visitor_info_ingest/2026-09-08_154130.md`.

**Result.** 39 statements, 36 stored, **0 naming drops**. 28/29 gold
lines. The only miss is `ניתן להגיע לחניון הלילה בהתאם לשעות הכניסה
המפורסמות באתר` (emit-nothing, as asked). Reservation stored as
`entry_by_reservation_only` true. Sea stored (`near_water` merged
into it). `late_check_in_end_time` 18:00 was offered `check_in_end_time`
(−0.823) and the judge rejected it. 50% and 100% both named
`late_check_in_fee_percent` — CONFLICTING, 100 dropped. Southern-only
Shabbat still merged into site-wide `shabbat_observance_suitable_allowed`
true. No muzzle line on this panel; `cant_` rewrite is unit-tested.

**Decision.** Ship the `cant_` rewrite and the late-fee few-shot.
Open: two fee percents on one subject; no kg/date units; Shabbat
scope; `near_water` collapsed onto `sea`.
design.md "The `מידע למבקר` accordion".

### 6. After a full Akhziv info scrape, does query 3 still miss the sea?

**Question.** The 8 Sep canvas replay of query 3
(`רוצים קמפינג ליד הים ל3 אנשים ביום חמישי הבא ללילה אחד, עם חשמל`)
was 0 fits / 21 rejected: electricity retrieved on inland sites, the
judge dropped those as not near the sea, and Akhziv tents never had
electricity. After visitor-info stores `sea` as a site amenity, does
a full `scrape-info` of Akhziv only put those tents through the sea
gate, and do they still fail electricity?

**Setup.** `just setup-experiments copy`. Deleted 105 site-level
`campsite_rules` for campsites 2/37/38 (unit-level rows kept).
`just on-experiments scrape-info -- --site 2` (rooms → prices →
rules, visitor-info included). Vocabulary kept from the copy so
resolve merged. Availability copied from `public`, not re-scraped.
Then `build_graph(stop_after="planner")` on query 3.
Production not written. Dump `temp/akhziv_query3_planner.json`.
Scrape 933 s, **$0.048** (rooms $0.013, prices $0.002, rules $0.034).
Planner 58 s, 1 extract + 14 judge calls (7 inland sites × 2 queries).

**Result.** Extract: party ≥ 3, `near the sea` (site) ∧ `electricity`
(site), Thu next **17–18 Sep**. **0 fits / 19 rejected.** Akhziv
north (37) and south (38) tent pitches are in the list: sea retrieved,
then `missing_stated_amenity` electricity (no judge). Inland powered
slots still fail the sea judge. `sea` and `near_water` both stored
true on 37 and 38; `phone_charging_points` did not retrieve for
"electricity". Full-page conflict pass renamed the 100% late fee to
`late_check_in_after_1800_fee_percent` (case #26).

**Decision.** No planner change. Visitor-info `sea` is on the site
amenity lane. Tent electricity is still absent, which is the page.
design.md "The `מידע למבקר` accordion".

### 7. Same stay, fridge instead of electricity — does Akhziv fit?

**Question.** §6's Akhziv tents died on electricity. The southern
huts have `mini_refrigerator`; both subcamps have site-level
`refrigerator` from `מקררים (3)`. Same query with `עם מקרר` in
place of `עם חשמל`: do they fit?

**Setup.** Same `experiments` copy and scrape as §6. No further
writes. `build_graph(stop_after="planner")` on
`רוצים קמפינג ליד הים ל3 אנשים ביום חמישי הבא ללילה אחד, עם מקרר`.
Dump `temp/akhziv_query3_fridge.json`. 76 s, 1 extract + judge
calls on the slots that passed retrieve.

**Result.** Extract: party ≥ 3, `near the sea` (site) ∧ `fridge`
(**site**, not room), 17–18 Sep. **0 fits / 21 rejected.** Only
Akhziv **tents** were in the vacancy list (huts not vacant that
night). Sea now **satisfies** (beach-access claim). Fridge retrieve
was **complaints** about the communal kitchen fridges; the judge
said all relevant claims are negative and "no official rule provides
fridge". The official `refrigerator` true (`מקררים (3)`) never
entered `why` — claims already hit, so the site-amenity lane was
skipped. Inland sites whose `why` carried `refrigerator` passed
fridge and still failed sea.

**Decision.** No code change. Fridge-as-site is what `עם מקרר`
extracted. Open: a claim hit shadows the official row, so a
complaint pack can veto a listing that does provide fridges.
experiments.md this entry.

### 8. Retrieve site amenities even when a claim already hit?

**Question.** §7: Akhziv listed `refrigerator` (`מקררים (3)`) but
fridge retrieve was complaints, so `search_site_amenities` never
ran and the judge said there was no official rule. The judge's
rules query used `COALESCE(parent_id, id)`, so subcamp 37/38
searched parent 2, which has no visitor-info rows. Should both
lanes always retrieve, and should a subcamp's own rules reach
the judge?

**Setup.** No live planner. The skip is
`test_site_amenity_search_scoped_to_still_unmatched_sites`; the
parent-only LATERAL is `search_campsite_rules`. Decision from §7
plus the ingest writing onto children.

**Result.** Not a model error. Retrieve skipped the listing;
rules search looked at the wrong campsite id.

**Decision.** Always retrieve claim **and** site amenity; match
if either hits. `search_campsite_rules` uses
`cr.campsite_id IN (site.id, site.parent_id)`. design.md "Planner
claim/rule judge".

### 9. Subsite+parent only; nos do not veto a yes?

**Question.** §8 opened child+parent rules. Should south Akhziv
see north's visitor-info? And when a listing grants fridge but
claims complain, does the judge drop the site?

**Setup.** No live planner. Spec: retrieve this site and its
parent, never other children of that parent; always feed the
judge both rules and claims; satisfies = any granting rule OR
any granting claim; nos are recommender evidence only.

**Result.** `IN (id, parent_id)` already excludes sisters; the
same scope now applies to `search_site_amenities`. The fridge
§7 fail was a missing rule in the pack plus a model that treated
complaints as a veto.

**Decision.** Scope is own+parent on both rule scans. Judge
prompt: a no does not set `satisfies` false when a yes exists.
design.md "Planner claim/rule judge".

### 10. Replay §7 fridge query after retrieve + judge yes-OR?

**Question.** Same Hebrew as §7: does Akhziv now fit?

**Setup.** Same `experiments` copy and scrape as §6–§7. No
further writes. `build_graph(stop_after="planner")` on
`רוצים קמפינג ליד הים ל3 אנשים ביום חמישי הבא ללילה אחד, עם מקרר`.
Dump `temp/akhziv_query3_fridge_after.json`. 115 s.

**Result.** Extract unchanged: party ≥ 3, `near the sea` ∧
`fridge` (site), 17–18 Sep. **2 fits / 19 rejected:** Akhziv
south (38) and north (37) tents. Fridge `why` is
`site_amenity=refrigerator` at −0.984 (`מקררים (3)`). Judge
`satisfies=true`, `satisfy_by=rule`; kitchen-fridge complaints
stay in `relevant_claims`. North's pack has the communal
listing, not south's `mini_refrigerator` huts.

**Decision.** §8–§9 hold under the original query. design.md
"Planner claim/rule judge".

### 11. Freeze occupancy for a 26-query planner gold set?

**Question.** Live `availability` moves every scrape. Can a
benchmark keep vacancies still while ingest/retrieve change?

**Setup.** `public.availability` on 2026-09-08: 226 nights,
2026-09-07–19, 14 vacant parks. Copy into
`experiments.availability_frozen` (no FK to `public`). Gold for
amenities/rules from parks.org.il pages the same day, not
`campsite_rules`. 26 queries in `evals/planner_v1.json`. No
planner run of the full set.

**Result.** Snapshot exists. Search reads
`TRIPPY_AVAILABILITY_TABLE`. Relative dates pin with
`TRIPPY_TODAY=2026-09-08`.

**Decision.** Freeze occupancy only. design.md "Planner claim/rule
judge".

## 2026-09-07

### 8. After rebuilding claims, can the judge sift amenity −0.7 listing hits?

**Question.** §7 kept amenity −0.8 because tent-as-desert would enter
`fits` and the judge never saw listing-only rows. After `just
populate-claims` with the despite-split few-shot, do desert location
claims retrieve at −0.6? If amenity retrieve is −0.7 and the judge
also sifts listing hits (`relevant_rules` + `satisfies`), does it keep
`electric_outlet` / PITCH hookup / fridge and drop tent-as-desert,
`electric_stove`, and caravan-bay-only hookup?

**Setup.** No writes. Amenity gate **−0.7**, claim gate **−0.6** top 5,
nearest 5 rules. Candidates = sites with a stated/site amenity ≤ −0.7
(not claim-gated). Queries: desert, electricity, quiet, fridge. One
235B call per (query, site), 8 workers, short `reason`. Gold satisfies
= a location-desert claim in the pack (atmosphere / in the desert; not
safari, animals, music, sandy pitches); electricity = outlet or PITCH
hookup or site-wide נקודות חשמל or an electricity claim, not
caravan-bay-only; quiet = a claim containing “quiet”; fridge =
refrigerator rule/hit or fridge/freezer claim. 60 pairs. Dump
`temp/judge_sift_amenity07.json`.

**Result.** **59/60 satisfies.** 349 s wall, ~40 s/call TTFT, ~880 in /
~45 out, $0.012. The miss is gold, not the model: Horshat Tal
`"Noise and music are prohibited at night"` for `"quiet"` — judge yes,
gold required the substring “quiet”.

Retrieve after rebuild did **not** add Mamshit to `"desert"`. The
despite sentence is still one glued row (`despite being in the desert
with winds`) at **−0.577**, outside −0.6. `"in the desert"` still
passes it at **−0.635**. New −0.6 noise: Mamshit sandy tent slots
**−0.612**, Hai-Bar desert safari **−0.604**. The judge dropped both.
Masada atmosphere **−0.748** is still the only `"desert"` location hit.
Yarkon electricity claim **−0.656** unchanged. Quiet and fridge claims
now retrieve (Nahal Amud / Tel Arad / Be'erot quiet; many refrigerator
claims).

Sift of listing −0.7: desert tents → no (except Masada via claim);
Besor caravan-bay hookup → no; bungalow `electric_outlet`, PITCH
hookup, site-wide נקודות חשמל → yes.

**Decision.** The judge can sift amenity −0.7 listing noise if listing
hits are in the call. Prod still −0.8 and still skips amenity-only
fits — not wired. Despite-split few-shot did not unglue the live
Mamshit review. design.md amenity gate; claims.md split.

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
