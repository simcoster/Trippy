# Plan to MVP

Dated 2026-09-10. Forward path only. `docs/PLAN.md` stays the historical log.

Extractor + planner + eval are in. The remaining sequence:

1. Recommender
2. Cloudify
3. Give people to try
4. Final fixes
5. Post on Medium + LinkedIn

Do them in that order. Do not start the next one until the current one is usable.

---

## 1. Recommender

Close **request → cited recommendation**. The graph already ends on
`recommender_node` (235B, Hebrew from planner `fits`). That is a first
draft, not the product: every turn re-runs extractor → planner; there is
no `last_recommendations`; “why that one?” and “anything cheaper?” start
a new search.

Done when a Hebrew ask gets a non-empty rec that only uses `fits`, cites
official `why` vs guest claims correctly, and a follow-up can talk about
the same recs without inventing sites.

Streamlit is the workbench. Telegram is still the intended live channel.

## 2. Cloudify

Host it so someone else can hit it. Sketch: `docs/scaling.md` and
`PLAN.md` §7 (API + scrape jobs, not K8s). Conversations cannot stay in
a process dict. Scrapes cannot stay “Omri ran `just` on a laptop.”

Cadence (the jobs *are* the product staying true):

| Job | How often |
|-----|-----------|
| Vacancy (`scrape-availability`) | once a day |
| Reviews + claims (`scrape-reviews`, `populate-claims`) | once a week |
| The rest (`scrape-sites`, `scrape-info`, place ids, …) | about once a month |

Done when a URL (Telegram webhook or equivalent) stays up without a
dev machine, secrets are not in the repo, and those jobs actually run
on that schedule.

## 3. Give people to try

Hand it to a small set of real campers. Not a launch. Watch what they
type, where the rec is wrong, and how long a turn feels.

Done when a handful of people have used it on real dates and we have
notes, not when the eval score moves.

## 4. Final fixes

Fix what those people actually hit. Known leftover (not a blocker to
start §1): H07 caravan bays as tent stays
(`PLAN.md` Open 2026-09-10). Everything else waits for real use.

Done when the embarrassing failures from §3 are gone or explicitly
wontfix.

## 5. Post on Medium + LinkedIn

Write it up after people have tried it. Architecture, eval, what the
judge does, what we would not do again. LinkedIn points at Medium.

Done when both are published.

---

Not this plan: group-trip prefs, a second booking source, notices /
PDF ingest, pin mixing as a research track. Those stay in `PLAN.md`.
