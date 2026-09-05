"""LLM clients for subject resolution: adjudicate, then classify.

The judge runs on the 235B: the 30B was measured merging 4 of 6 direction pairs
(`child_max_age` into `child_min_age`, …) that the 235B kept apart on the same
prompt, for 2x the per-call cost -- cents, since these calls scale with
vocabulary growth. The classifier stays on the 30B: asked for the category of a
bare `dogs_allowed`, the 235B answered "amenity" 9 times in 10 where the 30B was
20/20 "rule" (experiments.md 2026-09-04 §5, §6). Temperature 0 on either model
is not a determinism guarantee. Both follow
following `source.scraper.info_site.classify.RateCardClassifier`. They only run
on a cache-and-alias miss, so the cost is bounded by vocabulary growth rather
than by ingest volume.
"""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from source.scraper.amenity_enrichment.llm import (
    QWEN_INSTRUCT_30B_MODEL,
    QWEN_INSTRUCT_MODEL,
    LlmUsage,
    _parse_json_payload,
    make_nebius_openai_client,
)
from source.scraper.subjects.schemas import AdjudicationPayload, ClassificationPayload

ADJUDICATE_SYSTEM_PROMPT = """You decide whether a new term names the SAME subject as one of the existing subjects.

Subjects are snake_case English labels for things a campsite provides or rules on.

Rules:
- Output valid JSON only, without markdown wrappers.
- Answer with the exact candidate string, copied character for character, or null.
- Match only true synonyms or spelling/pluralisation variants of the SAME subject.
- Do NOT match two different facts about the same noun. They are separate subjects.
- Two names are different subjects when they ask different KINDS of question about
  the noun -- a permission (may I?), an obligation (must I? must I pay?), a time,
  a price, a count, an age or night limit -- however alike the words:
  - "late_check_out_allowed" vs "late_check_out_end_time"   -> null
  - "barbecue_allowed" vs "barbecue"                         -> null
  - "last_dogs_entry_time" vs "dogs_allowed"                 -> null
  But different WORDS for the same kind of question are one subject. Match them:
  - "late_check_out_available" / "late_check_out_allowed" / "late_check_out_permitted"
  - "late_check_out_available_until" / "late_check_out_end_time"
  - "late_check_out_fee_applies" / "late_check_out_fee_required"
  - "towels_included" / "towels_provided" / "towels"
- Identical contexts are NOT evidence of sameness. One sentence often states
  several facts: "מטבח שדה (1) בשלב הזה בלי גז" names a field kitchen AND says it has
  no gas. When the term and a candidate quote the same sentence, judge the names
  alone -- and a term that adds a noun to the candidate is a PART of it, so null.
- Do NOT match a broader subject to a narrower one. If one name is the other plus
  a qualifying word, they are different subjects — the qualifier is usually the
  thing a guest is searching for, and the two often have opposite polarities.
  The extra word can sit anywhere in the name:
  - "accessible_toilets" vs "toilets"                          -> null
  - "gas_in_field_kitchen" vs "field_kitchen"                  -> null
  - "early_arrival_parking_allowed" vs "early_arrival_allowed" -> null
  - "late_check_out_saturday_allowed" vs "late_check_out_allowed" -> null
  Only a provision word (included / provided / available) may be ignored.
- A trailing included / provided / available says only that an amenity is on offer,
  which is recorded separately. It names no new subject, so treat "X_included" and
  "X" as the SAME subject when X is identical:
  - "towels_included" and "towels"                  -> match
  - "electric_hookup_included" and "electric_hookup" -> match
  BUT the noun still has to be the same:
  - "barbecue_equipment_included" and "barbecue"    -> null (equipment is not the
    activity; one asks what is supplied, the other names the thing itself)
- Word order does not make a new subject. "child_min_age" and "min_child_age" are
  one subject; so are "for_rent_mattress" and "mattress_for_rent". Match them.
- Two names that differ by a DIRECTION word state opposite bounds of one thing and
  are NEVER one subject, however identical the rest of the name: min/max,
  minimum/maximum, start/end, first/last, earliest/latest, open/close, entry/exit,
  arrival/departure, before/after, pickup/return.
  - "child_min_age" vs "child_max_age"                         -> null
  - "mattress_pickup_start_time" vs "mattress_pickup_end_time" -> null
  - "gate_open_time" vs "gate_close_time"                      -> null
- A different ACTOR or OBJECT is a different subject even when the predicate is the
  same. Vehicles are not guests and dogs are not guests:
  - "car_entry_time" vs "check_in_time"  -> null
  - "dogs_entry_time" vs "check_in_time" -> null
- When unsure, answer null. A wrong merge is worse than a duplicate.
- Context lines quote the sentence each subject was first read from. Use them:
  two subjects whose names look alike but whose contexts describe different
  things (a communal block vs something inside a room) are NOT the same subject.
- Each side may carry a `states:` line -- the polarity or number it asserts. Two
  statements read from ONE page that state different numbers, or opposite
  polarities, are two facts and never one subject. Across different campsites
  a different number is normal (one site has 1 freezer, another 2).

Examples:
- term "air_conoditioning", candidates ["air_conditioner", ...] -> {"match": "air_conditioner"}
- term "wifi", candidates ["wireless_internet", ...] -> {"match": "wireless_internet"}
- term "showers", candidates ["shower", ...] -> {"match": "shower"}
- term "last_dogs_entry_time", candidates ["dogs_allowed", ...] -> {"match": null}
  (one is a permission, the other is a deadline — different subjects)
- term "min_weekend_nights", candidates ["min_nights", ...] -> {"match": null}
  (the weekend qualifier makes it a different rule)
- term "hot_water_shower", candidates ["shower", ...] -> {"match": null}
- term "toilets" (context: "שירותים (15 תאי שירותי נשים ו- 15 תאי שירותי גברים)"),
  candidate "bathroom" (context: "בכל חדר: ... שירותים, מקלחת מים חמים")
  -> {"match": null}  (a shared block on the site is not a room's own bathroom)
- term "gas_stove_in_field_kitchen" (context: "מטבח שדה (1) בשלב הזה בלי גז"),
  candidate "field_kitchen" (context: the very same sentence)
  -> {"match": null}  (the stove is a part of the kitchen; the sentence says the
     kitchen exists and the gas does not -- two facts, two subjects)
- term "accessible_toilets", candidates ["toilets", ...] -> {"match": null}
  (accessibility is what someone is searching for; merging it hides that)
- term "mattresses_for_rent", candidates ["mattress", ...] -> {"match": null}
- term "water_hookup_included", candidates ["water_hookup", ...] -> {"match": "water_hookup"}
- term "min_child_age", candidates ["child_min_age", ...] -> {"match": "child_min_age"}
- term "barbecue_allowed", candidates ["barbecue", ...] -> {"match": null}
  (one asks whether grilling is permitted, the other names the equipment)
- term "late_check_out_fee", candidates ["late_check_out_available", ...] -> {"match": null}

Schema:
{"match": "<one of the candidate strings>" | null,
 "confidence": <number 0..1: how sure you are of THIS answer, match or null>}
"""

CLASSIFY_SYSTEM_PROMPT = """You name a new campsite subject and say what kind it is.

Given a raw term, and the sentence it was read from when available, return its
canonical label and category.

Rules:
- Output valid JSON only, without markdown wrappers.
- category is 1 for an amenity (something the site provides or does not provide:
  shower, refrigerator, electric_hookup, towels); 2 for a boolean rule (something
  a guest may, must, or must not do, answered yes or no: dogs_allowed,
  late_check_out_fee_required); 3 for a numeric rule (a time or a limit on a
  stay, answered by a number: quiet_hours_start, min_weekend_nights,
  check_out_time, adult_min_age).
- canonical_name is lower snake_case English, and states the predicate.
- ALWAYS phrase the name POSITIVELY. Negation is recorded separately, not in the
  name. Never emit not_/no_/cant_/cannot_/without_/_forbidden/_banned names.
  - "no dogs allowed"          -> canonical_name "dogs_allowed"
  - "bring your own towels"    -> canonical_name "towels_included"
  - "dogs cannot be unmuzzled" -> canonical_name "dogs_must_wear_a_muzzle"
- Direction belongs in the name, not in a separate field:
  min_weekend_nights, max_occupancy, check_out_time, latest_arrival_time,
  pool_min_age, last_dogs_entry_time.
- Name amenities in context: a caravan pitch has electric_hookup, not electricity.
- PREFER THE TERM YOU WERE GIVEN. If it is already lower snake_case, positively
  phrased and states a predicate, return it UNCHANGED. Only rewrite to fix a real
  problem: a misspelling, a plural, a negation, or a name that states no predicate.
  Do not reorder words, do not add words, do not drop words:
  - "child_max_age"                  -> "child_max_age"       (already fine)
  - "suitable_for_shabbat_observers" -> "suitable_for_shabbat_observers"
  - "picnic_tables_and_benches"      -> "picnic_tables"       (dropping "and_benches"
    is a real simplification; adding a suffix would not be)
  A term that comes back renamed for no reason forks the vocabulary: the same word
  must produce the same canonical name on every run, or the next campsite creates a
  duplicate subject instead of an alias.
- A trailing included / provided / available marks an AMENITY, never a rule — it
  says the site supplies something, and whether it does is recorded separately:
  - "towels_included", "electric_hookup_included" -> category 1
  Never append _included to a bare noun; "picnic_tables" is already the amenity.
- A bare noun naming a physical thing the site has is an amenity and stays a bare
  noun: refrigerator, cooler, shower, picnic_table. Do not append _allowed to it.
- Keep separate facts about one noun separate. barbecue_allowed (may I grill?) and
  barbecue_equipment_included (is a grill provided?) are two subjects, not one, and
  so are late_check_out_available and late_check_out_fee.
Schema:
{"category": 1 | 2 | 3, "canonical_name": "<snake_case>"}
"""


# A match below this is treated as "no match". Measured on 87 judge calls
# (experiments.md 2026-09-04 §12, §14): right answers 0.95 (once 1.0), wrong
# merges 0.30-0.85, one true merge at 0.80 (`picnic_table` vs
# `picnic_tables_and_benches`) -- the missed merge the project tolerates.
MATCH_MIN_CONFIDENCE = 0.9


@dataclass(frozen=True)
class Judgement:
    """What the judge answered, how sure it was, and whether the gate let it through."""

    match: str | None
    confidence: float | None
    accepted: bool


# TODO(rename): Adjudicator -> MergeJudge across all files, keeping each
# occurrence's letter case: SubjectAdjudicatorLLMClient -> SubjectMergeJudgeLLMClient,
# ADJUDICATE_SYSTEM_PROMPT -> MERGE_JUDGE_SYSTEM_PROMPT, `adjudicator` -> `merge_judge`,
# AdjudicationPayload -> MergeJudgementPayload, and the "ADJUDICATOR merged into"
# trace text. Its own branch and PR: a rename touches every caller and test.
class SubjectAdjudicatorLLMClient:
    """Is this term one of the nearest existing subjects, or a new one?

    Two jobs, reported as two cost roles: `pick_match` is the merge judge
    (`merge_judge`); `classify` answers amenity-or-rule for a term the caller
    gave no category for (`classify_amenity_or_rule`).
    """

    # The judge's model. `classify()` uses CLASSIFY_MODEL -- see the module
    # docstring for why they differ.
    MODEL = QWEN_INSTRUCT_MODEL
    CLASSIFY_MODEL = QWEN_INSTRUCT_30B_MODEL
    TEMPERATURE = 0

    def __init__(
        self,
        client: OpenAI | None = None,
        *,
        model: str | None = None,
        classify_model: str | None = None,
    ) -> None:
        self._client = client
        self.model = model or self.MODEL
        self.classify_model = classify_model or self.CLASSIFY_MODEL

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = make_nebius_openai_client()
        return self._client

    def pick_match(
        self,
        term: str,
        candidates: list[str],
        *,
        term_context: str | None = None,
        candidate_contexts: dict[str, str | None] | None = None,
        term_states: str | None = None,
        candidate_states: dict[str, str | None] | None = None,
        usage: LlmUsage | None = None,
        judgement_sink: list[Judgement] | None = None,
    ) -> str | None:
        """Return a candidate name, or None. Never returns a name not offered.

        `term_context` and `candidate_contexts` are the sentences the subjects
        were read from. Names alone cannot separate a communal toilet block from
        a room's own bathroom; the contexts can.

        `term_states` and `candidate_states` are what each side asserts --
        "qualifier=30 count", "polarity=true" -- the term's from its statement,
        a candidate's from its existing rows. With both sentences already in
        view the judge still merged a 30-person minimum into an 80-person one;
        shown the numbers it did not (experiments.md 2026-09-04 §10, §14).

        A match is returned only when the judge's own confidence is at least
        MATCH_MIN_CONFIDENCE: every wrong merge in 87 probed calls came back
        below 0.9 and every right answer at 0.95, so the gate only ever turns a
        merge into an insert -- the tolerable failure. A reply without a
        confidence is accepted as before. `judgement_sink` receives the raw
        answer, confidence and whether it was accepted, for the trace.
        """
        if not candidates:
            return None
        contexts = candidate_contexts or {}
        states = candidate_states or {}
        listed = "\n".join(
            f"- {name}"
            + (f"\n    context: {contexts[name]}" if contexts.get(name) else "")
            + (f"\n    states: {states[name]}" if states.get(name) else "")
            for name in candidates
        )
        asked = f"Term: {term}"
        if term_context:
            asked += f"\n    context: {term_context}"
        if term_states:
            asked += f"\n    states: {term_states}"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": ADJUDICATE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"{asked}\n\nExisting subjects:\n{listed}",
                },
            ],
            temperature=self.TEMPERATURE,
        )
        if usage is not None:
            usage.add_chat(response.usage, role="merge_judge", model=self.model)
        data = _parse_json_payload(response.choices[0].message.content or "")
        payload = AdjudicationPayload.model_validate(data)
        match = payload.match
        # Same guard as InfoWebsiteNameMatcher.pick_name: a name the model
        # invented is not a match, however confident it sounds.
        if match is None or match not in candidates:
            match = None
        confidence = payload.confidence
        accepted = match is not None and (
            confidence is None or confidence >= MATCH_MIN_CONFIDENCE
        )
        if judgement_sink is not None:
            judgement_sink.append(Judgement(match, confidence, accepted))
        return match if accepted else None

    def classify(
        self,
        term: str,
        *,
        context: str | None = None,
        usage: LlmUsage | None = None,
    ) -> ClassificationPayload:
        """Category (and a canonical name the resolver ignores) for a term.

        The extractor names subjects; the resolver calls this only when it was
        given no category. `canonical_name` is still returned for callers and
        tests that want the classifier's opinion, but nothing renames a subject.
        """
        asked = f"Term: {term}"
        if context:
            asked += f"\nContext: {context}"
        response = self.client.chat.completions.create(
            model=self.classify_model,
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": asked},
            ],
            temperature=self.TEMPERATURE,
        )
        if usage is not None:
            usage.add_chat(
                response.usage, role="classify_amenity_or_rule", model=self.classify_model
            )
        data = _parse_json_payload(response.choices[0].message.content or "")
        return ClassificationPayload.model_validate(data)
