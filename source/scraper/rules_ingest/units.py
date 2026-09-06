"""Per-unit rules from a booking-engine tooltip.

The availability scrape reads one tooltip per accommodation type. That text is
a section like any other, so it goes through the same path a page section does
-- `RuleExtractorLLMClient` for the statements, `resolve_subject` for the
subjects, `upsert_campsite_rules` for the write -- and lands in `campsite_rules`
with its `evidence_span`, `source_url` and `confidence` filled in. The only
difference from a site section is the scope: rows carry the type's
`accommodation_type_id` instead of NULL, and the prompt is reframed so the
extractor reads the text as being about ONE unit rather than the whole site.

Before this, the availability scrape had its own amenity path that wrote bare
`(subject_id, polarity)` rows: no sentence, no URL, no confidence, no
qualifier, and a merge judge that could not see what either side asserted.
"""

from __future__ import annotations

from source.scraper.amenity_enrichment.llm import EmbeddingLLMClient, LlmUsage
from source.scraper.amenity_enrichment.schemas import ALLOWED_CATEGORIES
from source.scraper.rules_ingest.db import ResolvedRule, upsert_campsite_rules
from source.scraper.rules_ingest.ingest import SiteReport, rules_from_sections
from source.scraper.rules_ingest.llm import SYSTEM_PROMPT, RuleExtractorLLMClient
from source.scraper.rules_ingest.sections import Section
from source.scraper.subjects.llm import SubjectAdjudicatorLLMClient
from source.scraper.subjects.resolve import DEFAULT_STORE, SubjectRef, SubjectStore

_CATEGORIES = ", ".join(sorted(ALLOWED_CATEGORIES))


def unit_prompt(type_name: str) -> str:
    """The production extractor prompt reframed for one accommodation type.

    In front, not appended, for the reason `subcamp_prompt` gives: the
    production prompt ends with its output schema, and an instruction after
    that reads as a note on the schema rather than as the frame for the task.

    The prefix has to countermand one line of the production prompt -- "This
    section describes the CAMPSITE AS A WHOLE. Ignore anything specific to one
    room or unit type" -- which is exactly backwards here.
    """
    return f"""UNIT SCOPE — apply this before every other rule below.

This text is the booking-engine description of ONE accommodation unit at a
campsite: `{type_name}`. It is NOT a description of the campsite as a whole.
Where the rules below say to ignore anything specific to one room or unit type,
read the opposite: everything here is about this unit, and that is what you are
extracting.

- Every statement is about this unit. Never name the unit in a subject: it is
  understood, and the row already records which unit it belongs to.
  `shower`, never `shower_in_bungalow` — unless the tooltip itself nests one
  amenity inside another, which the part/container rule below still covers.
- A facility the guest walks to (the site's toilet block, the campsite kitchen)
  is still stated here because it serves this unit. Extract it as an amenity;
  the site-level page states its own, and the two scopes are separate rows.
- Emit exactly one statement naming this unit's category, polarity true,
  category amenity, qualifier null: one of {_CATEGORIES}. Infer it from the
  unit name first and the text second, and use the unit name as its
  evidence_span. This is what makes the unit findable by shape.
- A named place, landmark or region — a lake, a crater, a desert, a beach —
  is kept AND generalised: emit the specific label and the geographic or
  feature type as separate amenity statements, so a generic query matches.
  Apply this to any place you recognise, not to a fixed list.
    "מול הכינרת"      -> near_the_kineret / amenity / true
                       -> near_a_lake      / amenity / true
                       -> near_water       / amenity / true
    "במכתש רמון"      -> near_ramon_crater / amenity / true
                       -> near_a_crater    / amenity / true
                       -> near_a_desert    / amenity / true
- Bed counts, occupancy, check-in and check-out times and minimum-night
  policies are read from this tooltip by a separate extractor and stored as
  columns on the unit. Extract them here too when the text states them: a
  number stated in a sentence is a statement like any other.

{SYSTEM_PROMPT}"""


def unit_section(type_name: str, tooltip: str, *, source_url: str | None) -> Section:
    """One unit's tooltip as a section the rule extractor can read.

    The title is the unit name, which is what the extractor is told the text
    describes and what every resolution trace is keyed by.
    """
    return Section(title=type_name, text=tooltip.strip(), source_url=source_url)


def ingest_unit_rules(
    conn,
    *,
    campsite_id: int,
    accommodation_type_id: int,
    type_name: str,
    tooltip: str,
    source_url: str | None = None,
    embedder: EmbeddingLLMClient,
    adjudicator: SubjectAdjudicatorLLMClient,
    store: SubjectStore = DEFAULT_STORE,
    rules_table: str = "campsite_rules",
    cache: dict[str, SubjectRef] | None = None,
    usage: LlmUsage | None = None,
    report: SiteReport | None = None,
) -> int:
    """Extract one unit's tooltip and write its rules. Returns rows upserted.

    `cache` is shared across the units of one site so they converge on the same
    subject ids; without it every unit builds its own vocabulary for the same
    words. `report` collects the resolution traces and the upsert collisions,
    so per-unit rows reach the conflict resolver and the run report on the same
    terms as site-level ones.
    """
    text = (tooltip or "").strip()
    if not text:
        return 0
    rules: list[ResolvedRule] = rules_from_sections(
        conn,
        [unit_section(type_name, text, source_url=source_url)],
        extractor=RuleExtractorLLMClient(system_prompt=unit_prompt(type_name)),
        embedder=embedder,
        adjudicator=adjudicator,
        store=store,
        cache=cache,
        usage=usage,
        trace_sink=report.traces if report is not None else None,
        campsite_id=campsite_id,
    )
    if not rules:
        print(f"    no statements extracted for {type_name!r}")
        return 0
    with conn.cursor() as cur:
        written = upsert_campsite_rules(
            cur,
            campsite_id=campsite_id,
            rules=rules,
            accommodation_type_id=accommodation_type_id,
            table=rules_table,
            dropped_sink=report.drops if report is not None else None,
        )
    print(f"    {written} rule(s) upserted for {type_name!r}")
    return written
