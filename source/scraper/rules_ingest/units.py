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

from openai import OpenAI

from source.scraper.amenity_enrichment.llm import EmbeddingLLMClient, LlmUsage
from source.scraper.amenity_enrichment.schemas import ALLOWED_CATEGORIES
from source.scraper.rules_ingest.db import ResolvedRule, upsert_campsite_rules
from source.scraper.rules_ingest.ingest import SiteReport, rules_from_sections
from source.scraper.rules_ingest.llm import SYSTEM_PROMPT, RuleExtractorLLMClient
from source.scraper.rules_ingest.resolve_conflicts import drop_redundant_permissions
from source.scraper.rules_ingest.sections import Section
from source.scraper.subjects.llm import SubjectAdjudicatorLLMClient
from source.scraper.subjects.resolve import DEFAULT_STORE, SubjectRef, SubjectStore

_CATEGORIES = ", ".join(sorted(ALLOWED_CATEGORIES))


# The production extractor prompt, reframed for one accommodation unit.
#
# In front, not appended, for the reason `subcamp_prompt` gives: the production
# prompt ends with its output schema, and an instruction placed after that reads
# as a note on the schema rather than as the frame for the task. The prefix has
# to countermand one line of it -- "This section describes the CAMPSITE AS A
# WHOLE. Ignore anything specific to one room or unit type" -- which is exactly
# backwards here.
#
# It is a CONSTANT, deliberately. It used to interpolate the unit name, which
# made the prompt differ per unit, so `ingest_unit_rules` built a fresh client
# per unit -- and each client builds its own transport, which costs 5.2 s on a
# machine with `TLS_TRUST_OS_STORE` set because `ssl_context()` reloads the OS
# certificate store every time. Six minutes over 68 units, for a name the user
# message already carries twice: `extract()` sends `Section: <name>`, and
# `unit_section` puts the name on the first line of the text.
UNIT_PROMPT = f"""UNIT SCOPE — apply this before every other rule below.

This text is the description of ONE accommodation unit at a campsite. The unit
is named on the `Section:` line below, and again on the first line of the text.
It is NOT a description of the campsite as a whole. Where the rules below say to
ignore anything specific to one room or unit type, read the opposite: everything
here is about this unit, and that is what you are extracting.

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
- Bed counts, sleeping capacity and how many rooms the listing joins are NOT
  yours. A separate pass reads the same text for those and stores them as
  columns on the unit, so emitting them here would state one fact twice, in two
  places that can disagree. Skip them however they are phrased:
    "4 מיטות (מתוכם: מיטה זוגית ומיטה דו קומתית)"   -> nothing
    "עד 4 לנים בכל בונגלו"                          -> nothing
    "2 חושות מחוברות עם דלת מקשרת"                   -> nothing
  Everything else in the same sentence is still yours: from
  "בכל חדר: 4 מיטות, מזרנים, כריות, מזגן" emit mattress, pillow and
  air_conditioning, and quote the whole sentence as the evidence span.
- Check-in and check-out times, minimum-night rules and pet policies ARE yours.
  They used to be columns on the unit and are now rules like any other.
- The unit's NAME is the first line of the text and is a description in its own
  right, sometimes the only one. Read it for what the unit is.
- Glossary: `אוהלים פרטיים` means the guest brings their own tent. The unit is a
  pitch to put it on, NOT a tent the site rents out. So
  `לינת שטח באוהלים פרטיים` is a tent-pitching area:
    -> tent_pitch  / amenity / true  / null / none
    -> tent        / amenity / false / null / none   (the guest supplies it)
  A unit whose name says `השכרת אוהל` is the opposite -- there the site does
  provide the tent.

{SYSTEM_PROMPT}"""


def unit_extractor(client: OpenAI | None = None) -> RuleExtractorLLMClient:
    """One extractor for every unit on every site.

    `ingest.py` has always built its extractor once. This is the same thing for
    units, and it is only possible because `UNIT_PROMPT` is a constant.
    """
    return RuleExtractorLLMClient(client, system_prompt=UNIT_PROMPT)


def unit_section(type_name: str, tooltip: str, *, source_url: str | None) -> Section:
    """One unit's description as a section the rule extractor can read.

    **The name is the first line of the text, not only the title.** It is part
    of the description and often the whole of it: `לינת שטח באוהלים פרטיים`
    states a tent-pitching area on its own, and Yehudia's panel is that heading
    and nothing else — read as a title alone it yielded no rules at all.

    It also makes the category statement quotable. `unit_prompt` asks for one
    statement naming the unit's category with the unit name as its evidence
    span; with the name only in the title, that span cited text that appeared
    nowhere in the section, which is 17 of the 27 non-verbatim spans measured
    over 18 sites.
    """
    name = type_name.strip()
    body = tooltip.strip()
    return Section(
        title=name,
        text=f"{name}\n{body}" if body else name,
        source_url=source_url,
    )


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
    extractor: RuleExtractorLLMClient | None = None,
) -> int:
    """Extract one unit's tooltip and write its rules. Returns rows upserted.

    `cache` is shared across the units of one site so they converge on the same
    subject ids; without it every unit builds its own vocabulary for the same
    words. `report` collects the resolution traces and the upsert collisions,
    so per-unit rows reach the conflict resolver and the run report on the same
    terms as site-level ones.

    Pass `extractor` -- one `unit_extractor()` for the whole run. Building one
    per unit costs 5.2 s each on a machine with `TLS_TRUST_OS_STORE` set,
    because every client builds its own transport and `ssl_context()` reloads
    the OS certificate store; that is six minutes over 68 units.
    """
    # No early return on an empty tooltip: the unit still has a name, and the
    # name is a description. Only a unit with no name at all has nothing to read.
    if not type_name.strip():
        return 0
    rules: list[ResolvedRule] = rules_from_sections(
        conn,
        [unit_section(type_name, tooltip or "", source_url=source_url)],
        extractor=extractor or unit_extractor(),
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
    drop_redundant_permissions(
        conn,
        campsite_id=campsite_id,
        rules=rules,
        accommodation_type_id=accommodation_type_id,
        table=rules_table,
        sink=report.redundant if report is not None else None,
        scope=type_name,
    )
    print(f"    {written} rule(s) upserted for {type_name!r}")
    return written
