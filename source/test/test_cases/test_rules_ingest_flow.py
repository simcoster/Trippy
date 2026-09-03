"""Section -> statement -> resolved rule, and the site orchestration around it."""

from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from db.models import QualifierUnit
from source.scraper.rules_ingest.ingest import ingest_site, rules_from_sections
from source.scraper.rules_ingest.schemas import RuleExtract, RuleStatement
from source.scraper.rules_ingest.sections import Section
from source.scraper.rules_ingest.subcamps import Subcamp, hybrid_slice
from source.scraper.subjects.resolve import SubjectRef

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "info_site"
    / "hurshat_tal_camping.html"
)

SECTION = Section("מה בחניון?", "מקררים (11)", "https://x")


def make_extractor(*extracts):
    extractor = MagicMock()
    extractor.extract.side_effect = list(extracts)
    return extractor


def extract_of(*statements) -> RuleExtract:
    return RuleExtract(statements=list(statements))


def test_the_extractor_category_reaches_the_resolver():
    """It is the only signal that keeps a rule out of an amenity's candidates."""
    extractor = make_extractor(
        extract_of(
            RuleStatement(subject="barbecue_allowed", category=2, polarity=True),
            RuleStatement(
                subject="barbecue_equipment_included", category=1, polarity=False
            ),
        )
    )
    with patch(
        "source.scraper.rules_ingest.ingest.resolve_subject",
        side_effect=[SubjectRef(1, "barbecue_allowed", 2), SubjectRef(2, "x", 1)],
    ) as resolve:
        rules_from_sections(
            MagicMock(),
            [SECTION],
            extractor=extractor,
            embedder=MagicMock(),
            adjudicator=MagicMock(),
        )

    assert [c.kwargs["category"] for c in resolve.call_args_list] == [2, 1]


def test_resolution_traces_are_collected_when_a_sink_is_given():
    extractor = make_extractor(extract_of(RuleStatement(subject="shower", polarity=True)))
    sink: list = []

    def record(_conn, _term, **kwargs):
        kwargs["trace_sink"].append("traced")
        return SubjectRef(6, "shower", 1)

    with patch(
        "source.scraper.rules_ingest.ingest.resolve_subject", side_effect=record
    ):
        rules_from_sections(
            MagicMock(),
            [SECTION],
            extractor=extractor,
            embedder=MagicMock(),
            adjudicator=MagicMock(),
            trace_sink=sink,
        )

    assert sink == ["traced"]


def test_statements_become_rules_carrying_the_section_source_url():
    extractor = make_extractor(
        extract_of(
            RuleStatement(
                subject="refrigerator",
                polarity=True,
                qualifier=Decimal("11"),
                qualifier_unit=int(QualifierUnit.COUNT),
                evidence_span="מקררים (11)",
                confidence=0.9,
            )
        )
    )
    with patch(
        "source.scraper.rules_ingest.ingest.resolve_subject",
        return_value=SubjectRef(3, "refrigerator", 1),
    ):
        rules = rules_from_sections(
            MagicMock(),
            [SECTION],
            extractor=extractor,
            embedder=MagicMock(),
            adjudicator=MagicMock(),
        )

    assert len(rules) == 1
    rule = rules[0]
    assert rule.subject_id == 3
    assert rule.polarity is True
    assert rule.qualifier == Decimal("11")
    assert rule.qualifier_unit == int(QualifierUnit.COUNT)
    assert rule.source_url == "https://x"
    assert rule.evidence_span == "מקררים (11)"


def test_a_de_negated_subject_overrides_the_extractor_polarity():
    """"dogs_not_allowed" with polarity true is the model contradicting itself."""
    extractor = make_extractor(
        extract_of(RuleStatement(subject="dogs_not_allowed", polarity=True))
    )
    with patch(
        "source.scraper.rules_ingest.ingest.resolve_subject",
        return_value=SubjectRef(4, "dogs_allowed", 2, implied_polarity=False),
    ):
        rules = rules_from_sections(
            MagicMock(),
            [SECTION],
            extractor=extractor,
            embedder=MagicMock(),
            adjudicator=MagicMock(),
        )

    assert rules[0].polarity is False


def test_extractor_polarity_survives_when_the_name_implied_nothing():
    extractor = make_extractor(
        extract_of(RuleStatement(subject="towels_included", polarity=False))
    )
    with patch(
        "source.scraper.rules_ingest.ingest.resolve_subject",
        return_value=SubjectRef(5, "towels_included", 1, implied_polarity=None),
    ):
        rules = rules_from_sections(
            MagicMock(),
            [SECTION],
            extractor=extractor,
            embedder=MagicMock(),
            adjudicator=MagicMock(),
        )

    assert rules[0].polarity is False


def test_an_unresolvable_subject_is_skipped_not_fatal():
    extractor = make_extractor(
        extract_of(
            RuleStatement(subject="cant_be_without_muzzle", polarity=True),
            RuleStatement(subject="shower", polarity=True),
        )
    )
    with patch(
        "source.scraper.rules_ingest.ingest.resolve_subject",
        side_effect=[None, SubjectRef(6, "shower", 1)],
    ):
        rules = rules_from_sections(
            MagicMock(),
            [SECTION],
            extractor=extractor,
            embedder=MagicMock(),
            adjudicator=MagicMock(),
        )

    assert [r.subject_id for r in rules] == [6]


def test_one_failing_section_does_not_lose_the_others():
    extractor = MagicMock()
    extractor.extract.side_effect = [
        RuntimeError("model timed out"),
        extract_of(RuleStatement(subject="shower", polarity=True)),
    ]
    with patch(
        "source.scraper.rules_ingest.ingest.resolve_subject",
        return_value=SubjectRef(6, "shower", 1),
    ):
        rules = rules_from_sections(
            MagicMock(),
            [Section("a", "text-a"), Section("b", "text-b")],
            extractor=extractor,
            embedder=MagicMock(),
            adjudicator=MagicMock(),
        )

    assert len(rules) == 1


def test_the_resolver_cache_is_shared_across_sections():
    extractor = make_extractor(
        extract_of(RuleStatement(subject="shower", polarity=True)),
        extract_of(RuleStatement(subject="shower", polarity=True)),
    )
    seen: list[dict] = []

    def record(_conn, _term, **kwargs):
        seen.append(kwargs["cache"])
        return SubjectRef(6, "shower", 1)

    with patch(
        "source.scraper.rules_ingest.ingest.resolve_subject", side_effect=record
    ):
        rules_from_sections(
            MagicMock(),
            [Section("a", "text-a"), Section("b", "text-b")],
            extractor=extractor,
            embedder=MagicMock(),
            adjudicator=MagicMock(),
        )

    assert len(seen) == 2 and seen[0] is seen[1]


def test_ingest_site_writes_the_rules_it_extracted():
    html = FIXTURE.read_text(encoding="utf-8")
    site = {"id": 5, "name": "חורשת טל", "url": "https://x"}
    conn = MagicMock()

    with (
        patch(
            "source.scraper.rules_ingest.ingest.rules_from_sections",
            return_value=[MagicMock(subject_id=1)],
        ),
        patch(
            "source.scraper.rules_ingest.ingest.upsert_campsite_rules", return_value=1
        ) as upsert,
    ):
        written = ingest_site(
            conn,
            site,
            html,
            extractor=MagicMock(),
            embedder=MagicMock(),
            adjudicator=MagicMock(),
        )

    assert written == 1
    assert upsert.call_args.kwargs["campsite_id"] == 5


def test_ingest_site_writes_nothing_when_no_statements_are_extracted():
    html = FIXTURE.read_text(encoding="utf-8")
    with (
        patch(
            "source.scraper.rules_ingest.ingest.rules_from_sections", return_value=[]
        ),
        patch("source.scraper.rules_ingest.ingest.upsert_campsite_rules") as upsert,
    ):
        written = ingest_site(
            MagicMock(),
            {"id": 5, "name": "x", "url": "https://x"},
            html,
            extractor=MagicMock(),
            embedder=MagicMock(),
            adjudicator=MagicMock(),
        )

    assert written == 0
    upsert.assert_not_called()


def test_ingest_site_on_a_page_with_no_sections_does_nothing():
    with patch("source.scraper.rules_ingest.ingest.rules_from_sections") as extract:
        written = ingest_site(
            MagicMock(),
            {"id": 5, "name": "x", "url": "https://x"},
            "<html><body>nothing here</body></html>",
            extractor=MagicMock(),
            embedder=MagicMock(),
            adjudicator=MagicMock(),
        )
    assert written == 0
    extract.assert_not_called()


def test_the_statement_context_carries_the_section_and_the_sentence():
    """The sameness judge needs to know a `toilets` came from a site list."""
    extractor = make_extractor(
        extract_of(
            RuleStatement(
                subject="toilets",
                category=1,
                polarity=True,
                evidence_span="שירותים (15 תאי שירותי נשים ו- 15 תאי שירותי גברים)",
            )
        )
    )
    with patch(
        "source.scraper.rules_ingest.ingest.resolve_subject",
        return_value=SubjectRef(1, "toilets", 1),
    ) as resolve:
        rules_from_sections(
            MagicMock(),
            [Section("מה בחניון?", "...", "https://x")],
            extractor=extractor,
            embedder=MagicMock(),
            adjudicator=MagicMock(),
        )

    context = resolve.call_args.kwargs["context"]
    assert context.startswith("מה בחניון?: ")
    assert "15 תאי שירותי נשים" in context


def test_context_falls_back_to_the_section_title():
    extractor = make_extractor(extract_of(RuleStatement(subject="shower", polarity=True)))
    with patch(
        "source.scraper.rules_ingest.ingest.resolve_subject",
        return_value=SubjectRef(1, "shower", 1),
    ) as resolve:
        rules_from_sections(
            MagicMock(),
            [Section("מה בחניון?", "...")],
            extractor=extractor,
            embedder=MagicMock(),
            adjudicator=MagicMock(),
        )
    assert resolve.call_args.kwargs["context"] == "מה בחניון?"


def test_the_store_reaches_the_resolver():
    from source.scraper.subjects.resolve import SubjectStore

    store = SubjectStore(table="test_subject_vectors", has_context=True)
    extractor = make_extractor(extract_of(RuleStatement(subject="shower", polarity=True)))
    with patch(
        "source.scraper.rules_ingest.ingest.resolve_subject",
        return_value=SubjectRef(1, "shower", 1),
    ) as resolve:
        rules_from_sections(
            MagicMock(),
            [SECTION],
            extractor=extractor,
            embedder=MagicMock(),
            adjudicator=MagicMock(),
            store=store,
        )
    assert resolve.call_args.kwargs["store"] is store


def test_a_statement_with_neither_polarity_nor_qualifier_is_dropped():
    """It would add a permanent subject no query can use."""
    extractor = make_extractor(
        extract_of(
            RuleStatement(subject="service_center_on_demand", category=2),
            RuleStatement(subject="visitor_service_center", category=1, polarity=True),
        )
    )
    with patch(
        "source.scraper.rules_ingest.ingest.resolve_subject",
        return_value=SubjectRef(1, "visitor_service_center", 1),
    ) as resolve:
        rules = rules_from_sections(
            MagicMock(),
            [SECTION],
            extractor=extractor,
            embedder=MagicMock(),
            adjudicator=MagicMock(),
        )

    assert len(rules) == 1
    assert resolve.call_count == 1
    assert resolve.call_args.args[1] == "visitor_service_center"


@pytest.mark.parametrize(
    "statement",
    [
        RuleStatement(subject="dogs_allowed", category=2, polarity=False),
        RuleStatement(subject="max_occupancy", category=2, qualifier=Decimal("4")),
    ],
)
def test_a_statement_asserting_either_one_survives(statement):
    extractor = make_extractor(extract_of(statement))
    with patch(
        "source.scraper.rules_ingest.ingest.resolve_subject",
        return_value=SubjectRef(1, statement.subject, 2),
    ):
        rules = rules_from_sections(
            MagicMock(),
            [SECTION],
            extractor=extractor,
            embedder=MagicMock(),
            adjudicator=MagicMock(),
        )
    assert len(rules) == 1


# --- subcamps: one pass per child campsite ------------------------------------

AKHZIV = {"id": 2, "name": "אכזיב", "url": "https://x"}
SUBCAMP_ROWS = [
    (37, "אכזיב – חניון צפוני", {"heading": "חניון צפוני", "aliases": ["אכזיב צפון"]}),
    (38, "אכזיב – חניון דרומי", {"heading": "חניון דרומי", "aliases": ["אכזיב דרום"]}),
]


def subcamp_conn(rows=SUBCAMP_ROWS):
    """A connection whose `load_subcamps` query returns these child rows."""
    cursor = MagicMock()
    cursor.fetchall.return_value = list(rows)
    cursor.__enter__ = lambda self: cursor
    cursor.__exit__ = lambda *a: False
    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn


def run_ingest_site(conn):
    """ingest_site over the fixture page, with extraction and writes stubbed."""
    html = FIXTURE.read_text(encoding="utf-8")
    with (
        patch(
            "source.scraper.rules_ingest.ingest.rules_from_sections",
            return_value=[MagicMock(subject_id=1)],
        ) as extract,
        patch(
            "source.scraper.rules_ingest.ingest.upsert_campsite_rules", return_value=1
        ) as upsert,
    ):
        written = ingest_site(
            conn,
            AKHZIV,
            html,
            extractor=MagicMock(),
            embedder=MagicMock(),
            adjudicator=MagicMock(),
        )
    return written, extract, upsert


def test_a_split_site_is_ingested_once_per_subcamp():
    written, extract, upsert = run_ingest_site(subcamp_conn())

    assert written == 2
    assert extract.call_count == 2
    # Written to the children, never to the parent — that is what lets
    # campsite_rules keep its key with no subcamp dimension.
    assert [c.kwargs["campsite_id"] for c in upsert.call_args_list] == [37, 38]


def test_every_subcamp_pass_shares_one_subject_cache():
    """Otherwise the two halves of one site build parallel vocabularies."""
    _written, extract, _upsert = run_ingest_site(subcamp_conn())

    caches = [c.kwargs["cache"] for c in extract.call_args_list]
    assert len(caches) == 2
    assert caches[0] is caches[1]


def test_each_subcamp_pass_gets_its_own_filtered_prompt():
    _written, extract, _upsert = run_ingest_site(subcamp_conn())

    prompts = [c.kwargs["extractor"].system_prompt for c in extract.call_args_list]
    assert "חניון צפוני` ONLY" in prompts[0]
    assert "חניון דרומי` ONLY" in prompts[1]
    # Each pass is told which areas are *not* its own.
    assert "חניון דרומי" in prompts[0] and "חניון צפוני" in prompts[1]


def test_an_ordinary_site_is_still_ingested_in_one_pass():
    """The other 17 sites must be untouched by the subcamp path: one extraction,
    written against the site's own campsite id rather than any child's."""
    written, extract, upsert = run_ingest_site(subcamp_conn(rows=[]))

    assert written == 1
    assert extract.call_count == 1
    assert upsert.call_args.kwargs["campsite_id"] == 2


# --- subcamps: what each pass is allowed to read -------------------------------

# Aliases match the shipped config: the definite forms (`חניון הצפוני`) are what
# running prose uses, and without them a passing mention goes unnoticed —
# see test_the_definite_form_alias_is_what_catches_a_passing_mention.
NORTH = Subcamp(37, "north", "חניון צפוני", ("חניון הצפוני", "אכזיב צפון"))
SOUTH = Subcamp(38, "south", "חניון דרומי", ("חניון הדרומי", "אכזיב דרום"))
BOTH = [NORTH, SOUTH]

# The shape of the real page: a heading, a list, a heading, a list, then a
# site-wide trailer, one line of which names a single subcamp in passing.
PANEL = "\n".join(
    [
        "חניון צפוני",
        "מקררים (3)",
        "ספסלים ושולחנות פיקניק (80)",
        "חניון דרומי",
        "מקררים (3)",
        "ספסלים ושולחנות פיקניק (60)",
        "מרכז שירות למבקר (בחניון הצפוני)",
        "מזרנים (100)",
    ]
)


def test_both_lists_survive_the_slice():
    """Cutting to one list makes the extractor sum gendered counts (4/4 runs).

    So the slice keeps every list and every heading, and lets the prompt do the
    separating — see source/scraper/rules_ingest/subcamps.py.
    """
    kept = hybrid_slice(PANEL, NORTH, BOTH)

    assert "ספסלים ושולחנות פיקניק (80)" in kept
    assert "ספסלים ושולחנות פיקניק (60)" in kept
    assert "חניון צפוני" in kept and "חניון דרומי" in kept


def test_a_line_naming_another_subcamp_in_passing_is_dropped():
    """The southern pass claimed the northern service centre without this."""
    assert "מרכז שירות למבקר" in hybrid_slice(PANEL, NORTH, BOTH)
    assert "מרכז שירות למבקר" not in hybrid_slice(PANEL, SOUTH, BOTH)


def test_the_definite_form_alias_is_what_catches_a_passing_mention():
    """`(בחניון הצפוני)` contains `חניון הצפוני`, never the bare heading.

    Drop that alias and the northern service centre reaches both subcamps, which
    is why config.json carries it and why the config validator checks every
    alias still occurs on the page.
    """
    bare = [
        Subcamp(37, "north", "חניון צפוני", ()),
        Subcamp(38, "south", "חניון דרומי", ()),
    ]
    assert "מרכז שירות למבקר" in hybrid_slice(PANEL, bare[1], bare)
    assert "מרכז שירות למבקר" not in hybrid_slice(PANEL, SOUTH, BOTH)


def test_site_wide_lines_reach_every_subcamp():
    for subcamp in BOTH:
        assert "מזרנים (100)" in hybrid_slice(PANEL, subcamp, BOTH)


def test_a_line_naming_both_subcamps_is_cut_to_this_one_s_clause():
    """`נגישות` states both in one sentence, with no markup to cut on."""
    line = "אכזיב צפון: חניה, שירותים, ושביל לאזור הקמת האוהלים אכזיב דרום: חניה, שתי חושות"

    north = hybrid_slice(line, NORTH, BOTH)
    south = hybrid_slice(line, SOUTH, BOTH)

    assert "שביל לאזור הקמת האוהלים" in north and "שתי חושות" not in north
    assert "שתי חושות" in south and "שביל לאזור הקמת האוהלים" not in south


def test_the_clause_label_is_stripped_so_it_is_not_read_as_a_fact():
    """`northern_parking_area` reached the production dictionary this way."""
    line = "אכזיב צפון: חניה מונגשת אכזיב דרום: שבילים"

    assert not hybrid_slice(line, NORTH, BOTH).startswith("אכזיב צפון")
    assert "חניה מונגשת" in hybrid_slice(line, NORTH, BOTH)
