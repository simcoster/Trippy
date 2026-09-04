"""resolve_subject: alias hits are free, near-misses merge, novelties insert."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from source.scraper.subjects.resolve import (
    DEFAULT_STORE,
    MATCH_MAX_DISTANCE,
    REJECT_FAR,
    REJECT_OPPOSED,
    Candidate,
    ResolutionTrace,
    SubjectRef,
    SubjectStore,
    format_trace,
    resolve_subject,
)

CLOSE = MATCH_MAX_DISTANCE - 0.1  # more negative == nearer under <#>
FAR = MATCH_MAX_DISTANCE + 0.1


class FakeCursor:
    """Dispatches on the SQL it is handed, so tests read as intent not order.

    `nearest` rows are (id, name, category, distance) or that plus a context.
    The category filter is applied here because it is applied in the SQL: a
    post-filter would spend all five slots on the wrong category.
    """

    def __init__(self, *, alias_hit=None, name_hit=None, nearest=(), inserted=None):
        self.alias_hit = alias_hit
        self.name_hit = name_hit
        self.nearest = [tuple(r) + (None,) * (5 - len(r)) for r in nearest]
        self.inserted = inserted or (99, "new_subject", 2)
        self.statements: list[tuple[str, object]] = []
        self._result = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.statements.append((sql, params))
        if "aliases @> ARRAY[%s]" in sql and sql.strip().startswith("SELECT"):
            self._result = self.alias_hit
        elif "WHERE name = %s" in sql:
            self._result = self.name_hit
        elif "ORDER BY embedding" in sql:
            rows = self.nearest
            if "category = %s" in sql:
                wanted = params[1]
                rows = [r for r in rows if r[2] == wanted]
            self._result = rows
        elif sql.strip().startswith("INSERT"):
            self._result = self.inserted
        else:
            self._result = None

    def fetchone(self):
        return self._result

    def fetchall(self):
        return self._result or []

    def sql_matching(self, needle: str) -> list[tuple[str, object]]:
        return [s for s in self.statements if needle in s[0]]


def make_conn(cursor: FakeCursor) -> MagicMock:
    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn


def make_embedder(dims: int = 4):
    embedder = MagicMock()
    embedder.embed.side_effect = lambda texts, **kw: [[0.1] * dims for _ in texts]
    return embedder


def make_adjudicator(*, match=None, category=2, canonical_name="new_subject"):
    adjudicator = MagicMock()
    adjudicator.pick_match.return_value = match
    adjudicator.classify.return_value = SimpleNamespace(
        category=category, canonical_name=canonical_name
    )
    return adjudicator


def test_exact_alias_hit_costs_no_llm_and_no_embedding():
    cursor = FakeCursor(alias_hit=(7, "air_conditioning", 1))
    embedder, adjudicator = make_embedder(), make_adjudicator()

    ref = resolve_subject(
        make_conn(cursor),
        "Air Conditioning",
        embedder=embedder,
        adjudicator=adjudicator,
        verbose=False,
    )

    assert ref == SubjectRef(7, "air_conditioning", 1, None)
    embedder.embed.assert_not_called()
    adjudicator.pick_match.assert_not_called()
    adjudicator.classify.assert_not_called()


def test_adjudicated_match_appends_an_alias_instead_of_inserting():
    cursor = FakeCursor(
        alias_hit=None,
        nearest=[(7, "air_conditioning", 1, CLOSE), (8, "heating", 1, CLOSE)],
    )
    adjudicator = make_adjudicator(match="air_conditioning")

    ref = resolve_subject(
        make_conn(cursor),
        "air_conoditioning",
        embedder=make_embedder(),
        adjudicator=adjudicator,
        verbose=False,
    )

    assert ref.id == 7
    assert ref.name == "air_conditioning"
    adjudicator.classify.assert_not_called()
    assert not cursor.sql_matching("INSERT")
    updates = cursor.sql_matching("array_append")
    assert len(updates) == 1
    assert updates[0][1] == {"id": 7, "alias": "air_conoditioning"}


def test_adjudicator_only_sees_candidates_within_the_distance_threshold():
    cursor = FakeCursor(
        nearest=[(7, "near_enough", 1, CLOSE), (8, "too_far", 1, FAR)],
    )
    adjudicator = make_adjudicator(match=None)

    resolve_subject(
        make_conn(cursor),
        "brand_new_thing",
        embedder=make_embedder(),
        adjudicator=adjudicator,
        verbose=False,
    )

    offered = adjudicator.pick_match.call_args.args[1]
    assert offered == ["near_enough"]


def test_no_nearby_candidates_skips_adjudication_entirely():
    cursor = FakeCursor(nearest=[(8, "too_far", 1, FAR)])
    adjudicator = make_adjudicator()

    resolve_subject(
        make_conn(cursor),
        "brand_new_thing",
        embedder=make_embedder(),
        adjudicator=adjudicator,
        verbose=False,
    )

    adjudicator.pick_match.assert_not_called()
    adjudicator.classify.assert_called_once()


def test_rejected_match_inserts_the_term_itself_and_never_renames_it():
    """The extractor names subjects. The classifier's name is ignored; only its
    category is used, and only because none was given here.

    Renaming on insert was measured turning `dogs_entry_time` ("from 16:00")
    into `last_dogs_entry_time`; a wrong name on a real fact is as bad as a
    wrong merge.
    """
    cursor = FakeCursor(
        nearest=[(7, "dogs_allowed", 2, CLOSE)],
        inserted=(42, "last_dog_entry_hour", 2),
    )
    embedder = make_embedder()
    adjudicator = make_adjudicator(
        match=None, category=2, canonical_name="last_dogs_entry_time"  # ignored
    )

    ref = resolve_subject(
        make_conn(cursor),
        "last_dog_entry_hour",
        embedder=embedder,
        adjudicator=adjudicator,
        verbose=False,
    )

    assert ref == SubjectRef(42, "last_dog_entry_hour", 2, None)
    inserts = cursor.sql_matching("INSERT")
    assert len(inserts) == 1
    params = inserts[0][1]
    assert params["name"] == "last_dog_entry_hour"
    assert params["category"] == 2
    assert params["aliases"] == ["last_dog_entry_hour"]
    # One embedding: the probe that ran the neighbour search is stored as the
    # row's vector, since the term is also the name.
    assert [c.args[0] for c in embedder.embed.call_args_list] == [
        ["last_dog_entry_hour"],
    ]
    adjudicator.classify.assert_called_once()


def test_negative_term_resolves_positively_and_reports_the_implied_polarity():
    cursor = FakeCursor(alias_hit=(7, "dogs_allowed", 2))

    ref = resolve_subject(
        make_conn(cursor),
        "dogs_not_allowed",
        embedder=make_embedder(),
        adjudicator=make_adjudicator(),
        verbose=False,
    )

    assert ref.name == "dogs_allowed"
    assert ref.implied_polarity is False
    # The lookup went out under the positive name.
    assert cursor.sql_matching("aliases @> ARRAY[%s]")[0][1] == ("dogs_allowed",)


def test_unrewritable_negative_term_is_dropped():
    cursor = FakeCursor()
    adjudicator = make_adjudicator()

    ref = resolve_subject(
        make_conn(cursor),
        "dogs_cant_be_without_muzzle",
        embedder=make_embedder(),
        adjudicator=adjudicator,
        verbose=False,
    )

    assert ref is None
    assert cursor.statements == []
    adjudicator.pick_match.assert_not_called()


@pytest.mark.parametrize("term", ["", "   "])
def test_empty_term_is_dropped(term):
    cursor = FakeCursor()
    assert (
        resolve_subject(
            make_conn(cursor),
            term,
            embedder=make_embedder(),
            adjudicator=make_adjudicator(),
            verbose=False,
        )
        is None
    )
    assert cursor.statements == []


def test_cache_short_circuits_a_repeated_term():
    cursor = FakeCursor(alias_hit=(7, "shower", 1))
    embedder, adjudicator = make_embedder(), make_adjudicator()
    cache: dict = {}
    conn = make_conn(cursor)

    first = resolve_subject(
        conn, "Showers", embedder=embedder, adjudicator=adjudicator, cache=cache, verbose=False
    )
    second = resolve_subject(
        conn, "showers", embedder=embedder, adjudicator=adjudicator, cache=cache, verbose=False
    )

    assert first == second
    assert len(cursor.sql_matching("aliases @> ARRAY[%s]")) == 1


def test_a_candidate_with_a_different_predicate_is_offered_and_the_judge_rejects_it():
    """A permission and a price about one noun are two subjects.

    A suffix list used to keep this pair from the judge altogether; that gate
    fragmented `late_check_out_*` and was removed. Now the pair is offered, the
    judge says no, and the rejected neighbour is handed to the classifier so the
    new name is chosen to stay clear of it.
    """
    cursor = FakeCursor(
        nearest=[(7, "late_check_out_available", 2, CLOSE)],
        inserted=(50, "late_check_out_fee", 2),
    )
    adjudicator = make_adjudicator(
        match=None, category=2, canonical_name="late_check_out_fee"
    )

    resolve_subject(
        make_conn(cursor),
        "late_check_out_fee",
        embedder=make_embedder(),
        adjudicator=adjudicator,
        verbose=False,
    )

    adjudicator.pick_match.assert_called_once()
    assert adjudicator.pick_match.call_args.args[1] == ["late_check_out_available"]
    # No category was given, so the classifier is asked for one -- and only that.
    adjudicator.classify.assert_called_once()
    assert cursor.sql_matching("INSERT")[0][1]["name"] == "late_check_out_fee"


def test_candidates_sharing_the_predicate_still_reach_the_adjudicator():
    """Same predicate, different noun, no antonym — the judge decides."""
    cursor = FakeCursor(nearest=[(7, "child_min_age", 2, CLOSE)])
    adjudicator = make_adjudicator(match=None)

    resolve_subject(
        make_conn(cursor),
        "adult_min_age",
        embedder=make_embedder(),
        adjudicator=adjudicator,
        verbose=False,
    )

    assert adjudicator.pick_match.call_args.args[1] == ["child_min_age"]


# --- category: the extractor knows which side of the line a term falls on -----


def test_a_rule_never_merges_into_an_amenity():
    """barbecue_allowed sits closer to barbecue than the right amenity does."""
    cursor = FakeCursor(
        nearest=[
            (7, "barbecue", 1, -0.99),
            (8, "barbecue_pit", 1, CLOSE),
        ],
        inserted=(60, "barbecue_allowed", 2),
    )
    adjudicator = make_adjudicator(
        match=None, category=2, canonical_name="barbecue_allowed"
    )

    sink: list[ResolutionTrace] = []
    ref = resolve_subject(
        make_conn(cursor),
        "barbecue_allowed",
        embedder=make_embedder(),
        adjudicator=adjudicator,
        category=2,
        trace_sink=sink,
        verbose=False,
    )

    adjudicator.pick_match.assert_not_called()
    assert ref.id == 60
    assert cursor.sql_matching("INSERT")[0][1]["category"] == 2
    # The amenities were excluded by the query, not filtered afterwards.
    assert sink[0].candidates == []
    nn_sql, nn_params = cursor.sql_matching("ORDER BY embedding")[0]
    assert "category = %s" in nn_sql
    assert nn_params[1] == 2


def test_an_amenity_never_merges_into_a_rule():
    cursor = FakeCursor(
        nearest=[(8, "barbecue_allowed", 2, -0.99)],
        inserted=(61, "barbecue_equipment_included", 1),
    )
    adjudicator = make_adjudicator(
        match=None, category=1, canonical_name="barbecue_equipment_included"
    )

    resolve_subject(
        make_conn(cursor),
        "barbecue_equipment_included",
        embedder=make_embedder(),
        adjudicator=adjudicator,
        category=1,
        verbose=False,
    )

    adjudicator.pick_match.assert_not_called()


def test_the_extractor_category_beats_the_classifier_guess():
    """The extractor read the sentence; the classifier saw one word."""
    cursor = FakeCursor(nearest=[], inserted=(62, "cooler", 1))
    # classify() insists this is a rule; the extractor said amenity.
    adjudicator = make_adjudicator(match=None, category=2, canonical_name="cooler")

    resolve_subject(
        make_conn(cursor),
        "coolers",
        embedder=make_embedder(),
        adjudicator=adjudicator,
        category=1,
        verbose=False,
    )

    assert cursor.sql_matching("INSERT")[0][1]["category"] == 1


def test_without_a_category_every_row_is_a_candidate():
    """ensure_amenities pins a category; other callers may not."""
    cursor = FakeCursor(nearest=[(8, "shower", 1, CLOSE), (9, "toilets", 2, CLOSE)])
    adjudicator = make_adjudicator(match=None)

    resolve_subject(
        make_conn(cursor),
        "showers_block",
        embedder=make_embedder(),
        adjudicator=adjudicator,
        verbose=False,
    )

    assert adjudicator.pick_match.call_args.args[1] == ["shower", "toilets"]


# --- trace --------------------------------------------------------------------


def test_trace_records_every_candidate_and_why_it_was_dropped():
    cursor = FakeCursor(
        nearest=[
            (7, "shower", 1, CLOSE),  # offered
            (8, "shower_allowed", 1, CLOSE),  # offered too: predicates are the judge's call
            (9, "toilets", 2, CLOSE),  # excluded by the query itself
            (10, "far_thing", 1, FAR),  # distance
        ],
    )
    sink: list[ResolutionTrace] = []
    resolve_subject(
        make_conn(cursor),
        "hot_showers",
        embedder=make_embedder(),
        adjudicator=make_adjudicator(match="shower"),
        category=1,
        trace_sink=sink,
        verbose=False,
    )

    (trace,) = sink
    assert trace.term == "hot_showers"
    assert trace.category == 1
    reasons = {c.name: c.rejected_for for c in trace.candidates}
    assert reasons["shower"] is None
    assert reasons["shower_allowed"] is None
    assert reasons["far_thing"] == REJECT_FAR
    # The rule was never fetched, so it is not in the trace at all.
    assert "toilets" not in reasons
    assert [c.name for c in trace.offered] == ["shower", "shower_allowed"]
    assert trace.outcome == "ADJUDICATOR merged into 'shower'."
    assert trace.kind == "merged"
    assert trace.subject_name == "shower"


def test_a_predicate_difference_is_the_judges_call_not_a_gates():
    """`barbecue_allowed` vs `barbecue` used to be blocked by a suffix list.

    Now the pair reaches the judge, which rejects it, and the term is inserted
    under its own name. The extractor supplied the category, so the classifier
    is not consulted at all.
    """
    cursor = FakeCursor(
        nearest=[(7, "barbecue", 1, CLOSE)],
        inserted=(63, "barbecue_allowed", 1),
    )
    adjudicator = make_adjudicator(
        match=None, category=1, canonical_name="barbecue_allowed"
    )
    sink: list[ResolutionTrace] = []
    resolve_subject(
        make_conn(cursor),
        "barbecue_allowed",
        embedder=make_embedder(),
        adjudicator=adjudicator,
        category=1,
        trace_sink=sink,
        verbose=False,
    )

    (trace,) = sink
    assert trace.candidates[0].rejected_for is None
    assert [c.name for c in trace.offered] == ["barbecue"]
    adjudicator.pick_match.assert_called_once()
    adjudicator.classify.assert_not_called()
    assert trace.kind == "inserted"
    assert "ADJUDICATOR rejected all." in trace.outcome


def test_trace_records_an_alias_hit_with_no_candidates():
    cursor = FakeCursor(alias_hit=(7, "shower", 1))
    sink: list[ResolutionTrace] = []
    resolve_subject(
        make_conn(cursor),
        "Showers",
        embedder=make_embedder(),
        adjudicator=make_adjudicator(),
        trace_sink=sink,
        verbose=False,
    )

    (trace,) = sink
    assert trace.candidates == []
    assert trace.outcome == "alias hit -> 'shower'."


def test_trace_records_a_dropped_negative():
    sink: list[ResolutionTrace] = []
    resolve_subject(
        make_conn(FakeCursor()),
        "dogs_cant_be_without_muzzle",
        embedder=make_embedder(),
        adjudicator=make_adjudicator(),
        trace_sink=sink,
        verbose=False,
    )

    (trace,) = sink
    assert "DROPPED" in trace.outcome


def test_format_trace_reads_as_the_story_of_one_decision():
    trace = ResolutionTrace(
        term="air_conoditioning",
        normalized="air_conoditioning",
        category=1,
        candidates=[
            Candidate(1, "air_conditioning", -0.94, 1),
            Candidate(2, "heating", -0.70, 1, REJECT_FAR),
        ],
        outcome="ADJUDICATOR merged into 'air_conditioning'.",
    )
    line = format_trace(trace)

    assert "'air_conoditioning' from extractor (amenity)." in line
    assert "no alias match. ran NN, top 2:" in line
    assert "air_conditioning -0.940" in line
    assert "heating -0.700[far]" in line
    assert "considered 1." in line
    assert "ADJUDICATOR merged into 'air_conditioning'." in line
    assert "\n" not in line


def test_resolution_prints_by_default(capsys):
    resolve_subject(
        make_conn(FakeCursor(alias_hit=(7, "shower", 1))),
        "Showers",
        embedder=make_embedder(),
        adjudicator=make_adjudicator(),
    )
    assert "alias hit -> 'shower'." in capsys.readouterr().out


# --- injected store: experiments never touch the production table -------------


def test_every_statement_targets_the_injected_table():
    store = SubjectStore(table="test_subject_vectors", has_context=True)
    cursor = FakeCursor(nearest=[], inserted=(5, "shower", 1))

    resolve_subject(
        make_conn(cursor),
        "showers",
        embedder=make_embedder(),
        adjudicator=make_adjudicator(match=None, category=1, canonical_name="shower"),
        store=store,
        verbose=False,
    )

    assert cursor.statements
    for sql, _params in cursor.statements:
        assert "test_subject_vectors" in sql
        assert "FROM subject_vectors" not in sql
        assert "INTO subject_vectors" not in sql


def test_the_default_store_is_the_production_table_and_keeps_context():
    """Promoted to production in migration 026."""
    assert DEFAULT_STORE.table == "subject_vectors"
    assert DEFAULT_STORE.has_context is True
    cursor = FakeCursor(alias_hit=(1, "shower", 1))
    resolve_subject(
        make_conn(cursor),
        "showers",
        embedder=make_embedder(),
        adjudicator=make_adjudicator(),
        verbose=False,
    )
    assert "FROM subject_vectors" in cursor.statements[0][0]


@pytest.mark.parametrize(
    "table", ["drop table x; --", "public.subject_vectors", "Subject", "1_table", ""]
)
def test_a_store_rejects_anything_that_is_not_a_plain_identifier(table):
    """The table name is interpolated into SQL, so it is never free text."""
    with pytest.raises(ValueError):
        SubjectStore(table=table)


# --- context ------------------------------------------------------------------


def test_context_is_stored_with_a_new_subject_when_the_table_holds_it():
    store = SubjectStore(table="test_subject_vectors", has_context=True)
    cursor = FakeCursor(nearest=[], inserted=(5, "toilets", 1))

    resolve_subject(
        make_conn(cursor),
        "toilets",
        embedder=make_embedder(),
        adjudicator=make_adjudicator(match=None, category=1, canonical_name="toilets"),
        category=1,
        context="שירותים (15 תאי שירותי נשים ו- 15 תאי שירותי גברים)",
        store=store,
        verbose=False,
    )

    sql, params = cursor.sql_matching("INSERT")[0]
    assert "context" in sql
    assert params["context"].startswith("שירותים")


def test_context_is_left_out_when_the_table_has_no_such_column():
    """A store without the column still resolves; the context is just dropped."""
    cursor = FakeCursor(nearest=[], inserted=(5, "toilets", 1))

    resolve_subject(
        make_conn(cursor),
        "toilets",
        embedder=make_embedder(),
        adjudicator=make_adjudicator(match=None, category=1, canonical_name="toilets"),
        context="שירותים (15 תאים)",
        store=SubjectStore(has_context=False),
        verbose=False,
    )

    sql, params = cursor.sql_matching("INSERT")[0]
    assert "context" not in sql
    assert "context" not in params
    nn_sql = cursor.sql_matching("ORDER BY embedding")[0][0]
    assert "NULL::text AS context" in nn_sql


def test_the_judge_is_shown_the_context_of_both_sides():
    """Names alone cannot separate a communal block from a room's own bathroom."""
    store = SubjectStore(table="test_subject_vectors", has_context=True)
    cursor = FakeCursor(
        nearest=[(7, "bathroom", 1, CLOSE, "בכל חדר: שירותים, מקלחת מים חמים")],
    )
    adjudicator = make_adjudicator(match=None, category=1, canonical_name="toilets")

    resolve_subject(
        make_conn(cursor),
        "toilets",
        embedder=make_embedder(),
        adjudicator=adjudicator,
        category=1,
        context="שירותים (15 תאי שירותי נשים)",
        store=store,
        verbose=False,
    )

    kwargs = adjudicator.pick_match.call_args.kwargs
    assert kwargs["term_context"].startswith("שירותים")
    assert kwargs["candidate_contexts"]["bathroom"].startswith("בכל חדר")


def test_the_classifier_is_shown_the_context_too():
    cursor = FakeCursor(nearest=[], inserted=(5, "cooler", 1))
    adjudicator = make_adjudicator(match=None, category=1, canonical_name="cooler")

    resolve_subject(
        make_conn(cursor),
        "coolers",
        embedder=make_embedder(),
        adjudicator=adjudicator,
        context="מֵקַרים (קולרים) (4)",
        verbose=False,
    )

    assert adjudicator.classify.call_args.kwargs["context"] == "מֵקַרים (קולרים) (4)"


def test_an_antonym_candidate_is_never_offered():
    """A production run merged all four mattress-window bounds into one subject."""
    cursor = FakeCursor(
        nearest=[(7, "mattress_pickup_start_time", 2, -0.95)],
        inserted=(70, "mattress_pickup_end_time", 2),
    )
    adjudicator = make_adjudicator(
        match=None, category=2, canonical_name="mattress_pickup_end_time"
    )
    sink: list[ResolutionTrace] = []

    resolve_subject(
        make_conn(cursor),
        "mattress_pickup_end_time",
        embedder=make_embedder(),
        adjudicator=adjudicator,
        category=2,
        trace_sink=sink,
        verbose=False,
    )

    adjudicator.pick_match.assert_not_called()
    assert sink[0].candidates[0].rejected_for == REJECT_OPPOSED
    assert cursor.sql_matching("INSERT")
