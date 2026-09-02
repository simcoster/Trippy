"""ensure_amenities keeps its batched fast path and delegates misses."""

from unittest.mock import MagicMock, patch

from source.scraper.amenity_enrichment.db import ensure_amenities
from source.scraper.subjects.resolve import SubjectRef


class FakeCursor:
    def __init__(self, existing: dict[str, int]):
        self.existing = existing
        self.statements: list[tuple[str, object]] = []
        self._rows: list[tuple[int, str]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.statements.append((sql, params))
        names = params[0] if params else []
        self._rows = [(self.existing[n], n) for n in names if n in self.existing]

    def fetchall(self):
        return self._rows


def make_conn(existing: dict[str, int]):
    cursor = FakeCursor(existing)
    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn, cursor


@patch("source.scraper.amenity_enrichment.db.register_vector")
def test_all_names_known_resolves_in_one_query_with_no_llm(_register):
    conn, cursor = make_conn({"shower": 1, "fridge": 2})
    embedder = MagicMock()

    with patch("source.scraper.subjects.resolve.resolve_subject") as resolve:
        mapping = ensure_amenities(conn, embedder, ["shower", "fridge", "shower"])

    assert mapping == {"shower": 1, "fridge": 2}
    assert len(cursor.statements) == 1
    assert "FROM subject_vectors WHERE name = ANY(%s)" in cursor.statements[0][0]
    resolve.assert_not_called()
    embedder.embed.assert_not_called()


@patch("source.scraper.amenity_enrichment.db.register_vector")
def test_unknown_names_go_through_the_resolver(_register):
    conn, _cursor = make_conn({"shower": 1})
    embedder, adjudicator = MagicMock(), MagicMock()

    with patch(
        "source.scraper.subjects.resolve.resolve_subject",
        return_value=SubjectRef(9, "air_conditioning", 1),
    ) as resolve:
        mapping = ensure_amenities(
            conn,
            embedder,
            ["shower", "air_conoditioning"],
            adjudicator=adjudicator,
        )

    assert mapping == {"shower": 1, "air_conoditioning": 9}
    resolve.assert_called_once()
    assert resolve.call_args.args[1] == "air_conoditioning"
    assert resolve.call_args.kwargs["adjudicator"] is adjudicator


@patch("source.scraper.amenity_enrichment.db.register_vector")
def test_a_dropped_name_is_simply_absent_from_the_mapping(_register):
    """An unrewritable negative resolves to nothing; callers filter on presence."""
    conn, _cursor = make_conn({})
    with patch("source.scraper.subjects.resolve.resolve_subject", return_value=None):
        mapping = ensure_amenities(conn, MagicMock(), ["cant_be_without_muzzle"])
    assert mapping == {}


@patch("source.scraper.amenity_enrichment.db.register_vector")
def test_blank_and_duplicate_names_are_collapsed_before_lookup(_register):
    conn, cursor = make_conn({"shower": 1})
    with patch("source.scraper.subjects.resolve.resolve_subject", return_value=None):
        ensure_amenities(conn, MagicMock(), [" shower ", "shower", "", "   "])
    assert cursor.statements[0][1] == (["shower"],)


@patch("source.scraper.amenity_enrichment.db.register_vector")
def test_no_names_touches_nothing(_register):
    conn, cursor = make_conn({})
    assert ensure_amenities(conn, MagicMock(), []) == {}
    assert cursor.statements == []


@patch("source.scraper.amenity_enrichment.db.register_vector")
def test_the_resolver_cache_is_shared_across_names(_register):
    """One cache per call, so a term repeated in two lists resolves once."""
    conn, _cursor = make_conn({})
    seen: list[dict] = []

    def record(_conn, _term, **kwargs):
        seen.append(kwargs["cache"])
        return SubjectRef(1, "x", 1)

    with patch("source.scraper.subjects.resolve.resolve_subject", side_effect=record):
        ensure_amenities(conn, MagicMock(), ["a", "b"])

    assert len(seen) == 2
    assert seen[0] is seen[1]
