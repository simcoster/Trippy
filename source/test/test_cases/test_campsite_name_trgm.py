"""pg_trgm named-site lookup against Hebrew name and english_name."""

from __future__ import annotations

from source.agent.search import match_campsites_by_name
from source.test.test_cases.experiments_schema import SITE_ID

HORSHAT = SITE_ID
AKHZIV = SITE_ID + 1


def _seed(cur) -> None:
    cur.execute(
        "INSERT INTO campsites (id, name, url, english_name) VALUES "
        "(%s, %s, %s, %s), (%s, %s, %s, %s)",
        (
            HORSHAT,
            "חניון לילה גן לאומי חורשת טל",
            "https://example.invalid/horshat",
            "Horshat Tal",
            AKHZIV,
            "חניון לילה גן לאומי אכזיב – חניון דרומי",
            "https://example.invalid/akhziv-south",
            "Achziv South",
        ),
    )


def test_english_typo_matches_horshat(experiments_conn):
    with experiments_conn.cursor() as cur:
        _seed(cur)
        hits = match_campsites_by_name(cur, "Horashat Tal")
    assert [h["id"] for h in hits] == [HORSHAT]


def test_hebrew_fragment_matches_long_title(experiments_conn):
    with experiments_conn.cursor() as cur:
        _seed(cur)
        hits = match_campsites_by_name(cur, "אכזיב")
    assert AKHZIV in {h["id"] for h in hits}


def test_transliteration_drift_matches_english_name(experiments_conn):
    with experiments_conn.cursor() as cur:
        _seed(cur)
        hits = match_campsites_by_name(cur, "Akhziv")
    assert AKHZIV in {h["id"] for h in hits}


def test_unrelated_name_does_not_match(experiments_conn):
    with experiments_conn.cursor() as cur:
        _seed(cur)
        hits = match_campsites_by_name(cur, "Yehudiya")
    assert hits == []
