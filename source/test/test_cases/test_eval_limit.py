"""Eval --limit keeps the first N cases per difficulty."""

import pytest

from source.eval.run import select_eval_rows


def _rows() -> list[dict]:
    return [
        {"id": "E01", "difficulty": "easy"},
        {"id": "E02", "difficulty": "easy"},
        {"id": "E03", "difficulty": "easy"},
        {"id": "H01", "difficulty": "hard"},
        {"id": "H02", "difficulty": "hard"},
        {"id": "H03", "difficulty": "hard"},
    ]


def test_limit_two_takes_first_easy_and_hard():
    kept = select_eval_rows(_rows(), limit=2)
    assert [row["id"] for row in kept] == ["E01", "E02", "H01", "H02"]


def test_limit_zero_keeps_all():
    rows = _rows()
    assert select_eval_rows(rows, limit=0) == rows


def test_limit_after_ids():
    kept = select_eval_rows(_rows(), ids="E02,E03,H02,H03", limit=1)
    assert [row["id"] for row in kept] == ["E02", "H02"]


def test_limit_negative_exits():
    with pytest.raises(SystemExit, match="--limit must be >= 0"):
        select_eval_rows(_rows(), limit=-1)
