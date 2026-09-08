"""`TRIPPY_SCHEMA` points `db.connect` at the experiments schema."""

import pytest

from db.connect import EXPERIMENTS_OPTIONS, SCHEMA_ENV, connect_options


def test_unset_means_production(monkeypatch):
    monkeypatch.delenv(SCHEMA_ENV, raising=False)
    assert connect_options() is None


def test_experiments_sets_search_path(monkeypatch):
    monkeypatch.setenv(SCHEMA_ENV, "experiments")
    assert connect_options() == EXPERIMENTS_OPTIONS


def test_unknown_schema_is_rejected(monkeypatch):
    monkeypatch.setenv(SCHEMA_ENV, "public")
    with pytest.raises(RuntimeError, match="not supported"):
        connect_options()
