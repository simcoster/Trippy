"""Planner query embeddings are one request for every distinct phrase."""

from __future__ import annotations

from source.agent.search import embed


def test_query_vec_literals_uses_query_vec_literal(monkeypatch):
    monkeypatch.setattr(
        "source.agent.search.embed._query_vec_literal", lambda query: f"[{query}]"
    )
    out = embed._query_vec_literals(["quiet", "shade", "quiet"])
    assert out == {"quiet": "[quiet]", "shade": "[shade]"}


def test_distinct_phrases_are_one_embed_call(monkeypatch):
    seen: list[list[str]] = []

    def fake_embed(texts, **_kwargs):
        seen.append(list(texts))
        return [[float(i)] for i in range(len(texts))]

    monkeypatch.setattr(embed._claims_embedder, "embed", fake_embed)
    queries = ["quiet", "shade", "quiet", "fridge"]
    out = embed._query_vec_literals(queries)
    assert seen == [["quiet", "shade", "fridge"]]
    assert list(out) == ["quiet", "shade", "fridge"]
    assert out["quiet"] == "[0.00000000]"
    assert out["fridge"] == "[2.00000000]"
