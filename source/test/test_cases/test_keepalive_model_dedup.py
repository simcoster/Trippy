"""Keepalive pings one replica per model, including the embedder."""

from types import SimpleNamespace

from source.agent.keepalive import ping_models


class _Chat:
    def __init__(self, name: str, seen: list, binds: list) -> None:
        self.model_name = name
        self._seen = seen
        self._binds = binds

    def bind(self, **kwargs):
        self._binds.append((self.model_name, kwargs))
        return self

    def invoke(self, messages, config=None):
        self._seen.append(
            {"kind": "chat", "model": self.model_name, "config": config}
        )
        return SimpleNamespace(content="ok")


class _Embed:
    def __init__(self, name: str, seen: list) -> None:
        self.MODEL = name
        self._seen = seen

    def embed(self, texts, **_kwargs):
        self._seen.append({"kind": "embed", "model": self.MODEL, "texts": texts})
        return [[0.0, 1.0]]


def test_same_instruct_model_is_pinged_once():
    seen: list[dict] = []
    binds: list[tuple] = []
    ping_models(
        chats={
            "recommender": _Chat("kimi", seen, binds),
            "light": _Chat("qwen-235b", seen, binds),
            "extractor": _Chat("qwen-235b", seen, binds),
        }
    )
    assert [row["model"] for row in seen] == ["kimi", "qwen-235b"]
    assert [row["kind"] for row in seen] == ["chat", "chat"]


def test_embedder_is_pinged_with_hi():
    seen: list[dict] = []
    binds: list[tuple] = []
    ping_models(
        chats={
            "recommender": _Chat("kimi", seen, binds),
            "embed": _Embed("Qwen/Qwen3-Embedding-8B", seen),
        }
    )
    kinds = {row["kind"] for row in seen}
    assert kinds == {"chat", "embed"}
    embed = next(row for row in seen if row["kind"] == "embed")
    assert embed["texts"] == ["hi"]
    assert embed["model"] == "Qwen/Qwen3-Embedding-8B"


def test_embed_keepalive_skips_when_just_used(monkeypatch):
    from source.agent import keepalive as keepalive_mod

    monkeypatch.setattr(keepalive_mod, "_last_ok", {})
    seen: list[dict] = []
    ping_models(
        chats={"embed": _Embed("Qwen/Qwen3-Embedding-8B", seen)},
    )
    assert len(seen) == 1
    ping_models(
        chats={"embed": _Embed("Qwen/Qwen3-Embedding-8B", seen)},
    )
    assert len(seen) == 1
