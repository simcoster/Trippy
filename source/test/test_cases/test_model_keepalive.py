"""Streamlit pings recommender, light, and extractor so Nebius stays warm."""

from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from source.agent import keepalive as keepalive_mod
from source.agent.keepalive import ping_models, start_model_keepalive


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
            {
                "model": self.model_name,
                "messages": messages,
                "config": config,
            }
        )
        return SimpleNamespace(content="ok")


def test_ping_models_sends_hi_to_recommender_light_extractor():
    seen: list[dict] = []
    binds: list[tuple] = []
    chats = {
        "recommender": _Chat("kimi", seen, binds),
        "light": _Chat("qwen-light", seen, binds),
        "extractor": _Chat("qwen-extract", seen, binds),
    }
    ping_models(chats=chats)
    roles = {row["config"]["metadata"]["role"] for row in seen}
    assert roles == {"recommender", "light", "extractor"}
    assert sorted(binds) == sorted(
        [
            ("kimi", {"max_tokens": 5}),
            ("qwen-light", {"max_tokens": 5}),
            ("qwen-extract", {"max_tokens": 5}),
        ]
    )
    for row in seen:
        assert isinstance(row["messages"][0], HumanMessage)
        assert row["messages"][0].content == "hi"
        assert row["config"]["run_name"] == f"keepalive-{row['config']['metadata']['role']}"
        assert "keepalive" in row["config"]["tags"]


def test_start_blocking_zero_interval_pings_once(monkeypatch):
    monkeypatch.setattr(keepalive_mod, "_started", False)
    seen: list[dict] = []
    binds: list[tuple] = []
    chats = {"recommender": _Chat("kimi", seen, binds)}
    start_model_keepalive(chats=chats, interval_sec=0, blocking=True)
    assert len(seen) == 1
    assert seen[0]["messages"][0].content == "hi"


def test_start_model_keepalive_starts_once(monkeypatch):
    monkeypatch.setattr(keepalive_mod, "_started", False)
    threads: list[str] = []

    class _Thread:
        def __init__(self, target, daemon, name):
            threads.append(name)
            self.target = target

        def start(self):
            return None

    monkeypatch.setattr(keepalive_mod.threading, "Thread", _Thread)
    start_model_keepalive(interval_sec=30)
    start_model_keepalive(interval_sec=30)
    assert threads == ["model-keepalive"]


def test_keepalive_interval_sec_reads_env(monkeypatch):
    monkeypatch.setenv("TRIPPY_KEEPALIVE_INTERVAL_SEC", "90")
    assert keepalive_mod.keepalive_interval_sec() == 90.0
    monkeypatch.setenv("TRIPPY_KEEPALIVE_INTERVAL_SEC", "nope")
    assert keepalive_mod.keepalive_interval_sec() == keepalive_mod.DEFAULT_INTERVAL_SEC
