"""Streamlit pings recommender, light, and extractor so Nebius stays warm."""

from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from source.agent import keepalive as keepalive_mod
from source.agent.keepalive import ping_models, ping_new_session, start_model_keepalive

_ROOT = Path(__file__).resolve().parents[3]


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


class _Boom(_Chat):
    def invoke(self, messages, config=None):
        raise RuntimeError("nebius down")


class _StopLoop(Exception):
    """End a blocking keepalive loop after N sleeps."""


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


def test_failed_role_does_not_skip_others():
    seen: list[dict] = []
    binds: list[tuple] = []
    ping_models(
        chats={
            "recommender": _Boom("kimi", seen, binds),
            "light": _Chat("qwen-light", seen, binds),
            "extractor": _Chat("qwen-extract", seen, binds),
        }
    )
    roles = {row["config"]["metadata"]["role"] for row in seen}
    assert roles == {"light", "extractor"}


def test_start_blocking_zero_interval_pings_once(monkeypatch):
    monkeypatch.setattr(keepalive_mod, "_started", False)
    seen: list[dict] = []
    binds: list[tuple] = []
    chats = {"recommender": _Chat("kimi", seen, binds)}
    start_model_keepalive(chats=chats, interval_sec=0, blocking=True)
    assert len(seen) == 1
    assert seen[0]["messages"][0].content == "hi"


def test_start_keepalive_repeats_after_interval(monkeypatch):
    monkeypatch.setattr(keepalive_mod, "_started", False)
    seen: list[dict] = []
    binds: list[tuple] = []
    chats = {"recommender": _Chat("kimi", seen, binds)}
    sleeps: list[float] = []

    def _sleep(sec):
        sleeps.append(sec)
        if len(sleeps) >= 2:
            raise _StopLoop

    monkeypatch.setattr(keepalive_mod.time, "sleep", _sleep)
    try:
        start_model_keepalive(chats=chats, interval_sec=240, blocking=True)
    except _StopLoop:
        pass
    assert sleeps == [240, 240]
    assert len(seen) == 1


def test_process_loop_sleeps_before_first_ping(monkeypatch):
    monkeypatch.setattr(keepalive_mod, "_started", False)
    seen: list[dict] = []
    binds: list[tuple] = []
    chats = {"recommender": _Chat("kimi", seen, binds)}

    def _sleep(sec):
        raise _StopLoop

    monkeypatch.setattr(keepalive_mod.time, "sleep", _sleep)
    try:
        start_model_keepalive(chats=chats, interval_sec=240, blocking=True)
    except _StopLoop:
        pass
    assert seen == []


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


def test_default_keepalive_interval_is_ten_minutes(monkeypatch):
    monkeypatch.delenv("TRIPPY_KEEPALIVE_INTERVAL_SEC", raising=False)
    assert keepalive_mod.DEFAULT_INTERVAL_SEC == 600.0
    assert keepalive_mod.keepalive_interval_sec() == 600.0


def test_keepalive_interval_sec_reads_env(monkeypatch):
    monkeypatch.setenv("TRIPPY_KEEPALIVE_INTERVAL_SEC", "90")
    assert keepalive_mod.keepalive_interval_sec() == 90.0
    monkeypatch.setenv("TRIPPY_KEEPALIVE_INTERVAL_SEC", "nope")
    assert keepalive_mod.keepalive_interval_sec() == keepalive_mod.DEFAULT_INTERVAL_SEC


def test_ping_new_session_once_per_session():
    seen: list[dict] = []
    binds: list[tuple] = []
    chats = {"recommender": _Chat("kimi", seen, binds)}
    session: dict = {}
    assert ping_new_session(session, chats=chats, blocking=True) is True
    assert ping_new_session(session, chats=chats, blocking=True) is False
    assert len(seen) == 1
    assert ping_new_session({}, chats=chats, blocking=True) is True
    assert len(seen) == 2


def test_streamlit_chat_starts_keepalive():
    text = (_ROOT / "scripts" / "streamlit_chat.py").read_text(encoding="utf-8")
    assert "from source.agent.keepalive import ping_new_session, start_model_keepalive" in text
    assert "start_model_keepalive()" in text
    assert "ping_new_session(st.session_state)" in text
