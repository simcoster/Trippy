"""Session ping warms the claim judge with its system prompt only."""

from types import SimpleNamespace

from source.agent import keepalive as keepalive_mod
from source.agent.claim_judge import _judge_system, warmup_claim_judge
from source.agent.keepalive import ping_models


class _Completions:
    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content="")
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice], usage=None)


def test_warmup_sends_only_the_system_prompt(monkeypatch):
    completions = _Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(
        "source.agent.claim_judge.make_nebius_openai_client",
        lambda: client,
    )
    warmup_claim_judge()
    assert completions.kwargs is not None
    assert completions.kwargs["messages"] == [
        {"role": "system", "content": _judge_system()}
    ]
    assert completions.kwargs["max_tokens"] == 5


def test_session_ping_warms_the_claim_judge(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(keepalive_mod, "_targets", lambda chats: ())
    monkeypatch.setattr(
        "source.agent.claim_judge.warmup_claim_judge",
        lambda: seen.append("warm"),
    )
    ping_models(reason="session")
    assert seen == ["warm"]


def test_interval_ping_does_not_warm_the_claim_judge(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(keepalive_mod, "_targets", lambda chats: ())
    monkeypatch.setattr(
        "source.agent.claim_judge.warmup_claim_judge",
        lambda: seen.append("warm"),
    )
    ping_models(reason="interval")
    assert seen == []
