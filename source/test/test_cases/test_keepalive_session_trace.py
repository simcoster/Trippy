"""Session keepalive workers keep the caller's trace context."""

import contextvars

from source.agent import keepalive as keepalive_mod
from source.agent.keepalive import ping_models

_probe: contextvars.ContextVar[int] = contextvars.ContextVar(
    "keepalive_session_probe", default=0
)


def test_session_ping_runs_on_the_caller_context(monkeypatch):
    seen: list[int] = []

    def _round(targets):
        seen.append(_probe.get())

    monkeypatch.setattr(keepalive_mod, "_ping_round", _round)
    monkeypatch.setattr(keepalive_mod, "_targets", lambda chats: ())
    monkeypatch.setattr(
        "source.agent.claim_judge.warmup_claim_judge",
        lambda: None,
    )
    _probe.set(3)
    ping_models(reason="session")
    assert seen == [3]
