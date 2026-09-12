"""Streamlit pings the recommender once per process so the first rec is not cold."""

from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from source.agent import recommender as recommender_mod
from source.agent.recommender import warmup_recommender


def test_warmup_recommender_sends_hi_once(monkeypatch):
    monkeypatch.setattr(recommender_mod, "_warmup_started", False)
    seen: list[list] = []
    binds: list[dict] = []

    class _Chat:
        def bind(self, **kwargs):
            binds.append(kwargs)
            return self

        def invoke(self, messages):
            seen.append(messages)
            return SimpleNamespace(content="ok")

    warmup_recommender(chat=_Chat(), blocking=True)
    warmup_recommender(chat=_Chat(), blocking=True)
    assert binds == [{"max_tokens": 1}]
    assert len(seen) == 1
    assert isinstance(seen[0][0], HumanMessage)
    assert seen[0][0].content == "hi"
