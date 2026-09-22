"""The first-token thread keeps the caller's context for LangSmith."""

import contextvars

from source.agent.recommender.stream import iter_chat_chunks

_probe: contextvars.ContextVar[int] = contextvars.ContextVar(
    "recommend_trace_probe", default=0
)


class _Chunk:
    content = "ok"
    additional_kwargs: dict = {}


class _Chat:
    def stream(self, messages, **kwargs):
        assert _probe.get() == 7
        yield _Chunk()


def test_first_token_thread_sees_the_caller_context():
    _probe.set(7)
    chunks = list(iter_chat_chunks(_Chat(), [], first_token_sec=5))
    assert len(chunks) == 1
    assert chunks[0].content == "ok"
