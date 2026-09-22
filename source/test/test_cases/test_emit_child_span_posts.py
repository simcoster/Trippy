"""A keepalive child span is posted, not only built in memory."""

from source.agent.tracing import emit_child_span


def test_emit_child_span_posts_the_run(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    posted: dict = {}

    class _Child:
        def end(self, **kwargs):
            posted["outputs"] = kwargs.get("outputs")

        def post(self):
            posted["posted"] = True

    class _Parent:
        def create_child(self, **kwargs):
            posted["name"] = kwargs["name"]
            return _Child()

    monkeypatch.setattr(
        "langsmith.get_current_run_tree", lambda: _Parent()
    )
    emit_child_span(
        name="keepalive-claim_judge",
        inputs={"role": "claim_judge"},
        outputs={"reply": '{"relevant": []'},
    )
    assert posted["name"] == "keepalive-claim_judge"
    assert posted["posted"] is True
    assert posted["outputs"]["reply"] == '{"relevant": []'
