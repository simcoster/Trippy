"""One-shot sandbox loader does not belong on Streamlit or the planner."""

from source.price_sandbox.load import main, wait_for_sandbox


def test_wait_for_sandbox_ok(monkeypatch):
    monkeypatch.setattr(
        "source.price_sandbox.load.sandbox_reachable", lambda **_: True
    )
    assert wait_for_sandbox(wait_s=0.1) is True


def test_wait_for_sandbox_times_out(monkeypatch):
    monkeypatch.setattr(
        "source.price_sandbox.load.sandbox_reachable", lambda **_: False
    )
    assert wait_for_sandbox(wait_s=0.05) is False


def test_main_if_up_skips_when_url_missing(monkeypatch):
    monkeypatch.setattr("source.price_sandbox.load.sandbox_url", lambda: None)
    assert main(["--if-up"]) == 0


def test_main_without_url_fails(monkeypatch):
    monkeypatch.setattr("source.price_sandbox.load.sandbox_url", lambda: None)
    assert main([]) == 1
