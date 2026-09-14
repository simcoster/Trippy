"""LangSmith run config is env-gated and groups turns by thread."""

import os

from source.agent.tracing import (
    agent_run_config,
    configure_agent_tracing,
    project_name,
    run_name,
    tracing_configured,
)


def test_run_name_collapses_whitespace_and_truncates():
    assert run_name("") == "trippy-turn"
    assert run_name("  שלום   עולם  ") == "שלום עולם"
    long = "x" * 90
    named = run_name(long, max_len=80)
    assert named.endswith("…")
    assert len(named) == 80


def test_agent_run_config_names_thread_and_channel():
    config = agent_run_config(
        thread_id="sess-1",
        channel="streamlit",
        user_text="אוהל בשישי",
        extra_metadata={"public_ui": True, "stop_after": "recommender"},
    )
    assert config["run_name"] == "אוהל בשישי"
    assert config["tags"] == ["trippy", "streamlit"]
    assert config["configurable"] == {"thread_id": "sess-1"}
    assert config["metadata"] == {
        "channel": "streamlit",
        "thread_id": "sess-1",
        "public_ui": True,
        "stop_after": "recommender",
    }


def test_configure_agent_tracing_stays_off_without_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    assert configure_agent_tracing() is False
    assert tracing_configured() is False
    assert os.environ.get("LANGSMITH_TRACING") != "true"


def test_configure_agent_tracing_sets_project_from_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    assert configure_agent_tracing() is True
    assert tracing_configured() is True
    assert project_name() == "trippy"
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
