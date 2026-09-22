"""An empty price sandbox is not healthy."""

from __future__ import annotations

import json
import urllib.error

from source.price_sandbox.client import require_healthy_sandbox, sandbox_reachable
from source.price_sandbox.load import main
from source.price_sandbox.server import health_status, load_functions

_SOURCE = (
    "def quote(lodging, adults_num, child_num=0, child_ages=(), "
    'guest_type="רגיל", is_weekend_or_holiday=False, '
    "planned_entry_time=None, planned_exit_time=None):\n"
    "    return 1.0, 'ok'\n"
)


def test_health_is_not_ok_when_nothing_is_loaded():
    load_functions([])
    status, body = health_status()
    assert status == 503
    assert body == {"ok": False, "loaded": 0, "error": "no functions loaded"}


def test_health_is_ok_after_a_function_loads():
    loaded = load_functions(
        [{"site_id": 1, "source": _SOURCE, "sha256": "abc"}]
    )
    assert loaded["ok"] is True
    status, body = health_status()
    assert status == 200
    assert body == {"ok": True, "loaded": 1}
    load_functions([])


def test_reachable_treats_503_as_down_for_quotes(monkeypatch):
    def urlopen(url, timeout=0):
        raise urllib.error.HTTPError(url, 503, "unavailable", hdrs=None, fp=None)

    monkeypatch.setattr("source.price_sandbox.client.urllib.request.urlopen", urlopen)
    monkeypatch.setenv("PRICE_SANDBOX_URL", "http://127.0.0.1:8503")
    assert sandbox_reachable() is False
    assert sandbox_reachable(require_loaded=False) is True


def test_reachable_requires_a_loaded_count(monkeypatch):
    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"ok": True, "loaded": 2}).encode()

    monkeypatch.setattr(
        "source.price_sandbox.client.urllib.request.urlopen",
        lambda url, timeout=0: _Response(),
    )
    monkeypatch.setenv("PRICE_SANDBOX_URL", "http://127.0.0.1:8503")
    assert sandbox_reachable() is True


def test_require_healthy_sandbox_exits_when_down(monkeypatch):
    monkeypatch.delenv("TRIPPY_SANDBOX_CHECKED", raising=False)
    monkeypatch.setattr(
        "source.price_sandbox.client.sandbox_url", lambda: "http://127.0.0.1:8503"
    )
    monkeypatch.setattr("source.price_sandbox.client.sandbox_reachable", lambda **_: False)
    try:
        require_healthy_sandbox()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("expected SystemExit")


def test_require_healthy_sandbox_latches_when_up(monkeypatch):
    monkeypatch.delenv("TRIPPY_SANDBOX_CHECKED", raising=False)
    monkeypatch.setattr(
        "source.price_sandbox.client.sandbox_url", lambda: "http://127.0.0.1:8503"
    )
    monkeypatch.setattr("source.price_sandbox.client.sandbox_reachable", lambda **_: True)
    require_healthy_sandbox()
    monkeypatch.setattr("source.price_sandbox.client.sandbox_reachable", lambda **_: False)
    require_healthy_sandbox()


def test_loader_fails_when_it_stores_nothing(monkeypatch):
    monkeypatch.setattr(
        "source.price_sandbox.load.sandbox_url", lambda: "http://127.0.0.1:8503"
    )
    monkeypatch.setattr(
        "source.price_sandbox.load.wait_for_sandbox", lambda **_: True
    )
    monkeypatch.setattr(
        "source.price_sandbox.load.load_sandbox",
        lambda **_: {"ok": True, "loaded": [], "rejected": [], "n_db": 0},
    )
    assert main([]) == 1
