"""Laptop price sandbox is up and has at least one quote loaded.

CI has no sandbox container. `pytest -m "not llm and not local"` skips this.
Run it with `pytest -m local`.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from urllib.parse import urljoin

import pytest

pytestmark = pytest.mark.local

_DEFAULT_URL = "http://127.0.0.1:8503"


def test_local_sandbox_is_up_with_a_loaded_function():
    url = (os.environ.get("PRICE_SANDBOX_URL") or _DEFAULT_URL).rstrip("/")
    try:
        response = urllib.request.urlopen(urljoin(url + "/", "health"), timeout=3)
    except urllib.error.HTTPError as exc:
        response = exc
    body = json.loads(response.read().decode("utf-8"))
    assert response.status == 200, body
    assert body["ok"] is True
    assert body["loaded"] >= 1
