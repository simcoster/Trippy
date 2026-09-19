"""TLS context avoids stdlib create_default_context (Windows OPENSSL_Applink)."""

import ssl

import certifi

from source.scraper import tls as tls_mod
from source.scraper.tls import ssl_context


def test_ssl_context_loads_certifi():
    ctx = ssl_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname is True


def test_create_default_context_is_patched():
    assert ssl.create_default_context is tls_mod._create_default_context
    assert ssl._create_default_https_context is tls_mod._create_default_context
    ctx = ssl.create_default_context(cafile=certifi.where())
    assert ctx.verify_mode == ssl.CERT_REQUIRED


def test_patched_context_ignores_sslkeylogfile(monkeypatch):
    monkeypatch.setenv("SSLKEYLOGFILE", r"\\.\nllMonFltProxy\test")
    ctx = ssl.create_default_context(cafile=certifi.where())
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    https = ssl._create_default_https_context()
    assert https.verify_mode == ssl.CERT_REQUIRED
