"""
One TLS context for every outbound HTTPS call in the project.

Import `ssl_context()` instead of building a context per module, so scrapers,
ingest jobs and LLM clients all verify the same way.
"""

from __future__ import annotations

import os
import ssl

import certifi

# Set this in .env on a machine sitting behind a TLS-inspecting appliance
# (Norton, Zscaler, ...). Those re-sign traffic with a locally injected root
# that the OS and browsers trust but Mozilla's bundle does not, so certifi
# alone cannot verify anything there.
TRUST_OS_STORE_ENV = "TLS_TRUST_OS_STORE"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def trust_os_store() -> bool:
    """Whether this machine opted into trusting its own certificate store."""
    return os.environ.get(TRUST_OS_STORE_ENV, "").strip().lower() in _TRUTHY


def ssl_context() -> ssl.SSLContext:
    """
    TLS context verifying against certifi's CA bundle.

    certifi is passed explicitly rather than relying on OpenSSL's default
    paths: a python.org framework build leaves
    `Versions/<x>/etc/openssl/cert.pem` absent until
    "Install Certificates.command" is run, and a bare
    `create_default_context()` there trusts nothing, failing every request
    with CERTIFICATE_VERIFY_FAILED "unable to get local issuer certificate".

    With `TLS_TRUST_OS_STORE` set, the machine's own store is loaded on top of
    certifi and VERIFY_X509_STRICT is cleared. Python enables that flag by
    default from 3.13; its extra RFC 5280 checks reject the chains an
    inspecting appliance rewrites, which browsers still accept.

    The environment is read on every call, so `load_dotenv()` only has to run
    before the first request -- not before this module is imported.
    """
    ctx = ssl.create_default_context(cafile=certifi.where())
    if trust_os_store():
        ctx.load_default_certs(ssl.Purpose.SERVER_AUTH)
        if hasattr(ssl, "VERIFY_X509_STRICT"):
            ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return ctx
