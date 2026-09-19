"""
One TLS context for every outbound HTTPS call in the project.

Import `ssl_context()` instead of building a context per module, so scrapers,
ingest jobs and LLM clients all verify the same way.
"""

from __future__ import annotations

import os
import ssl
import sys

import certifi

# Set this in .env on a machine sitting behind a TLS-inspecting appliance
# (Norton, Zscaler, ...). Those re-sign traffic with a locally injected root
# that the OS and browsers trust but Mozilla's bundle does not, so certifi
# alone cannot verify anything there.
TRUST_OS_STORE_ENV = "TLS_TRUST_OS_STORE"

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_ORIG_CREATE_DEFAULT_CONTEXT = ssl.create_default_context


def _suppress_windows_keylog() -> None:
    """Drop Norton/Avast SSLKEYLOGFILE before OpenSSL opens it.

    Those products set SSLKEYLOGFILE to a device path
    (``\\\\.\\nllMonFltProxy\\…``). Python 3.14 + OpenSSL 3.5 then dies
    with OPENSSL_Applink when ``create_default_context()`` assigns
    ``context.keylog_filename``. A normal file path crashes the same
    uv Windows build, so the variable is removed entirely on win32.
    """
    if sys.platform == "win32":
        os.environ.pop("SSLKEYLOGFILE", None)


_suppress_windows_keylog()


def trust_os_store() -> bool:
    """Whether this machine opted into trusting its own certificate store."""
    return os.environ.get(TRUST_OS_STORE_ENV, "").strip().lower() in _TRUTHY


def _create_default_context(purpose=ssl.Purpose.SERVER_AUTH, **kwargs):
    """certifi via SSLContext; skip OPENSSLDIR and SSLKEYLOGFILE.

    Python 3.14 (OpenSSL 3.5.6) on Windows dies with OPENSSL_Applink when
    the stdlib helper loads OpenSSL's compiled-in OPENSSLDIR or assigns
    ``context.keylog_filename`` from SSLKEYLOGFILE (Norton/Avast device
    path). ``SSLContext`` + ``load_verify_locations`` does not. LangChain
    and ``urllib.request.urlopen`` (even for http://) call the stdlib
    helpers, so this module replaces both.
    """
    _suppress_windows_keylog()
    if purpose != ssl.Purpose.SERVER_AUTH:
        return _ORIG_CREATE_DEFAULT_CONTEXT(purpose, **kwargs)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(
        cafile=kwargs.get("cafile") or certifi.where(),
        capath=kwargs.get("capath"),
        cadata=kwargs.get("cadata"),
    )
    return ctx


ssl.create_default_context = _create_default_context
ssl._create_default_https_context = _create_default_context


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
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(cafile=certifi.where())
    if trust_os_store():
        ctx.load_default_certs(ssl.Purpose.SERVER_AUTH)
        if hasattr(ssl, "VERIFY_X509_STRICT"):
            ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return ctx
