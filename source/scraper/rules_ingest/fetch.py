"""Fetch a parks.org.il camping page.

Only the static response is needed: `אפשרויות לינה` and the policy-PDF lists are
the page's AJAX-loaded parts, and both are out of scope (docs/design.md).

`source.scraper.info_site.scrape` has the same helper, but that module still
uses bare `from info_site...` / `from amenity_enrichment...` imports that only
resolve under the justfile's PYTHONPATH, so it cannot be imported by module
path. The SSL relaxation is for corporate MITM chains, matching the other
scrapers.
"""

from __future__ import annotations

import ssl

import httpx

LISTING_URL = (
    "https://www.parks.org.il/"
    "%D7%94%D7%96%D7%9E%D7%A0%D7%95%D7%AA-%D7%9C%D7%97%D7%A0%D7%99%D7%95%D7%A0%D7%99-%D7%9C%D7%99%D7%9C%D7%94/"
)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/91.0.4472.124 Safari/537.36"
)
TIMEOUT_SECONDS = 45.0


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    if hasattr(ssl, "VERIFY_X509_STRICT"):
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return ctx


def fetch_page_html(url: str, *, referer: str = LISTING_URL) -> str:
    with httpx.Client(
        timeout=TIMEOUT_SECONDS,
        verify=_ssl_context(),
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT, "Referer": referer},
    ) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.text
