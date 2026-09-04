"""Newsflash / emergency helpers for parks.org.il info pages.

Not called from scrape.py yet. Fetch + parse + notices lifecycle are here so
the notices job can be wired later.
"""

from __future__ import annotations

import hashlib
from typing import Any

import httpx
from bs4 import BeautifulSoup

from source.scraper.tls import ssl_context

from .parse import parse_whats_new, parse_wp_post_id

FLASHBACKS_PATH = "/wp-json/jmi/v1/get-flashbacks"
EMERGENCY_PATH = "/ajax-handler-wp-newsflash.php"
PARKS_ORIGIN = "https://www.parks.org.il"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/91.0.4472.124 Safari/537.36"
)

UPSERT_NOTICE_SQL = """
INSERT INTO notices (
    site_id, source, page_url, lang, notice_he, html_element, html_element_sha256
) VALUES (
    %(site_id)s, %(source)s, %(page_url)s, %(lang)s,
    %(notice_he)s, %(html_element)s, %(html_element_sha256)s
)
ON CONFLICT ON CONSTRAINT notices_site_element_key DO UPDATE
SET last_seen = now(),
    updated_at = now(),
    notice_he = EXCLUDED.notice_he,
    page_url = EXCLUDED.page_url
RETURNING id;
"""

DELETE_MISSING_NOTICES_SQL = """
DELETE FROM notices
WHERE site_id = %(site_id)s
  AND html_element_sha256 <> ALL(%(keep)s)
"""


def http_client(*, referer: str | None = None) -> httpx.Client:
    headers = {"User-Agent": USER_AGENT}
    if referer:
        headers["Referer"] = referer
    return httpx.Client(
        timeout=45.0,
        verify=ssl_context(),
        follow_redirects=True,
        headers=headers,
    )


def hash_html_element(html: str) -> str:
    return hashlib.sha256((html or "").encode("utf-8")).hexdigest()


def parse_flashbacks_json(payload: Any) -> list[dict]:
    """Normalize the get-flashbacks JSON list to {title, permalink, html}."""
    if not payload:
        return []
    posts = payload if isinstance(payload, list) else payload.get("posts") or []
    items: list[dict] = []
    for post in posts:
        if not isinstance(post, dict):
            continue
        title = (post.get("title") or "").strip()
        permalink = (post.get("permalink") or "").strip()
        if not title:
            continue
        html = (
            f'<a class="flashback" href="{permalink}">{title}</a>'
            if permalink
            else f'<span class="flashback">{title}</span>'
        )
        items.append({"title": title, "permalink": permalink or None, "html": html})
    return items


def parse_emergency_html(html: str) -> list[dict]:
    """Pull emergency banner nodes from jmi_get_last_emergency HTML."""
    if not (html or "").strip():
        return []
    soup = BeautifulSoup(html, "html.parser")
    nodes = soup.select(".flashback-emergency, .wrap-flashback-emergency")
    if not nodes:
        text = soup.get_text(" ", strip=True)
        if not text:
            return []
        return [{"title": text, "permalink": None, "html": html.strip()}]
    items: list[dict] = []
    for node in nodes:
        title = node.get_text(" ", strip=True)
        if not title:
            continue
        items.append({"title": title, "permalink": None, "html": str(node)})
    return items


def fetch_flashbacks(
    post_id: str,
    *,
    page_url: str,
    pos_type: str = "page",
) -> list[dict]:
    params = {
        "posType": pos_type,
        "postId": post_id,
        "status": "1",
        "category_or_tag": "0",
        "tax": "0",
    }
    with http_client(referer=page_url) as client:
        response = client.get(f"{PARKS_ORIGIN}{FLASHBACKS_PATH}", params=params)
        response.raise_for_status()
        return parse_flashbacks_json(response.json())


def fetch_emergency(post_id: str, *, page_url: str, pos_type: str = "page") -> list[dict]:
    params = {
        "st": "5",
        "action": "jmi_get_last_emergency",
        "isHome": "0",
        "posType": pos_type,
        "postId": post_id,
    }
    with http_client(referer=page_url) as client:
        response = client.get(f"{PARKS_ORIGIN}{EMERGENCY_PATH}", params=params)
        response.raise_for_status()
        return parse_emergency_html(response.text)


def collect_page_newsflashes(html: str, *, page_url: str) -> list[dict]:
    """Parse on-page hooks (post id / מה חדש). Does not hit AJAX."""
    items: list[dict] = []
    post_id = parse_wp_post_id(html)
    if post_id:
        items.append(
            {
                "title": f"wp_post:{post_id}",
                "permalink": page_url,
                "html": f'<body data-id="{post_id}">',
                "kind": "post_id",
            }
        )
    for bullet in parse_whats_new(html):
        items.append(
            {
                "title": bullet,
                "permalink": page_url,
                "html": f"<li>{bullet}</li>",
                "kind": "whats_new",
            }
        )
    return items


def upsert_notices(
    conn,
    *,
    site_id: int,
    items: list[dict],
    page_url: str,
    source: str = "parks.org.il",
    lang: str = "he",
) -> list[int]:
    """Insert / bump last_seen; delete hashes no longer present."""
    hashes: list[str] = []
    ids: list[int] = []
    with conn.cursor() as cur:
        for item in items:
            html = item.get("html") or ""
            digest = hash_html_element(html)
            hashes.append(digest)
            cur.execute(
                UPSERT_NOTICE_SQL,
                {
                    "site_id": site_id,
                    "source": source,
                    "page_url": page_url,
                    "lang": lang,
                    "notice_he": item.get("title"),
                    "html_element": html,
                    "html_element_sha256": digest,
                },
            )
            row = cur.fetchone()
            if row:
                ids.append(int(row[0]))
        if hashes:
            cur.execute(
                DELETE_MISSING_NOTICES_SQL,
                {"site_id": site_id, "keep": hashes},
            )
        else:
            cur.execute(
                "DELETE FROM notices WHERE site_id = %(site_id)s",
                {"site_id": site_id},
            )
    return ids


def persist_flashbacks_to_notices(
    conn,
    *,
    site_id: int,
    items: list[dict],
    page_url: str,
) -> list[int]:
    """Not wired into the info scraper yet. Returns no rows."""
    return []
