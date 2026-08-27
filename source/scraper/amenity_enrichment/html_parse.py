"""Parse INPA results HTML for room tooltips and gallery images."""

from __future__ import annotations

from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup

RESULTS_BASE_URL = "https://secure-hotels.net/INPA/"
MAX_IMAGE_URLS = 3


def _absolute_image_url(href: str) -> str | None:
    href = (href or "").strip()
    if not href or href.startswith("#") or href.lower().startswith("javascript:"):
        return None
    return urljoin(RESULTS_BASE_URL, href)


def _image_urls_from_holder(cat) -> list[str]:
    """Up to MAX_IMAGE_URLS full-size gallery links from `.imageholder`."""
    urls: list[str] = []
    seen: set[str] = set()
    holder = cat.select_one(".imageholder")
    if not holder:
        return urls
    for anchor in holder.select("a[href]"):
        abs_url = _absolute_image_url(anchor.get("href") or "")
        if not abs_url or abs_url in seen:
            continue
        seen.add(abs_url)
        urls.append(abs_url)
        if len(urls) >= MAX_IMAGE_URLS:
            break
    return urls


def parse_room_categories(html: str, normalize_name) -> dict[str, dict[str, Any]]:
    """
    Map normalized room type → {description, image_urls}.

    Description: `#toolTip_n_m > div.tt-desc > span` (also `.tt-desc .desc-span`).
    Images: up to 3 absolute URLs from `.imageholder a[href]`.
    """
    soup = BeautifulSoup(html, "html.parser")
    out: dict[str, dict[str, Any]] = {}
    for cat in soup.select("div.roomcategory"):
        name_el = cat.select_one(".roomname") or cat.select_one(".tt-roomname")
        if not name_el:
            continue
        name = normalize_name(name_el.get_text(strip=True))
        if not name or name in out:
            continue
        desc_el = cat.select_one(".tt-desc span") or cat.select_one(".tt-desc")
        description = desc_el.get_text(" ", strip=True) if desc_el else ""
        out[name] = {
            "description": description,
            "image_urls": _image_urls_from_holder(cat),
        }
    return out


def parse_room_tooltips(html: str, normalize_name) -> dict[str, str]:
    """Map normalized room type → tooltip description text."""
    return {
        name: meta["description"]
        for name, meta in parse_room_categories(html, normalize_name).items()
        if meta.get("description")
    }
