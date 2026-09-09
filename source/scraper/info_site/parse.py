"""Pure HTML extractors for parks.org.il camping info pages."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, unquote, urlparse

from bs4 import BeautifulSoup

_PRICE_RE = re.compile(r"([\d]+(?:[.,]\d+)?)")
_WS_RE = re.compile(r"\s+")


def normalize_label(text: str) -> str:
    """Collapse whitespace and NBSP so labels compare cleanly."""
    return _WS_RE.sub(" ", (text or "").replace("\xa0", " ")).strip()


def parse_price(text: str) -> float | None:
    """Parse '76.00 ₪' / '76,00' into a float. None if no number."""
    match = _PRICE_RE.search((text or "").replace(",", ""))
    if not match:
        return None
    return float(match.group(1))


def parse_rate_table(html: str) -> list[dict]:
    """Extract raw rows from the רגיל tab (`.tableMain[data-id=1]`)."""
    soup = BeautifulSoup(html, "html.parser")
    table_wrap = soup.find("div", class_="tableMain", attrs={"data-id": "1"})
    if table_wrap is None:
        table_wrap = soup.select_one("div.tableMain[data-id='1']")
    if table_wrap is None:
        return []

    rows: list[dict] = []
    for tr in table_wrap.select("tbody tr"):
        cells = tr.find_all("td")
        if len(cells) < 2:
            continue
        label = normalize_label(cells[0].get_text(" ", strip=True))
        price = parse_price(cells[1].get_text(" ", strip=True))
        if not label or price is None:
            continue
        note_el = tr.find(attrs={"data-content": True})
        notes = None
        if note_el is not None:
            notes = normalize_label(note_el.get("data-content") or "") or None
        rows.append({"raw_label": label, "price": price, "notes": notes})
    return rows


def parse_booking_hotel_id(html: str) -> str | None:
    """Read `hotel=` from `#ReservingHotelCamp` iframe src."""
    soup = BeautifulSoup(html, "html.parser")
    iframe = soup.select_one("#ReservingHotelCamp")
    if iframe is None:
        return None
    src = iframe.get("src") or ""
    hotel = parse_qs(urlparse(src).query).get("hotel", [None])[0]
    hotel = (hotel or "").strip()
    return hotel or None


def parse_wp_post_id(html: str) -> str | None:
    """WordPress post id from `body[data-id]` (used by flashbacks AJAX)."""
    soup = BeautifulSoup(html, "html.parser")
    if soup.body is None:
        return None
    post_id = (soup.body.get("data-id") or "").strip()
    return post_id or None


def parse_whats_new(html: str) -> list[str]:
    """Bullets under the listing-page 'מה חדש' heading."""
    soup = BeautifulSoup(html, "html.parser")
    heading = None
    for tag in soup.find_all(["h2", "h3"]):
        if "מה חדש" in tag.get_text(" ", strip=True):
            heading = tag
            break
    if heading is None:
        return []
    ul = heading.find_next("ul")
    if ul is None:
        return []
    items = [normalize_label(li.get_text(" ", strip=True)) for li in ul.find_all("li")]
    return [item for item in items if item]


# Nested parks.org.il area URLs prefix the child slug with a two-letter
# area code: /area-north/an-upper-galilee/, /area-south/as-dead-sea/,
# /area-center/ac-coastal-plain/. Strip that so we embed "region:upper-galilee".
_AREA_CHILD_PREFIX = {
    "area-north": "an-",
    "area-south": "as-",
    "area-center": "ac-",
}


def _path_segments(href: str) -> list[str]:
    path = unquote(urlparse(href or "").path or "").strip("/")
    return [part for part in path.split("/") if part]


def _child_slug(area: str, nested: str) -> str:
    prefix = _AREA_CHILD_PREFIX.get(area)
    if prefix and nested.startswith(prefix) and nested != prefix:
        return nested[len(prefix) :]
    return nested


def _area_claim(segment: str) -> str:
    rest = segment.removeprefix("area-") if segment.startswith("area-") else segment
    return f"area:{rest}"


def _region_claim(area: str, nested: str) -> str:
    return f"region:{_child_slug(area, nested)}"


def parse_breadcrumb_regions(html: str) -> list[dict]:
    """Region crumbs from `#breadcrumbs`: [{slug, label, href}, ...].

    Home, the park page, and the overnight page itself are skipped. Unique
    slugs, trail order. `slug` is `area:north` / `region:upper-galilee`;
    `label` is the Hebrew link text.
    """
    soup = BeautifulSoup(html, "html.parser")
    el = soup.select_one("#breadcrumbs")
    if el is None:
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for anchor in el.find_all("a"):
        href = anchor.get("href") or ""
        parts = _path_segments(href)
        if not parts or not parts[0].startswith("area-"):
            continue
        label = normalize_label(anchor.get_text(" ", strip=True))
        area = parts[0]
        slugs = [_area_claim(area)]
        slugs.extend(_region_claim(area, part) for part in parts[1:])
        last_new = None
        for slug in slugs:
            if not slug or slug in seen:
                continue
            seen.add(slug)
            row = {"slug": slug, "label": "", "href": href}
            out.append(row)
            last_new = row
        if last_new is not None:
            last_new["label"] = label
    return out
