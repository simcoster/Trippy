"""Pure HTML extractors for parks.org.il camping info pages."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

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
