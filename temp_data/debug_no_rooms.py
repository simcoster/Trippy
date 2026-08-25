import re
import ssl
import sys
from html import unescape
from pathlib import Path

import httpx

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ctx = ssl.create_default_context()
if hasattr(ssl, "VERIFY_X509_STRICT"):
    ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT

url = (
    "https://secure-hotels.net/INPA/BE_Results.aspx"
    "?lang=heb&hotel=9_1&in=2026-08-24&out=2026-08-25"
    "&rooms=1&ad1=2&ch1=0&inf1=0"
)
url_ok = (
    "https://secure-hotels.net/INPA/BE_Results.aspx"
    "?lang=heb&hotel=9_1&in=2026-08-25&out=2026-08-26"
    "&rooms=1&ad1=2&ch1=0&inf1=0"
)


def fetch(u: str) -> str:
    with httpx.Client(
        timeout=45,
        verify=ctx,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0"},
    ) as client:
        r = client.get(u)
        print("status", r.status_code, "len", len(r.text), "url", r.url)
        return r.text


print("=== tonight 24-25 ===")
html = fetch(url)
Path("temp_data/debug_no_rooms.html").write_text(html, encoding="utf-8")
print("roomData", len(re.findall(r"roomData=", html)))
print("room-holder", html.count("room-holder"))
print("matrixButton", html.count("matrixButton"))
print("rooms-list-title", "rooms-list-title" in html)
title = re.search(r"<title>(.*?)</title>", html, re.I | re.S)
if title:
    print("title:", re.sub(r"\s+", " ", title.group(1)).strip()[:150])
for phrase in [
    "אין זמינות",
    "לא נמצאו",
    "אין חדרים",
    "לא ניתן",
    "sold",
    "NoResults",
    "בחרו את סוג",
    "OptimaHotelNumber",
]:
    print(f"  {phrase!r}: {phrase in html}")

print("\n=== tomorrow 25-26 (known good) ===")
html2 = fetch(url_ok)
print("roomData", len(re.findall(r"roomData=", html2)))
print("room-holder", html2.count("room-holder"))
