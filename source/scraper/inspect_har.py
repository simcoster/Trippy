import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRAPER_DIR = Path(__file__).resolve().parent
har = json.loads(Path(r"c:\Users\Omri\Downloads\secure-hotels.net.har").read_text(encoding="utf-8"))
entries = har["log"]["entries"]

e = entries[0]
body = e["response"]["content"].get("text", "") or ""
(SCRAPER_DIR / "be_results.html").write_text(body, encoding="utf-8")
print("status", e["response"]["status"], "len", len(body))

# Look for select / dropdown / hotel list structures
patterns = [
    r"ddl\w+",
    r"HotelId",
    r"hotelId",
    r"HotelsList",
    r"hotelList",
    r"cmbHotel",
    r"selectedHotel",
    r"data-hotel",
    r"hotel=\d",
    r"\d+_\d+",
]
for pat in patterns:
    hits = re.findall(pat, body)
    print(f"{pat}: {len(hits)} unique={sorted(set(hits))[:20]}")

# Any input/hidden with hotel
for m in re.finditer(r"<input[^>]{0,300}>", body, re.I):
    tag = m.group(0)
    if "hotel" in tag.lower() or re.search(r"\d+_\d+", tag):
        print("INPUT", tag[:250])

# Script blocks mentioning hotel ids
for m in re.finditer(r"<script[^>]*>(.*?)</script>", body, re.I | re.S):
    script = m.group(1)
    if "hotel" in script.lower() or re.search(r"\d+_\d+", script):
        print("--- script snippet ---")
        print(script[:1500])
        print("--- end ---")

# BE_Main.js
for ent in entries:
    url = ent["request"]["url"]
    if "BE_Main.js" not in url:
        continue
    text = ent["response"]["content"].get("text", "") or ""
    Path(SCRAPER_DIR / "be_main.js").write_text(text, encoding="utf-8")
    print("BE_Main.js saved", len(text))
    for pat in ["Hotels", "HotelList", "ddlHotel", "GetHotels", "hotelId", "HotelID", "hotelsArr", "HotelsArr"]:
        print(" ", pat, text.count(pat))
    # find functions related to hotel dropdown
    for m in re.finditer(r".{0,30}(Hotel|hotel).{0,80}", text):
        line = m.group(0).replace("\n", " ")
        if any(k in line for k in ["List", "ddl", "option", "select", "Id", "ID", "Arr"]):
            print(" ", line[:140])
