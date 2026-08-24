import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRAPER_DIR = Path(__file__).resolve().parent
html = (SCRAPER_DIR / "INPA.html").read_text(encoding="utf-8")

# ddlHotels options
opts = re.findall(
    r'<option[^>]*value=(["\'])(.*?)\1[^>]*>(.*?)</option>',
    html,
    flags=re.I | re.S,
)
print("ddlHotels options:")
for _q, v, t in opts:
    t = re.sub(r"\s+", " ", t).strip()
    print(f"  raw={v!r} name={t!r}")

# settings JSON
m = re.search(r'<input[^>]*id=["\']settings["\'][^>]*value=(["\'])(.*?)\1', html, re.I | re.S)
if not m:
    # value before id
    m = re.search(r'<input[^>]*value=(["\'])(.*?)\1[^>]*id=["\']settings["\']', html, re.I | re.S)
print("settings match", bool(m))
if m:
    raw = m.group(2)
    # HTML entities
    raw = raw.replace("&quot;", '"').replace("&#39;", "'")
    data = json.loads(raw)
    (SCRAPER_DIR / "settings.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    resorts = []
    for region in data.get("Regions", []):
        for r in region.get("Resorts", []):
            optima = r.get("OptimaResortID")
            wing = r.get("Wing", "1")
            hotel_id = f"{optima}_{wing}"
            resorts.append(
                {
                    "name": r.get("ResortName", "").strip(),
                    "hotel_id": hotel_id,
                    "optima_resort_id": optima,
                    "wing": wing,
                    "resort_id": r.get("ResortID"),
                    "region_id": r.get("RegionID"),
                }
            )
    print("resorts from settings:", len(resorts))
    for r in resorts:
        print(r)
