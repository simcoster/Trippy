import json
import re
import sys
from html import unescape
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

body = Path(r"c:\dev\Trippy\temp_data\search_results.html").read_text(encoding="utf-8")

# roomData="{...}"
raws = re.findall(r'roomData="(\{.*?\})"', body)
print("roomData attrs", len(raws))
seen = set()
for raw in raws:
    data = json.loads(unescape(raw).replace("&quot;", '"'))
    key = (data.get("RoomType"), data.get("PcName"), data.get("Price"), data.get("RoomCode"))
    if key in seen:
        continue
    seen.add(key)
    print(json.dumps({
        "RoomType": data.get("RoomType"),
        "PcName": data.get("PcName"),
        "Price": data.get("Price"),
        "Currency": data.get("Currency"),
        "RoomCode": data.get("RoomCode"),
        "MealPlan": data.get("MealPlan"),
    }, ensure_ascii=False))

# also roomname elements
from bs4 import BeautifulSoup
soup = BeautifulSoup(body, "html.parser")
print("\nroomname count", len(soup.select(".roomname")))
for el in soup.select(".room-holder"):
    name = el.select_one(".roomname")
    price = el.select_one(".PriceD")
    print("holder:", (name.get_text(strip=True) if name else None), (price.get("price") if price else None))
