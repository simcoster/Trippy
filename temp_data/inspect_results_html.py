import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

har = json.loads(
    Path(r"c:\Users\Omri\Downloads\secure-hotels_search_results.net.har").read_text(
        encoding="utf-8"
    )
)
body = har["log"]["entries"][0]["response"]["content"].get("text", "") or ""
out = Path(r"c:\dev\Trippy\temp_data\search_results.html")
out.write_text(body, encoding="utf-8")
print("len", len(body))

# find price-ish and room-ish class names
classes = sorted(set(re.findall(r'class=["\']([^"\']+)["\']', body)))
for c in classes:
    low = c.lower()
    if any(k in low for k in ("room", "price", "deal", "avail", "cart", "type")):
        print("CLASS", c)

print("\n--- snippets with price ---")
for m in re.finditer(r".{0,80}(price|Price|מחיר|₪|NIS).{0,120}", body):
    s = re.sub(r"\s+", " ", m.group(0))
    print(s[:200])
