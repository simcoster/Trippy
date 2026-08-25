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
entries = har["log"]["entries"]
print("entries", len(entries))
for i, e in enumerate(entries):
    req = e["request"]
    url = req["url"]
    method = req["method"]
    mime = (e["response"].get("content") or {}).get("mimeType", "")
    size = (e["response"].get("content") or {}).get("size", 0)
    interesting = any(
        k in url.lower()
        for k in (
            "result",
            "ajax",
            "api",
            "json",
            "room",
            "avail",
            "price",
            "be_",
            "hotel",
            "ws",
            "ashx",
            "asmx",
        )
    )
    if interesting or "json" in mime or method == "POST":
        print(f"{i}: {method} {mime} {size} {url[:180]}")
