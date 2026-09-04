import re
import sys
from pathlib import Path

import httpx

from source.scraper.tls import ssl_context

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRAPER_DIR = Path(__file__).resolve().parent

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

urls = [
    "https://secure-hotels.net/INPA/NoResults.aspx",
    "https://secure-hotels.net/INPA/NoResults.aspx?lang=heb",
    "https://secure-hotels.net/INPA/",
]

with httpx.Client(timeout=30, verify=ssl_context(), follow_redirects=True) as client:
    for url in urls:
        r = client.get(url, headers=headers)
        name = url.rstrip("/").split("/")[-1] or "index"
        name = name.split("?")[0] or "index"
        out = SCRAPER_DIR / f"{name}.html"
        out.write_text(r.text, encoding="utf-8")
        print("=" * 60)
        print(url, "->", r.url, "status", r.status_code, "len", len(r.text))

        opts = re.findall(
            r'<option[^>]*value=(["\'])(.*?)\1[^>]*>(.*?)</option>',
            r.text,
            flags=re.I | re.S,
        )
        print("options", len(opts))
        for _q, v, t in opts[:60]:
            t = re.sub(r"\s+", " ", t).strip()
            print(f"  {v!r} -> {t[:120]}")

        # selects
        for m in re.finditer(r"<select[^>]{0,300}>", r.text, re.I):
            print("SELECT", m.group(0)[:250])

        # hotel-ish ids
        ids = sorted(set(re.findall(r"\b\d+_\d+\b", r.text)))
        print("id-like", ids[:40], "count", len(ids))

        # interesting hidden/inputs
        for m in re.finditer(r"<input[^>]{0,400}>", r.text, re.I):
            tag = m.group(0)
            if "hotel" in tag.lower() or re.search(r"\d+_\d+", tag):
                print("INPUT", tag[:300])
