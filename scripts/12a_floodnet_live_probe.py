"""Stage 12a: can the live site get REAL current water depths from FloodNet?
Run on your Mac (needs internet):   python3 scripts/12a_floodnet_live_probe.py
It only reads. Report saved to reports/floodnet_live_probe.txt - paste the output to Claude.

Checks
  1. How fresh the public FloodNet events dataset on NYC Open Data is (newest event, delay in days)
  2. Where the FloodNet live dashboard (dataviz.floodnet.nyc) loads its readings from: it scans the page
     and its JavaScript for API addresses, then tries each one
  3. A few likely API addresses directly"""
import json, os, re, sys
from datetime import datetime, timezone
try:
    import requests
except ImportError:
    sys.exit("Please run: pip install requests")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "floodnet_live_probe.txt")
H = {"User-Agent": "nyc-flood-watch student project (read-only check)"}
lines = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)
def get(url, **kw):
    try: return requests.get(url, timeout=40, headers=H, **kw)
    except requests.RequestException as e:
        say("   could not reach", url[:90], "-", type(e).__name__); return None

now = datetime.now(timezone.utc)
say("FloodNet live-data check,", now.strftime("%Y-%m-%d %H:%M UTC"))

# 1. NYC Open Data events dataset (Socrata), newest rows
say("\n1. NYC Open Data FloodNet events (aq7i-eu5q)")
r = get("https://data.cityofnewyork.us/resource/aq7i-eu5q.json", params={"$limit": 3, "$order": ":updated_at DESC"})
if r is not None and r.ok and r.json():
    row = r.json()[0]; say("   columns:", list(row)[:12])
    tcol = next((k for k in row if "start" in k.lower() and "time" in k.lower()), None) or next((k for k in row if "time" in k.lower()), None)
    if tcol:
        r2 = get("https://data.cityofnewyork.us/resource/aq7i-eu5q.json", params={"$select": f"max({tcol})"})
        if r2 is not None and r2.ok:
            newest = list(r2.json()[0].values())[0]; say(f"   newest event ({tcol}): {newest}")
            try:
                lag = (now - datetime.fromisoformat(newest.replace("Z", "")).replace(tzinfo=timezone.utc)).days
                say(f"   delay: about {lag} days -> {'usable for near-live' if lag <= 1 else 'too old for live use; fine for training'}")
            except Exception: pass
m = get("https://data.cityofnewyork.us/api/views/aq7i-eu5q.json")
if m is not None and m.ok:
    j = m.json(); say("   dataset last updated:", datetime.fromtimestamp(j.get("rowsUpdatedAt", 0), timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

# 2. what the live dashboard calls
say("\n2. Live dashboard (dataviz.floodnet.nyc): looking for its data addresses")
found = set()
for page in ("https://dataviz.floodnet.nyc/", "https://www.floodnet.nyc/", "https://floodnet.nyc/"):
    r = get(page)
    if r is None: continue
    say(f"   {page}: HTTP {r.status_code}, {len(r.text)} bytes")
    urls = set(re.findall(r'https?://[A-Za-z0-9._\-/]*(?:api|graphql|influx|query|data)[A-Za-z0-9._\-/?=&]*', r.text))
    scripts = re.findall(r'<script[^>]+src="([^"]+)"', r.text)
    for s in scripts[:15]:
        su = s if s.startswith("http") else requests.compat.urljoin(page, s)
        js = get(su)
        if js is not None and js.ok:
            urls |= set(re.findall(r'https?://[A-Za-z0-9._\-/]*(?:api|graphql|influx|query|data)[A-Za-z0-9._\-/?=&]*', js.text))
    urls = {u for u in urls if not any(x in u for x in ("googleapis", "gstatic", "w3.org", "mapbox.com/styles", "github.com", "schema.org"))}
    found |= urls
for u in sorted(found)[:25]: say("   found:", u)

# 3. try the found addresses and a few likely ones
say("\n3. Trying data addresses")
cands = sorted(found)[:15] + ["https://api.dataviz.floodnet.nyc/", "https://api.floodnet.nyc/", "https://dataviz.floodnet.nyc/api/sensors"]
for u in cands:
    r = get(u)
    if r is None: continue
    ct = r.headers.get("content-type", "")
    say(f"   {u[:90]}: HTTP {r.status_code}, {ct.split(';')[0]}, {len(r.content)} bytes")
    if r.ok and "json" in ct:
        say("      sample:", r.text[:300].replace("\n", " "))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(lines) + "\n"); print("\nSaved to", OUT)
