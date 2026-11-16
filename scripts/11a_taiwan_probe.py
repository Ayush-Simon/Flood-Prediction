"""Stage 11a: check which public Taiwan data the Taipei version of the project can use.
Run on your Mac (needs internet):   python3 scripts/11a_taiwan_probe.py
It only reads; it saves a short report to reports/taiwan_probe.txt. Paste the output to Claude.

What it checks
  1. Road flood-depth sensors (Water Resources Agency, via the SensorThings service that replaced Civil IoT
     Taiwan at sta.colife.org.tw): how many sensors in Taipei / New Taipei, and how far back their records go
  2. The Water Resources Agency open-data API (flood sensor list)
  3. Rain: Open-Meteo hourly history for Taipei (same source as New York)
  4. Streets: OpenStreetMap through the Overpass API (a small test box in Taipei)"""
import json, os, sys, time
from datetime import datetime, timezone
try:
    import requests
except ImportError:
    sys.exit("Please run: pip install requests")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "taiwan_probe.txt")
BBOX = (121.40, 24.95, 121.65, 25.20)          # lon/lat box around Taipei and inner New Taipei
lines = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)
def get(url, **kw):
    try:
        r = requests.get(url, timeout=60, headers={"User-Agent": "nyc-flood-watch student project"}, **kw)
        return r
    except requests.RequestException as e:
        say("   could not reach:", url.split("?")[0], "-", type(e).__name__); return None

say("Taiwan data check,", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

# ---------------------------------------------------------------- 1. SensorThings flood sensors
say("\n1. Road flood sensors (SensorThings, sta.colife.org.tw)")
found = None
for svc in ("STA_WaterResource_v2", "STA_WaterResource", "STA_IoW", "STA_Flood"):
    base = f"https://sta.colife.org.tw/{svc}/v1.0"
    r = get(f"{base}/Things", params={"$top": 1, "$count": "true", "$filter": "substringof('淹水',name) or substringof('淹水',description)"})
    if r is None: continue
    say(f"   {svc}: HTTP {r.status_code}", (f"{r.json().get('@iot.count')} flood things" if r.ok else r.text[:120]))
    if r.ok and r.json().get("@iot.count"):
        found = base; break
if found:
    flt = (f"st_within(Locations/location, geography'POLYGON(({BBOX[0]} {BBOX[1]},{BBOX[2]} {BBOX[1]},{BBOX[2]} {BBOX[3]},"
           f"{BBOX[0]} {BBOX[3]},{BBOX[0]} {BBOX[1]}))')")
    r = get(f"{found}/Things", params={"$filter": flt, "$expand": "Locations,Datastreams", "$top": 500, "$count": "true"})
    if r is not None and r.ok:
        things = r.json().get("value", [])
        flood = [t for t in things if "淹水" in (t.get("name", "") + t.get("description", "")) or any("淹水" in d.get("name", "") for d in t.get("Datastreams", []))]
        say(f"   things in the Taipei box: {len(things)}, of which flood sensors: {len(flood)}")
        for t in flood[:3]:
            say("   example:", t.get("name"), "|", [d.get("name") for d in t.get("Datastreams", [])][:3])
        # history depth on up to 5 sensors
        for t in flood[:5]:
            ds = next((d for d in t.get("Datastreams", []) if "淹水" in d.get("name", "") or "深度" in d.get("name", "")), (t.get("Datastreams") or [None])[0])
            if not ds: continue
            link = ds.get("@iot.selfLink") or f"{found}/Datastreams({ds['@iot.id']})"
            first = get(f"{link}/Observations", params={"$orderby": "phenomenonTime asc", "$top": 1, "$count": "true"})
            last = get(f"{link}/Observations", params={"$orderby": "phenomenonTime desc", "$top": 1})
            if first is not None and first.ok and last is not None and last.ok:
                f, l = first.json(), last.json()
                say(f"   {t.get('name')}: {f.get('@iot.count')} readings from",
                    (f.get("value") or [{}])[0].get("phenomenonTime"), "to", (l.get("value") or [{}])[0].get("phenomenonTime"),
                    "| unit", ds.get("unitOfMeasurement", {}).get("symbol"))
            time.sleep(0.5)
    else:
        say("   bounding-box query failed:", None if r is None else r.status_code)
else:
    say("   no flood-sensor service found under the names tried")

# ---------------------------------------------------------------- 2. WRA open data API
say("\n2. Water Resources Agency open data")
for url in ("https://opendata.wra.gov.tw/api/v2/flood-sensor", "https://opendata.wra.gov.tw/openapi/api/OpenData/openapi"):
    r = get(url)
    if r is not None: say(f"   {url}: HTTP {r.status_code}, {len(r.content)} bytes")
r = get("https://data.gov.tw/api/v2/rest/dataset/142980")
if r is not None and r.ok:
    try:
        res = r.json()["result"]["distribution"]; say("   'IoW latest flood depth' files:", [d.get("resourceDownloadUrl") for d in res][:3])
    except Exception:
        say("   data.gov.tw answered but the format was unexpected")

# ---------------------------------------------------------------- 3. Rain
say("\n3. Rain: Open-Meteo history for Taipei")
r = get("https://archive-api.open-meteo.com/v1/archive", params={"latitude": 25.04, "longitude": 121.53, "start_date": "2024-07-24",
        "end_date": "2024-07-26", "hourly": "precipitation", "timezone": "Asia/Taipei"})
if r is not None and r.ok:
    p = r.json()["hourly"]["precipitation"]; say(f"   ok: {len(p)} hours, typhoon Gaemi 24-26 Jul 2024 total {sum(x or 0 for x in p):.0f} mm, wettest hour {max(x or 0 for x in p):.1f} mm")

# ---------------------------------------------------------------- 4. Streets
say("\n4. Streets: OpenStreetMap (Overpass)")
q = '[out:json][timeout:60];way["highway"](25.03,121.50,25.05,121.53);out count;'
r = get("https://overpass-api.de/api/interpreter", params={"data": q})
if r is not None and r.ok:
    try: say("   ok: streets in a 2 x 3 km test box:", r.json()["elements"][0]["tags"]["ways"])
    except Exception: say("   answered:", r.text[:120])

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\nSaved to", OUT)
