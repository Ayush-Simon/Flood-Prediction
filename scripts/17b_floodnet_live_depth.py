"""Stage 17b: current water depth at the flood sensors, from FloodNet's public dashboard data.

The FloodNet dashboard (dataviz.floodnet.nyc) runs on the FieldKit platform and reads its data from
api.fieldkit.org, project 174 (see github.com/fieldkit/cloud, portal/src/views/shared/partners/partners.ts).
These read-only calls need no login:
  GET /projects/174/stations                 every FloodNet sensor, its location and its latest reading
  GET /meta/stations?stations=<ids>          the module and sensor ids of each station's depth sensor
  GET /sensors/data?stations=..&sensors=..   readings over a time window (here: the last 70 minutes)

Writes backend/store/sensor_depth_now.json, which backend/updater.py reads:
  {"updated": ..., "source": ..., "depth_now": {"<Sensor ID>": inches}, "last_hour": {"<Sensor ID>": [12 values, oldest first]}}
Our sensors are matched to FieldKit stations by name ("BK - Richardson St/N 11th St"), else by location (within 150 m).
A sensor whose newest reading is more than 30 minutes old is left out. If anything fails, no file is written and the
site keeps saying that live depth is not connected.

Usage: python3 scripts/17b_floodnet_live_depth.py [--out backend/store/sensor_depth_now.json] [--verbose]"""
import argparse, json, math, os, re, sys, time
from datetime import datetime, timezone
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API, PROJECT = "https://api.fieldkit.org", 174
H = {"User-Agent": "nyc-flood-watch student project (read-only)", "Accept": "application/json"}
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(ROOT, "backend", "store", "sensor_depth_now.json"))
ap.add_argument("--verbose", action="store_true"); args = ap.parse_args()
say = lambda *a: print(*a, flush=True)

def get(path, **params):
    r = requests.get(API + path, params=params, headers=H, timeout=40); r.raise_for_status(); return r.json()
norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
def to_in(v, unit):
    u = (unit or "").lower()
    return v / 25.4 if u in ("mm", "millimeters", "millimetres") else v / 2.54 if u in ("cm",) else v / 0.0254 if u in ("m",) else v
def dist_m(a, b):
    return math.hypot((a[0] - b[0]) * 111320 * math.cos(math.radians(a[1])), (a[1] - b[1]) * 110540)

def main():
    import csv
    ours = list(csv.DictReader(open(os.path.join(ROOT, "data", "raw", "floodnet_sensors.csv"))))
    cell = {r["Sensor ID"] for r in csv.DictReader(open(os.path.join(ROOT, "data", "processed", "sensor_rain_cell.csv")))}
    ours = [r for r in ours if r["Sensor ID"] in cell]                      # the 27 sensors the model uses

    stations = get(f"/projects/{PROJECT}/stations").get("stations", [])
    say(f"FieldKit project {PROJECT}: {len(stations)} stations")
    def loc(s):
        p = (s.get("location") or {}).get("precise")
        return (float(p[0]), float(p[1])) if p and len(p) >= 2 else None
    def depth_sensor(s):                                                   # the station's depth sensor, with its latest reading
        for conf in ((s.get("configurations") or {}).get("all") or [])[:1]:
            for m in conf.get("modules") or []:
                for x in m.get("sensors") or []:
                    if "depth" in (x.get("key", "") + x.get("fullKey", "") + x.get("name", "")).lower():
                        return x
        return None
    by_name = {norm(s.get("name")): s for s in stations}
    match = {}
    for r in ours:
        s = by_name.get(norm(r["Sensor Name"]))
        if s is None:
            here = (float(r["Longitude"]), float(r["Latitude"]))
            near = [(dist_m(here, loc(x)), x) for x in stations if loc(x)]
            near = [n for n in near if n[0] <= 150]
            if near: s = min(near, key=lambda n: n[0])[1]
        if s is not None: match[r["Sensor ID"]] = s
    say(f"matched {len(match)} of our {len(ours)} sensors to FieldKit stations")

    now_ms = int(time.time() * 1000); depth_now, last_hour, units = {}, {}, {}
    for sid, s in match.items():
        x = depth_sensor(s)
        rd = (x or {}).get("reading") or {}
        if x is None or rd.get("last") is None or not rd.get("time"): continue
        if now_ms - int(rd["time"]) > 30 * 60 * 1000: continue           # offline or stale: leave it out
        units[sid] = x.get("unitOfMeasure"); depth_now[sid] = round(max(0.0, to_in(float(rd["last"]), units[sid])), 2)

    # last hour, in 5-minute steps, for the sensors that are reporting
    ids = {sid: match[sid]["id"] for sid in depth_now}
    try:
        meta = get("/meta/stations", stations=",".join(str(v) for v in ids.values())).get("stations", {})
        end = now_ms - now_ms % 300000; start = end - 12 * 300000
        for sid, stid in ids.items():
            rows = [m for m in meta.get(str(stid), []) if "depth" in str(m.get("sensorKey", "")).lower()]
            if not rows or rows[0].get("moduleId") is None: continue
            d = get("/sensors/data", stations=stid, sensors=f'{rows[0]["moduleId"]},{rows[0]["sensorId"]}', start=start, end=end, resolution=60, complete="true")
            pts = d.get("data", d) if isinstance(d, dict) else d
            pts = [p for p in (pts if isinstance(pts, list) else []) if isinstance(p, dict) and p.get("time") is not None and p.get("value") is not None]
            if not pts: continue
            bins = [[] for _ in range(12)]
            for p in pts:
                k = int((int(p["time"]) - start) // 300000)
                if 0 <= k < 12: bins[k].append(to_in(float(p["value"]), units[sid]))
            series, last = [], None
            for b in bins:                                                 # fill a gap with the reading before it
                if b: last = max(0.0, sum(b) / len(b))
                series.append(round(last, 2) if last is not None else None)
            first = next((v for v in series if v is not None), None)
            if first is None: continue
            last_hour[sid] = [first if v is None else v for v in series]
    except (requests.RequestException, ValueError, KeyError, TypeError) as e:
        say("last-hour series unavailable (current depth only):", type(e).__name__, str(e)[:120])

    if not depth_now: sys.exit("no current FloodNet readings matched; nothing written")
    out = {"updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "source": f"FloodNet via api.fieldkit.org (project {PROJECT})",
           "depth_now": depth_now, "last_hour": last_hour}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f: json.dump(out, f, indent=1)
    wet = {k: v for k, v in depth_now.items() if v >= 0.05}
    say(f"wrote {args.out}: {len(depth_now)} sensors reporting, {len(last_hour)} with the last hour, {len(wet)} with water now")
    if args.verbose: say(json.dumps(out, indent=1)[:3000])

try:
    main()
except requests.RequestException as e:
    sys.exit(f"FloodNet / FieldKit not reachable: {type(e).__name__}: {str(e)[:160]}")
