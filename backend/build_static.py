"""Build the whole website as plain files, for static hosting (Vercel, GitHub Pages, any web server).

  python3 backend/build_static.py                     the data store as it is now
  python3 backend/build_static.py --update            run one live update first (needs internet)
  python3 backend/build_static.py --update --previous https://your-site.vercel.app
        also keep the 30-day history from the site that is online now, and reuse its live
        forecast if this update fails (so a failed run never leaves the site without one)
  python3 backend/build_static.py --update --live-url https://raw.githubusercontent.com/OWNER/REPO/live-data/live.json
        Netlify: the page reads its live forecast from that file (refreshed by GitHub every 30 minutes,
        see backend/build_live_feed.py), so the site only needs redeploying when the code changes

It writes site/:
  site/index.html, lab.html, css/, js/, fonts/   the frontend, copied as it is
  site/api/...                                   one file per answer the Python server gives
  site/vercel.json, site/_headers                the same security headers as backend/server.py (Vercel, Netlify)

The GitHub Action in .github/workflows/deploy.yml runs this every 30 minutes and deploys site/ to Vercel.
"""
import argparse, gzip, json, os, re, shutil, sys, urllib.request
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import store                                                  # noqa: E402
from server import SECURITY_HEADERS, STATIC_TYPES, FRONTEND   # noqa: E402  (same headers and file types as the server)

OUT = os.path.join(ROOT, "site")
LIVE_URL = re.compile(r"^https://raw\.githubusercontent\.com/[\w.-]+/[\w.-]+/live-data/live\.json$")
SAFE = re.compile(r"^(index|roads|live|weather|lab_samples|storm:\d{4}-\d{2}-\d{2}|model:[a-z_]{1,32})$")


def fetch(url):
    """JSON from the site that is online now, or None"""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "flood-watch-build"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"  could not read {url}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="run one live update first")
    ap.add_argument("--previous", help="address of the site that is online now")
    ap.add_argument("--every", type=int, default=30, help="minutes between scheduled updates (shown on the site)")
    ap.add_argument("--live-url", help="raw.githubusercontent.com address of the live-data branch's live.json")
    a = ap.parse_args()
    if a.live_url and not LIVE_URL.match(a.live_url):
        sys.exit("--live-url must look like https://raw.githubusercontent.com/OWNER/REPO/live-data/live.json")

    if not os.path.exists(store.DB):
        print("Building the data store from backend/data ...")
        import build_store
        sys.argv = [sys.argv[0]]
        build_store.main()

    prev = a.previous.rstrip("/") if a.previous else None
    if prev and not prev.startswith("https://"):
        sys.exit("--previous must be an https:// address")

    updated = False
    if a.update:
        try:
            import updater
            updater.build()
            updated = True
            print("Live update done.")
        except Exception as e:
            print("Live update failed:", e)

    con = store.connect_rw()
    try:
        if not updated and prev:                              # keep the site's last forecast rather than none
            old = fetch(prev + "/api/live")
            if isinstance(old, dict) and old.get("updated"):
                store.put(con, "live", old)
                con.commit()
                print("Kept the live forecast from", old.get("updated"))
    finally:
        con.close()

    # ---- copy the frontend
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    shutil.copytree(FRONTEND, OUT, ignore=lambda d, names: [n for n in names if n.startswith(".") or (
        os.path.isfile(os.path.join(d, n)) and os.path.splitext(n)[1].lower() not in STATIC_TYPES)])

    # ---- one file per API answer
    api = os.path.join(OUT, "api")
    os.makedirs(os.path.join(api, "storm"))
    os.makedirs(os.path.join(api, "model"))
    con = store.connect_ro()
    try:
        names = [r[0] for r in con.execute("SELECT name FROM datasets ORDER BY name")]
        written = 0
        for name in names:
            if not SAFE.match(name):
                continue
            obj = json.loads(gzip.decompress(store.get_gz(con, name)))   # checksum checked on the way out
            if name == "live":
                obj["host"] = "static"                            # tells the page how often it is refreshed
                obj["refresh_minutes"] = a.every
            path = name.replace("storm:", "storm/").replace("model:", "model/")
            with open(os.path.join(api, path), "w") as f:
                json.dump(obj, f, separators=(",", ":"))
            written += 1
        if "weather" not in names:                                # optional: the server answers null
            with open(os.path.join(api, "weather"), "w") as f:
                f.write("null")

        # 30-day history: this run plus what the online site already had
        hist = {h["time"]: h for h in (fetch(prev + "/api/history") or [] if prev else []) if isinstance(h, dict) and "time" in h}
        hist.update({h["time"]: h for h in store.history(con, store.KEEP_DAYS)})
        cut = (datetime.now(timezone.utc) - timedelta(days=store.KEEP_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows = [hist[k] for k in sorted(hist) if k >= cut]
        with open(os.path.join(api, "history"), "w") as f:
            json.dump(rows, f, separators=(",", ":"))
        with open(os.path.join(api, "source"), "w") as f:            # where the page fetches the live forecast
            json.dump({"live": a.live_url} if a.live_url else None, f)
        live = con.execute("SELECT updated FROM datasets WHERE name='live'").fetchone()
        with open(os.path.join(api, "health"), "w") as f:
            json.dump({"ok": True, "datasets": len(names), "last_live_update": live[0] if live else None, "host": "static"}, f)
    finally:
        con.close()

    # ---- hosting settings: security headers on everything, JSON type and no caching on the data
    headers = dict(SECURITY_HEADERS)
    if a.live_url:                                                     # the page may also read the live-data file
        headers["Content-Security-Policy"] = headers["Content-Security-Policy"].replace(
            "connect-src 'self'", "connect-src 'self' https://raw.githubusercontent.com")
    sec = [{"key": k, "value": v} for k, v in headers.items()]
    cfg = {
        "cleanUrls": False,
        "trailingSlash": False,
        "headers": [
            {"source": "/(.*)", "headers": sec},
            {"source": "/api/(.*)", "headers": [{"key": "Content-Type", "value": "application/json; charset=utf-8"},
                                                {"key": "Cache-Control", "value": "no-store"}]},
            {"source": "/(.*).(html|js|css)", "headers": [{"key": "Cache-Control", "value": "no-cache"}]},
        ],
    }
    with open(os.path.join(OUT, "vercel.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(OUT, "_headers"), "w") as f:              # the same rules in Netlify's format
        f.write("/*\n" + "".join(f"  {k}: {v}\n" for k, v in headers.items()))
        f.write("/api/*\n  Content-Type: application/json; charset=utf-8\n  Cache-Control: no-store\n")
        for ext in ("html", "js", "css"):
            f.write(f"/*.{ext}\n  Cache-Control: no-cache\n")

    n = sum(len(fs) for _, _, fs in os.walk(OUT))
    size = sum(os.path.getsize(os.path.join(d, x)) for d, _, fs in os.walk(OUT) for x in fs)
    print(f"Wrote {OUT}: {n} files, {size / 1e6:.1f} MB ({written} datasets, {len(rows)} history entries).")


if __name__ == "__main__":
    main()
