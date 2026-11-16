"""Write just the live forecast, for the live-data branch that the hosted (Netlify) site reads.

  python3 backend/build_live_feed.py --out feed --history previous_history.json

Runs one live update and writes:
  feed/live.json      the forecast the page shows (same as /api/live on your own computer)
  feed/history.json   one summary line per update, last 30 days (the previous file plus this run)
  feed/health.json    when it last ran

Exits with code 1 if the live update fails, so the GitHub Action keeps the last good forecast instead of pushing.
"""
import argparse, gzip, json, os, sys
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import store  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="folder to write into")
    ap.add_argument("--history", help="the previous history.json, if there is one")
    ap.add_argument("--every", type=int, default=30, help="minutes between scheduled updates (shown on the site)")
    a = ap.parse_args()

    if not os.path.exists(store.DB):
        import build_store
        sys.argv = [sys.argv[0]]
        build_store.main()
    try:
        import updater
        updater.build()
    except Exception as e:
        print("Live update failed, keeping the last forecast:", e)
        sys.exit(1)

    con = store.connect_ro()
    try:
        live = json.loads(gzip.decompress(store.get_gz(con, "live")))
        recent = store.history(con, store.KEEP_DAYS)
    finally:
        con.close()
    live["host"] = "static"
    live["refresh_minutes"] = a.every

    old = []
    if a.history and os.path.exists(a.history):
        try:
            old = json.load(open(a.history))
        except ValueError:
            old = []
    hist = {h["time"]: h for h in old if isinstance(h, dict) and isinstance(h.get("time"), str)}
    hist.update({h["time"]: h for h in recent})
    cut = (datetime.now(timezone.utc) - timedelta(days=store.KEEP_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = [hist[k] for k in sorted(hist) if k >= cut]

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "live.json"), "w") as f:
        json.dump(live, f, separators=(",", ":"))
    with open(os.path.join(a.out, "history.json"), "w") as f:
        json.dump(rows, f, separators=(",", ":"))
    with open(os.path.join(a.out, "health.json"), "w") as f:
        json.dump({"ok": True, "last_live_update": live.get("updated"), "history_entries": len(rows)}, f)
    print(f"Live forecast {live.get('updated')} written to {a.out} ({len(rows)} history entries).")


if __name__ == "__main__":
    main()
