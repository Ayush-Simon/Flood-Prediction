"""Export the record of the live system for the report: every 15-minute update kept in the store.

  python3 backend/export_history.py                 last 7 days
  python3 backend/export_history.py --days 30       up to 30 days

Writes to reports/:
  live_run_updates.csv          one row per update: time, forecast rain, wettest hour, streets flagged, deepest sensor forecast
  live_run_sensor_forecasts.csv one row per update and sensor: forecast depth 5, 20 and 40 minutes ahead, rain the ANN expected
  fig_live_run.png              the run on one chart (needs matplotlib)
Use it after leaving backend/app.py running through a rainy day, then compare the sensor forecasts with
what FloodNet recorded (download the same days from NYC Open Data)."""
import argparse, gzip, json, os, sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import store

ROOT = os.path.dirname(store.HERE); REP = os.path.join(ROOT, "reports")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--days", type=int, default=7); a = ap.parse_args()
    con = store.connect_ro()
    cut = (datetime.now(timezone.utc) - timedelta(days=a.days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = con.execute("SELECT ts, body FROM live_history WHERE ts >= ? ORDER BY ts", (cut,)).fetchall()
    if not rows:
        print("No live updates stored yet. Run python3 backend/app.py first."); return
    os.makedirs(REP, exist_ok=True)
    up, sf = [], []
    for ts, body in rows:
        L = json.loads(gzip.decompress(body))
        sens = [s for s in L.get("sensors", []) if s.get("fc")]
        up.append([ts, L.get("now_local", ""), L["weather"].get("desc", ""), L["weather"].get("rain_now"), L.get("city_rain_mm"), L.get("peak_1h_mm"),
                   *(L.get("counts") or [0, 0, 0]), L.get("expected_reports"), round(max((max(s["fc"]) for s in sens), default=0), 2),
                   sum(1 for s in sens if s["fc"][7] > 0.5)])
        for s in sens:
            sf.append([ts, s["id"], s["name"], s["fc"][0], s["fc"][3], s["fc"][7], round(sum(s.get("rain_fc", [])), 2)])
    import csv
    with open(os.path.join(REP, "live_run_updates.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["time_utc", "time_new_york", "weather", "rain_last_15min_mm", "rain_next_24h_mm", "wettest_hour_mm",
                                       "streets_likely", "streets_risky", "streets_watch", "expected_reports", "deepest_sensor_forecast_in", "sensors_wet_in_40min"])
        w.writerows(up)
    with open(os.path.join(REP, "live_run_sensor_forecasts.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["time_utc", "sensor_id", "sensor_name", "depth_in_5min", "depth_in_20min", "depth_in_40min", "ann_rain_next_40min_mm"])
        w.writerows(sf)
    print(f"{len(up)} updates from {up[0][0]} to {up[-1][0]} -> reports/live_run_updates.csv, live_run_sensor_forecasts.csv")
    print(f"most rain forecast: {max(r[4] or 0 for r in up)} mm | most streets flagged: {max(r[6] for r in up)} | deepest sensor forecast: {max(r[10] for r in up)} in")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipped the chart"); return
    t = [datetime.strptime(r[0], "%Y-%m-%dT%H:%M:%SZ") for r in up]
    fig, ax = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    ax[0].plot(t, [r[4] or 0 for r in up], label="rain forecast, next 24 h (mm)"); ax[0].plot(t, [r[3] or 0 for r in up], label="rain in last 15 min (mm)"); ax[0].legend()
    ax[1].plot(t, [r[6] for r in up], color="#d92d20", label="streets likely flooded"); ax[1].plot(t, [r[7] for r in up], color="#e07b00", label="risky"); ax[1].legend()
    ax[2].plot(t, [r[10] for r in up], color="#1769c4", label="deepest sensor forecast (in)"); ax[2].legend()
    for x in ax: x.grid(alpha=.3)
    ax[0].set_title("NYC Flood Watch live run (times in UTC)"); fig.tight_layout(); fig.savefig(os.path.join(REP, "fig_live_run.png"), dpi=150)
    print("chart -> reports/fig_live_run.png")


if __name__ == "__main__":
    main()
