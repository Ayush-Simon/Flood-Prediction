"""Stage 11: build the data files the dashboard reads (backend/data/*.json; then run backend/build_store.py). Replay of 2026 storm days.
Run: python scripts/08_build_dashboard_data.py      Needs: pandas, numpy, shapely, xgboost.
- roads.json            every drivable road as a simplified line (streets first, same order as the model's street table)
- index.json            list of storm days + overall test results of every model
- storm_YYYY-MM-DD.json rain, street risk, sensor depths and model predictions for that day"""
import os, json, re
import numpy as np, pandas as pd, xgboost as xgb
from shapely import wkt
from shapely.ops import linemerge
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P, R, REP, OUT = f"{ROOT}/data/processed", f"{ROOT}/data/raw", f"{ROOT}/reports", f"{ROOT}/backend/data"
os.makedirs(OUT, exist_ok=True)
NY = "America/New_York"
LON0, LAT0, Q = -74.27, 40.48, 1e5
LEVELS = [("high", 0.010), ("medium", 0.003), ("watch", 0.001)]
RAINF = ["rain_total", "rain_max1h", "rain_max3h", "rain_max6h", "wet_hours", "rain_prev1", "rain_prev3"]
BORO = {1: "Manhattan", 2: "Bronx", 3: "Brooklyn", 4: "Queens", 5: "Staten Island"}
dump = lambda o, f: json.dump(o, open(f"{OUT}/{f}", "w"), separators=(",", ":"))

# ------------------------------------------------------------------ roads
st = pd.read_csv(f"{P}/nyc_street_live_features.csv")
cl = pd.read_csv(f"{R}/nyc_street_centerline.csv", usecols=["the_geom", "PHYSICALID", "STATUS", "RW_TYPE", "Full Street Name", "Borough Code"], low_memory=False)
cl = cl[cl["STATUS"] == 2].drop_duplicates("PHYSICALID").set_index("PHYSICALID")
other = cl[cl["RW_TYPE"].isin([2, 3, 4, 9, 13])]                    # highway, bridge, tunnel, ramp, U-turn: drawn and routable, not scored
ids = list(st["street_id"]) + list(other.index)
def line(w):
    g = wkt.loads(w); g = linemerge(g) if g.geom_type == "MultiLineString" else g
    if g.geom_type == "MultiLineString": g = max(g.geoms, key=lambda x: x.length)
    g = g.simplify(0.00003, preserve_topology=False)
    return [v for x, y in g.coords for v in (int(round((x - LON0) * Q)), int(round((y - LAT0) * Q)))]
pts, off, kinds, names = [], [0], [], []
name_ix = {}
for i in ids:
    r = cl.loc[i]; c = line(r["the_geom"]); pts += c; off.append(len(pts) // 2)
    kinds.append(int(r["RW_TYPE"])); nm = str(r["Full Street Name"]).title() if pd.notna(r["Full Street Name"]) else "Unnamed road"
    names.append(name_ix.setdefault(nm, len(name_ix)))
dump({"lon0": LON0, "lat0": LAT0, "q": Q, "n_streets": len(st), "pts": pts, "off": off, "kind": kinds, "name": names,
      "names": list(name_ix), "boro": [int(b) for b in st["boro"]]}, "roads.json")
print("roads:", len(ids), "| points:", len(pts) // 2, "| file MB:", round(os.path.getsize(f"{OUT}/roads.json") / 1e6, 1))

# ------------------------------------------------------------------ storm days
days = pd.read_csv(f"{P}/nyc_rain_daily_by_cell.csv", parse_dates=["date"])
share = st["rain_cell"].value_counts(normalize=True)
city = days.assign(w=days["rain_cell"].map(share)).groupby("date").apply(lambda g: (g["rain_total"] * g["w"]).sum())
storm_days = [d for d in city.index if d.year == 2026 and d <= pd.Timestamp("2026-10-31") and city[d] >= 5]

info = json.load(open(f"{ROOT}/models/road_risk_model_info.json")); FEATS = info["features"]
model = xgb.Booster(); model.load_model(f"{ROOT}/models/road_risk_xgb.json")
cm = pd.read_csv(f"{P}/nyc_complaints_matched.csv", parse_dates=["date"]); cm = cm[cm["kind"] == "street_flood"]
sid_ix = {s: i for i, s in enumerate(st["street_id"])}
last_report = cm["date"].max()

g5 = pd.read_csv(f"{P}/nyc_rain_5min_stations.csv", parse_dates=["time_utc"])
g5["t"] = g5["time_utc"].dt.tz_localize("UTC").dt.tz_convert(NY).dt.tz_localize(None)
g5.loc[g5["coverage"] < 0.6, "rain_mm"] = np.nan
hr = pd.read_csv(f"{R}/nyc_rain_hourly_openmeteo.csv"); hr["t"] = pd.to_datetime(hr["time"], utc=True).dt.tz_convert(NY).dt.tz_localize(None)
hr = hr.groupby("t")["precipitation"].mean()

dp = pd.read_csv(f"{P}/nyc_depth_rain_5min.csv", parse_dates=["time"], usecols=["Sensor ID", "time", "depth_in"])
dp["t"] = dp["time"].dt.tz_localize("UTC").dt.tz_convert(NY).dt.tz_localize(None)
pr = pd.read_csv(f"{REP}/depth5_predictions_test.csv", parse_dates=["time"])
MODELS = sorted({c.split("|")[0] for c in pr.columns if "|" in c}, key=lambda m: list(pr.columns).index(m + "|5min"))
pr["t"] = pr["time"].dt.tz_localize("UTC").dt.tz_convert(NY).dt.tz_localize(None)
pred_start = pr["t"].min()
se = pd.read_csv(f"{R}/floodnet_sensors.csv").drop_duplicates("Sensor ID").set_index("Sensor ID")
sensors = [s for s in dp["Sensor ID"].unique()]
slots = pd.timedelta_range("5min", "24h", freq="5min")

index = []
for d in storm_days:
    key = d.strftime("%Y-%m-%d"); nxt = d + pd.Timedelta(days=1)
    # --- street risk
    g = days[days["date"] == d].set_index("rain_cell")
    X = pd.DataFrame(index=st.index)
    for f in FEATS:
        if f in RAINF: X[f] = st["rain_cell"].map(g[f])
        elif f == "month_sin": X[f] = np.sin(2 * np.pi * d.month / 12)
        elif f == "month_cos": X[f] = np.cos(2 * np.pi * d.month / 12)
        else: X[f] = st[f].fillna(-1)
    p = model.predict(xgb.DMatrix(X[FEATS].values.astype(np.float32)), iteration_range=(0, int(model.attributes()["best_iteration"]) + 1))
    flag = np.where(p >= LEVELS[-1][1])[0]; flag = flag[np.argsort(-p[flag])]
    lvl = np.select([p >= t for _, t in LEVELS], [3, 2, 1], 0)
    has_rep = d <= last_report
    rep = sorted({sid_ix[s] for s in cm.loc[cm["date"].between(d, nxt), "street_id"] if s in sid_ix}) if has_rep else []
    boro = {BORO[b]: [int(((st["boro"] == b) & (lvl == k)).sum()) for k in (3, 2, 1)] for b in BORO}
    # --- rain
    gd = g5[(g5["t"] > d) & (g5["t"] <= nxt)]
    gauges = {s: [None if pd.isna(v) else round(float(v), 2) for v in x.set_index("t")["rain_mm"].reindex(d + slots)] for s, x in gd.groupby("station")}
    hd = hr[(hr.index > d) & (hr.index <= nxt)]
    # --- sensors
    dd = dp[(dp["t"] > d) & (dp["t"] <= nxt)]
    has_pred = d >= pred_start.normalize()
    sens = []
    for s in sensors:
        o = dd[dd["Sensor ID"] == s].set_index("t")["depth_in"].reindex(d + slots)
        row = {"id": s, "name": str(se.loc[s, "Sensor Name"]), "x": int(round((se.loc[s, "Longitude"] - LON0) * Q)),
               "y": int(round((se.loc[s, "Latitude"] - LAT0) * Q)), "peak": round(float(np.nan_to_num(o.max())), 1),
               "observed": bool(o.notna().any())}
        if has_pred and row["peak"] > 0:
            ps = pr[(pr["Sensor ID"] == s)]
            row["obs"] = [None if pd.isna(v) else round(float(v), 2) for v in o]
            row["pred"] = {}
            for h in (5, 20, 40):
                q = ps.assign(tt=ps["t"] + pd.Timedelta(minutes=h)).set_index("tt")
                q = q[~q.index.duplicated()]
                row["pred"][str(h)] = [[None if pd.isna(v) else round(float(v), 2) for v in q[f"{m}|{h}min"].reindex(d + slots)] for m in MODELS]
        sens.append(row)
    # --- per-storm model errors (20 min ahead, at times with water observed or predicted by persistence)
    errs = []
    if has_pred:
        q = pr[(pr["t"] > d) & (pr["t"] <= nxt)]
        q = q[(q["obs_20min"] > 0) | (q["Persistence|20min"] > 0)]
        if len(q) >= 10:
            errs = [[m] + [round(float(np.sqrt(((q[f"{m}|{h}min"] - q[f"obs_{h}min"]) ** 2).mean())), 2) for h in (5, 20, 40)] for m in MODELS]
    flooded = [s for s in sens if s["peak"] > 0]
    dump({"date": key, "city_rain_mm": round(float(city[d]), 1), "expected_reports": round(float(p.sum()), 1),
          "counts": [int((lvl == k).sum()) for k in (3, 2, 1)], "boro": boro,
          "risk_ix": flag.tolist(), "risk_p": [int(round(v * 1e4)) for v in p[flag]],
          "has_reports": bool(has_rep), "reported": rep, "reported_caught": int(sum(lvl[i] > 0 for i in rep)),
          "gauges": gauges, "hourly": [round(float(v), 2) for v in hd.values],
          "has_pred": bool(has_pred), "models": MODELS, "sensors": sens, "storm_errors": errs}, f"storm_{key}.json")
    gt = {s: float(np.nansum([v or 0 for v in a])) for s, a in gauges.items()}
    pk = max([max([v or 0 for v in a]) for a in gauges.values()] or [0])
    index.append({"date": key, "rain_mm": round(float(city[d]), 1), "gauge_mm": round(float(np.mean(list(gt.values()))), 1) if gt else None,
                  "peak5": round(pk, 1), "sensors_flooded": len(flooded), "deepest_in": max([s["peak"] for s in sens] or [0]),
                  "reports": len(rep) if has_rep else None, "has_pred": bool(has_pred), "has_reports": bool(has_rep)})

# ------------------------------------------------------------------ overall results
rf = pd.read_csv(f"{REP}/rain_forecast_metrics.csv"); rf = rf[(rf.period == "test") & (rf.subset == "wet hours")]
d5 = pd.read_csv(f"{REP}/depth5_metrics_test.csv"); rr = pd.read_csv(f"{REP}/road_risk_metrics_test.csv")
res = {
 "rain": [[m] + [round(float(g.set_index("horizon_h").loc[h, c]), 2) for c in ("RMSE", "R2") for h in (1, 2, 3)] for m, g in rf.groupby("model", sort=False)],
 "depth": [[m] + [round(float(g.set_index("minutes_ahead").loc[h, "R2_flood"]), 2) for h in (5, 20, 40)]
                + [round(float(g.set_index("minutes_ahead").loc[h, "RMSE_flood"]), 2) for h in (5, 20, 40)]
                + [round(float(g.set_index("minutes_ahead").loc[40, "onset_caught"]) * 100)] for m, g in d5.groupby("model", sort=False)],
 "road": [[r["model"], round(r["recall_top1pct"] * 100), round(r["recall_top5pct"] * 100), round(r["recall_top10pct"] * 100), round(r["ROC_AUC"], 2)] for _, r in rr.iterrows()],
}
dump({"storms": index, "results": res, "levels": LEVELS}, "index.json")
tot = sum(os.path.getsize(f"{OUT}/{f}") for f in os.listdir(OUT)) / 1e6
print("storm days:", len(index), "| with depth predictions:", sum(i["has_pred"] for i in index), "| with reports:", sum(i["has_reports"] for i in index), f"| total {tot:.1f} MB")
print(pd.DataFrame(index).sort_values("deepest_in", ascending=False).head(8).to_string(index=False))
