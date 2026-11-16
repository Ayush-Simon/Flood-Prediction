"""Stage 10d (phase 4): does the street model predict measured WATER, or only 311 complaints?
The street model was trained on complaints. Here it is scored against something it never saw:
whether a FloodNet sensor measured 1 inch of water or more on each storm day.
  - risk at a sensor = the highest flood chance among streets within 150 m of it
  - compared with simple rankings: rain alone, the city flood map, complaint history alone
  - complaint bias: are sensors in complaint-heavy areas over-ranked for the water they really get?
Usage: python 10d_street_model_vs_water.py <out_dir>"""
import sys, json
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
OUT = sys.argv[1]; RADIUS_FT = 492; T0 = pd.Timestamp("2020-10-01"); BIG = 10000
info = json.load(open(f"{OUT}/road_risk_model_info.json")); FEATS = info["features"]; SPLIT = info["split"]
bst = xgb.Booster(); bst.load_model(f"{OUT}/road_risk_xgb.json")
BEST = (0, int(bst.attributes().get("best_iteration", bst.num_boosted_rounds() - 1)) + 1)

st = pd.read_csv(f"{OUT}/nyc_street_features.csv"); N = len(st)
days = pd.read_csv(f"{OUT}/nyc_rain_daily_by_cell.csv", parse_dates=["date"])
cm = pd.read_csv(f"{OUT}/nyc_complaints_matched.csv", parse_dates=["date"]).merge(st[["street_id"]].assign(sidx=np.arange(N)), on="street_id")

# ---------- the same features as 05c (history counted only before each day) ----------
RAINF = ["rain_total", "rain_max1h", "rain_max3h", "rain_max6h", "wet_hours", "rain_prev1", "rain_prev3"]
STATIC = [f for f in FEATS if f not in RAINF + ["month_sin", "month_cos", "hist_flood", "hist_drain", "area_flood"]]
gx = (st["x"] // 1500).astype(int); gy = (st["y"] // 1500).astype(int); st["gidx"] = pd.factorize(gx * 100000 + gy)[0]
grid_n = st.groupby("gidx").size(); cm["gidx"] = cm["sidx"].map(st["gidx"])
ordn = lambda d: (d - T0).days
keys = lambda df, col: np.sort(df[col].values * BIG + (df["date"] - T0).dt.days.values)
K = {"hist_flood": keys(cm[cm.kind == "street_flood"], "sidx"), "hist_drain": keys(cm[cm.kind == "drain_clogged"], "sidx"),
     "area_flood": keys(cm[cm.kind == "street_flood"], "gidx")}
before = lambda k, u, d: np.searchsorted(k, u * BIG + d) - np.searchsorted(k, u * BIG)
cell_code, cells = pd.factorize(st["rain_cell"])
rain_lookup = {(c, d): v for (c, d), v in days.set_index(["rain_cell", "date"])[RAINF].apply(tuple, axis=1).items()}
w = st["rain_cell"].value_counts(normalize=True)
city = days.assign(wt=days["rain_cell"].map(w)).groupby("date").apply(lambda g: (g["rain_total"] * g["wt"]).sum())

# ---------- sensors and the streets around them ----------
sens = pd.read_csv(f"{OUT}/floodnet_sensors.csv").drop_duplicates("Sensor ID").set_index("Sensor ID")
d5 = pd.read_csv(f"{OUT}/nyc_depth_rain_5min.csv", usecols=["Sensor ID", "time", "depth_in"], parse_dates=["time"])
d5["date"] = d5["time"].dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
span = d5.groupby("Sensor ID")["date"].agg(["min", "max"])
wet_by = d5.groupby(["Sensor ID", "date"])["depth_in"].max()
# sensor position in the street table's coordinates (feet, NY state plane): fit from street midpoints
A = np.column_stack([st["mid_lon"], st["mid_lat"], np.ones(N)])
cx, cy = np.linalg.lstsq(A, st["x"], rcond=None)[0], np.linalg.lstsq(A, st["y"], rcond=None)[0]
near = {}
for sid in span.index:
    if sid not in sens.index: continue
    lon, lat = sens.loc[sid, "Longitude"], sens.loc[sid, "Latitude"]
    px, py = cx @ [lon, lat, 1], cy @ [lon, lat, 1]
    dd = np.hypot(st["x"].values - px, st["y"].values - py); ii = np.where(dd <= RADIUS_FT)[0]
    near[sid] = ii if len(ii) else np.array([int(np.argmin(dd))])
print(f"{len(near)} sensors, {np.mean([len(v) for v in near.values()]):.1f} streets each on average within 150 m")

wet_days = city[(city >= info["wet_day_mm"]) & (city.index > SPLIT["burn_in_end"]) & (city.index <= SPLIT["test_end"])].index
allidx = np.unique(np.concatenate(list(near.values())))
rows = []
for d in wet_days:
    act = [s for s in near if span.loc[s, "min"] <= d <= span.loc[s, "max"]]
    if not act: continue
    o = ordn(d); yrs = max(o / 365.25, 1.0); idx = allidx
    R = np.array([rain_lookup[(c, d)] for c in cells], dtype=np.float32)[cell_code[idx]]
    g = st["gidx"].values[idx]
    Hh = np.column_stack([before(K["hist_flood"], idx, o) / yrs, before(K["hist_drain"], idx, o) / yrs, before(K["area_flood"], g, o) / yrs / grid_n.values[g]])
    X = pd.DataFrame(st[STATIC].values[idx], columns=STATIC).fillna(-1)
    for j, f in enumerate(RAINF): X[f] = R[:, j]
    X["month_sin"], X["month_cos"] = np.sin(2 * np.pi * d.month / 12), np.cos(2 * np.pi * d.month / 12)
    X["hist_flood"], X["hist_drain"], X["area_flood"] = Hh[:, 0], Hh[:, 1], Hh[:, 2]
    p = bst.predict(xgb.DMatrix(X[FEATS].values.astype(np.float32)), iteration_range=BEST)
    pos = {s: i for i, s in enumerate(idx)}
    for s in act:
        ii = [pos[i] for i in near[s]]
        rows.append({"sensor": s, "date": d, "period": "test (2026)" if d > pd.Timestamp(SPLIT["val_end"]) else "earlier",
                     "water_in": float(wet_by.get((s, d), 0.0)), "risk": float(p[ii].max()),
                     "rain_mm": float(X["rain_total"].values[ii].max()), "hist": float(Hh[ii, 0].max() + 0.01 * Hh[ii, 2].max()),
                     "flood_map": float((3 * X["mod_deep"] + 2 * X["mod_nuis"] + 1.5 * X["ext_deep"] + X["ext_nuis"]).values[ii].max())})
df = pd.DataFrame(rows); df["flooded"] = df["water_in"] >= 1.0
df.to_csv(f"{OUT}/rig_street_vs_water.csv", index=False)

res = []
for per, g in [("all storm days", df), ("test (2026) only", df[df.period == "test (2026)"])]:
    for name, col in (("Street model (XGBoost)", "risk"), ("Rain alone", "rain_mm"), ("Complaint history alone", "hist"), ("City flood map alone", "flood_map")):
        res.append({"days": per, "ranking": name, "sensor_days": len(g), "flooded": int(g["flooded"].sum()),
                    "AUC": roc_auc_score(g["flooded"], g[col]), "avg_precision": average_precision_score(g["flooded"], g[col])})
res = pd.DataFrame(res); res.to_csv(f"{OUT}/rig_street_vs_water_scores.csv", index=False)
print("\nCan each ranking tell which sensor-days had 1 inch of water or more? (AUC: 0.5 = chance, 1 = perfect)")
print(res.round(3).to_string(index=False))

# ---------- complaint bias: per sensor, model risk vs how often water really came ----------
ps = df.groupby("sensor").agg(flood_rate=("flooded", "mean"), mean_risk=("risk", "mean"), hist=("hist", "mean"), days=("flooded", "size"))
ps = ps[ps["days"] >= 10]
ps["risk_rank"] = ps["mean_risk"].rank(pct=True); ps["water_rank"] = ps["flood_rate"].rank(pct=True)
ps["over_ranked"] = ps["risk_rank"] - ps["water_rank"]
rho = spearmanr(ps["mean_risk"], ps["flood_rate"]); rho_b = spearmanr(ps["hist"], ps["over_ranked"])
ps.to_csv(f"{OUT}/rig_street_bias_per_sensor.csv")
print(f"\n{len(ps)} sensors: model risk vs measured flood frequency, Spearman {rho.statistic:.2f} (p {rho.pvalue:.3f})")
print(f"complaint history vs over-ranking (risk rank minus water rank): Spearman {rho_b.statistic:.2f} (p {rho_b.pvalue:.3f})")
hi = ps["hist"] >= ps["hist"].median()
print(f"sensors in complaint-heavy areas: mean over-ranking {ps.loc[hi,'over_ranked'].mean():+.2f}; in quiet areas {ps.loc[~hi,'over_ranked'].mean():+.2f}")
json.dump({"spearman_risk_vs_water": [rho.statistic, rho.pvalue], "spearman_hist_vs_overrank": [rho_b.statistic, rho_b.pvalue],
           "overrank_high_complaint": ps.loc[hi, "over_ranked"].mean(), "overrank_low_complaint": ps.loc[~hi, "over_ranked"].mean()},
          open(f"{OUT}/rig_street_bias_summary.json", "w"))
