"""Stage 14d: correct the street model for how often people complain, and check it against measured water.

The street model (05c) learns from 311 street-flooding complaints. Busy areas complain more about everything, so the
model over-ranks them (10d: complaint history vs over-ranking, Spearman 0.57). The standard fix is an exposure offset:
  - exposure of an area = how many 311 requests of ANY other kind it makes (sample from 14c), per 1500-ft grid cell,
    smoothed over the 3 x 3 neighbouring cells; r = exposure / median exposure
  - complaint history features are divided by r (complaints per unit of complaining)
  - during training log(r) is given to XGBoost as a fixed offset (base_margin): the trees learn flood risk on top of
    "this area calls 311 r times as much as usual"
  - for the map, the offset is left out, so every street is scored as if it complained at the city's usual rate
Both models (as before / corrected) are retrained here with the same sample and settings as 05c, then scored as in 10d:
  1. against measured water at the FloodNet sensors (AUC; the target nobody trained on)
  2. bias: does complaint history still predict over-ranking? (Spearman, per sensor)
  3. against 2026 complaints (what 05c reported), to show what the correction costs there
Usage: python 14d_complaint_exposure.py <out_dir> <path to nyc_311_exposure_sample.csv>"""
import sys, json, warnings
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
OUT, EXPO = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(42)
WET_MM, T0, BIG, RADIUS_FT = 5.0, pd.Timestamp("2020-10-01"), 10000, 492
SPLIT = {"burn_in_end": "2021-09-30", "train_end": "2025-06-30", "val_end": "2025-12-31", "test_end": "2026-07-28"}

# ---------------------------------------------------------------- data and features: identical to 05c
st = pd.read_csv(f"{OUT}/nyc_street_features.csv"); st["sidx"] = np.arange(len(st)); N = len(st)
days = pd.read_csv(f"{OUT}/nyc_rain_daily_by_cell.csv", parse_dates=["date"])
cm = pd.read_csv(f"{OUT}/nyc_complaints_matched.csv", parse_dates=["date"]).merge(st[["street_id", "sidx"]], on="street_id")
w = st["rain_cell"].value_counts(normalize=True)
city = days.assign(wt=days["rain_cell"].map(w)).groupby("date").apply(lambda g: (g["rain_total"] * g["wt"]).sum())
wet = city[(city >= WET_MM) & (city.index > SPLIT["burn_in_end"]) & (city.index <= SPLIT["test_end"])].index; wetset = set(wet)
period = lambda d: "train" if d <= pd.Timestamp(SPLIT["train_end"]) else "val" if d <= pd.Timestamp(SPLIT["val_end"]) else "test"
sj = cm[cm["kind"] == "street_flood"].copy()
sj["storm"] = pd.to_datetime(np.where(sj["date"].isin(wetset), sj["date"],
                             np.where((sj["date"] - pd.Timedelta(days=1)).isin(wetset), sj["date"] - pd.Timedelta(days=1), pd.NaT)))
pos = sj.dropna(subset=["storm"]).drop_duplicates(["sidx", "storm"])[["sidx", "storm"]]; pos_key = set(zip(pos["sidx"], pos["storm"]))
ordn = lambda d: (d - T0).days
gxi, gyi = (st["x"] // 1500).astype(int), (st["y"] // 1500).astype(int)
st["gidx"] = pd.factorize(gxi * 100000 + gyi)[0]; grid_n = st.groupby("gidx").size(); cm["gidx"] = cm["sidx"].map(st["gidx"])
keys = lambda df, col: np.sort(df[col].values * BIG + (df["date"] - T0).dt.days.values)
K = {"hist_flood": keys(cm[cm.kind == "street_flood"], "sidx"), "hist_drain": keys(cm[cm.kind == "drain_clogged"], "sidx"),
     "area_flood": keys(cm[cm.kind == "street_flood"], "gidx")}
before = lambda k, u, d: np.searchsorted(k, u * BIG + d) - np.searchsorted(k, u * BIG)
STATIC = ["length_ft", "width_ft", "total_lanes", "travel_lanes", "speed_mph", "one_way", "boro", "drains", "drains_per_100ft",
          "drain_dist_ft", "drains_500ft", "nbr_streets", "mod_nuis", "mod_deep", "ext_nuis", "ext_deep", "mod_nbr_share"]
RAINF = ["rain_total", "rain_max1h", "rain_max3h", "rain_max6h", "wet_hours", "rain_prev1", "rain_prev3"]
HIST = ["hist_flood", "hist_drain", "area_flood"]; FEATS = STATIC + RAINF + ["month_sin", "month_cos"] + HIST
F = {f: i for i, f in enumerate(FEATS)}
S = st[STATIC].fillna(-1).values.astype(np.float32)
cell_code, cells = pd.factorize(st["rain_cell"])
rain_lookup = {(c, d): v for (c, d), v in days.set_index(["rain_cell", "date"])[RAINF].apply(tuple, axis=1).items()}

# ---------------------------------------------------------------- exposure: 311 requests of any other kind, per grid cell
ex = pd.read_csv(EXPO).rename(columns={"x_coordinate_state_plane": "x", "y_coordinate_state_plane": "y"})   # 14c file or a browser download
ex = ex.dropna(subset=["x", "y"])
if len(ex) < 10000:
    sys.exit(f"{EXPO} has only {len(ex)} rows: the 311 download did not finish. Run scripts/14c_download_311_exposure.py again.")
ex = ex[(ex["x"] > 900000) & (ex["x"] < 1080000) & (ex["y"] > 110000) & (ex["y"] < 280000)]      # inside New York City
cnt = pd.Series(1, index=pd.MultiIndex.from_arrays([(ex["x"] // 1500).astype(int), (ex["y"] // 1500).astype(int)])).groupby(level=[0, 1]).sum()
lut = cnt.to_dict()
smooth = np.zeros(N)
for dx in (-1, 0, 1):
    for dy in (-1, 0, 1):
        smooth += np.array([lut.get((a + dx, b + dy), 0) for a, b in zip(gxi, gyi)])
expo = smooth + 1.0
r = expo / np.median(expo); log_r = np.log(r).astype(np.float32)
cell_r = pd.Series(r).groupby(st["gidx"]).mean()
print(f"exposure sample: {len(ex):,} requests; area exposure from {r.min():.2f}x to {r.max():.1f}x the median (middle 90%: {np.quantile(r, .05):.2f}x to {np.quantile(r, .95):.1f}x)")
hist_all = before(K["hist_flood"], np.arange(N), ordn(pd.Timestamp(SPLIT["test_end"])))
print(f"streets: Spearman between exposure and their flooding-complaint count {spearmanr(r, hist_all).statistic:.2f}")

def build(day, idx, corrected):
    o = ordn(day); yrs = max(o / 365.25, 1.0)
    R = np.array([rain_lookup[(c, day)] for c in cells], dtype=np.float32)[cell_code[idx]]
    mo = np.full((len(idx), 2), [np.sin(2 * np.pi * day.month / 12), np.cos(2 * np.pi * day.month / 12)], dtype=np.float32)
    g = st["gidx"].values[idx]
    Hh = np.column_stack([before(K["hist_flood"], idx, o) / yrs, before(K["hist_drain"], idx, o) / yrs,
                          before(K["area_flood"], g, o) / yrs / grid_n.values[g]]).astype(np.float32)
    if corrected:                                                             # complaints per unit of complaining
        Hh[:, 0] /= r[idx]; Hh[:, 1] /= r[idx]; Hh[:, 2] /= cell_r.values[g]
    y = np.fromiter(((i, day) in pos_key for i in idx), bool, len(idx))
    return np.hstack([S[idx], R, mo, Hh]), y

def pick(dayset, frac, rg):
    """the same street sample as 05c: every flooded street plus a random share of the others, per storm day"""
    out = []
    for d in dayset:
        p = pos.loc[pos["storm"] == d, "sidx"].values
        neg = rg.choice(N, int(N * frac), replace=False); neg = neg[~np.isin(neg, p)]
        out.append((d, np.concatenate([p, neg])))
    return out

def assemble(picks, frac, corrected):
    X, Y, W, Bm = [], [], [], []
    for d, idx in picks:
        x, y = build(d, idx, corrected); X.append(x); Y.append(y); W.append(np.where(y, 1.0, 1.0 / frac)); Bm.append(log_r[idx])
    return np.vstack(X), np.concatenate(Y), np.concatenate(W), np.concatenate(Bm)

tr_days = [d for d in wet if period(d) == "train"]; va_days = [d for d in wet if period(d) == "val"]; te_days = [d for d in wet if period(d) == "test"]
rg42 = np.random.default_rng(42)                                         # 05c's generator, used in the same order
tr_pick, va_pick = pick(tr_days, 0.03, rg42), pick(va_days, 0.10, rg42)
models = {}
for name, corrected in (("As before", False), ("Exposure-corrected", True)):
    Xtr, ytr, wtr, btr = assemble(tr_pick, 0.03, corrected); Xva, yva, wva, bva = assemble(va_pick, 0.10, corrected)
    m = xgb.XGBClassifier(n_estimators=1500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                          n_jobs=2, eval_metric="aucpr", early_stopping_rounds=60, tree_method="hist")
    if corrected:
        m.fit(Xtr, ytr, sample_weight=wtr, base_margin=btr, eval_set=[(Xva, yva)], sample_weight_eval_set=[wva],
              base_margin_eval_set=[bva], verbose=False)
    else:
        m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], sample_weight_eval_set=[wva], verbose=False)
    models[name] = (m, corrected); print(f"{name}: {m.best_iteration} trees", flush=True)

def risk(name, X, idx, with_offset):
    m, corrected = models[name]
    bm = log_r[idx] if (corrected and with_offset) else None
    return m.predict_proba(X, base_margin=bm)[:, 1] if bm is not None else m.predict_proba(X)[:, 1]

# ---------------------------------------------------------------- 3. against 2026 complaints (as 05c)
comp = []
for name in models:
    for off in ((False, True) if models[name][1] else (False,)):
        Y, Pp = [], []
        for d in te_days:
            X, y = build(d, np.arange(N), models[name][1]); p = risk(name, X, np.arange(N), off)
            sub = rng.random(N) < 0.05; Y.append(np.r_[y[y], y[~y & sub]]); Pp.append(np.r_[p[y], p[~y & sub]])
        Y, Pp = np.concatenate(Y), np.concatenate(Pp); wt = np.where(Y, 1.0, 20.0)
        comp.append({"model": name + (" (with its area's exposure)" if off else ""), "complaint_AUC_2026": roc_auc_score(Y, Pp, sample_weight=wt),
                     "complaint_PR_AUC_2026": average_precision_score(Y, Pp, sample_weight=wt)})
comp = pd.DataFrame(comp)

# ---------------------------------------------------------------- 1 and 2. against measured water at FloodNet sensors (as 10d)
sens = pd.read_csv(f"{OUT}/floodnet_sensors.csv").drop_duplicates("Sensor ID").set_index("Sensor ID")
d5 = pd.read_csv(f"{OUT}/nyc_depth_rain_5min.csv", usecols=["Sensor ID", "time", "depth_in"], parse_dates=["time"])
d5["date"] = d5["time"].dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
span = d5.groupby("Sensor ID")["date"].agg(["min", "max"]); wet_by = d5.groupby(["Sensor ID", "date"])["depth_in"].max()
A = np.column_stack([st["mid_lon"], st["mid_lat"], np.ones(N)])
cx, cy = np.linalg.lstsq(A, st["x"], rcond=None)[0], np.linalg.lstsq(A, st["y"], rcond=None)[0]
near = {}
for sid in span.index:
    if sid not in sens.index: continue
    px, py = cx @ [sens.loc[sid, "Longitude"], sens.loc[sid, "Latitude"], 1], cy @ [sens.loc[sid, "Longitude"], sens.loc[sid, "Latitude"], 1]
    dd = np.hypot(st["x"].values - px, st["y"].values - py); ii = np.where(dd <= RADIUS_FT)[0]
    near[sid] = ii if len(ii) else np.array([int(np.argmin(dd))])
allidx = np.unique(np.concatenate(list(near.values())))
rows = []
for d in wet:
    act = [s for s in near if span.loc[s, "min"] <= d <= span.loc[s, "max"]]
    if not act: continue
    pos_i = {s: i for i, s in enumerate(allidx)}
    X0, _ = build(d, allidx, False); X1, _ = build(d, allidx, True)
    p0, p1 = risk("As before", X0, allidx, False), risk("Exposure-corrected", X1, allidx, False)
    for s in act:
        ii = [pos_i[i] for i in near[s]]
        rows.append({"sensor": s, "date": d, "period": "test (2026)" if d > pd.Timestamp(SPLIT["val_end"]) else "earlier",
                     "water_in": float(wet_by.get((s, d), 0.0)), "risk_before": float(p0[ii].max()), "risk_corrected": float(p1[ii].max()),
                     "rain_mm": float(X0[ii, F["rain_total"]].max()), "hist": float(X0[ii, F["hist_flood"]].max() + 0.01 * X0[ii, F["area_flood"]].max()),
                     "exposure": float(r[near[s]].mean())})
df = pd.DataFrame(rows); df["flooded"] = df["water_in"] >= 1.0
df.to_csv(f"{OUT}/rig_exposure_vs_water.csv", index=False)

res = []
for per, g in (("all storm days", df), ("test (2026) only", df[df.period == "test (2026)"])):
    for lab, col in (("Street model, as before", "risk_before"), ("Street model, exposure-corrected", "risk_corrected"), ("Rain alone", "rain_mm")):
        # resample whole sensors for an interval
        sens_ids = g["sensor"].unique(); gi = {s: np.where(g["sensor"].values == s)[0] for s in sens_ids}; bs = []
        for _ in range(500):
            i = np.concatenate([gi[s] for s in rng.choice(sens_ids, len(sens_ids))])
            if g["flooded"].values[i].any() and not g["flooded"].values[i].all(): bs.append(roc_auc_score(g["flooded"].values[i], g[col].values[i]))
        res.append({"days": per, "ranking": lab, "sensor_days": len(g), "flooded": int(g["flooded"].sum()),
                    "AUC": roc_auc_score(g["flooded"], g[col]), "AUC_lo": np.quantile(bs, .025), "AUC_hi": np.quantile(bs, .975),
                    "avg_precision": average_precision_score(g["flooded"], g[col])})
res = pd.DataFrame(res)
# is the corrected model really better against measured water? paired: the same resampled sensors for both models
pair, rgp = [], np.random.default_rng(0)
for per, g in (("all storm days", df), ("test (2026) only", df[df.period == "test (2026)"])):
    sids = g["sensor"].unique(); gi = {k: np.where(g["sensor"].values == k)[0] for k in sids}; diff = []
    for _ in range(2000):
        i = np.concatenate([gi[k] for k in rgp.choice(sids, len(sids))]); yy = g["flooded"].values[i]
        if yy.all() or not yy.any(): continue
        diff.append(roc_auc_score(yy, g["risk_corrected"].values[i]) - roc_auc_score(yy, g["risk_before"].values[i]))
    diff = np.array(diff)
    pair.append({"days": per, "AUC_gain": roc_auc_score(g["flooded"], g["risk_corrected"]) - roc_auc_score(g["flooded"], g["risk_before"]),
                 "lo": np.quantile(diff, .025), "hi": np.quantile(diff, .975), "p_boot": 2 * min((diff <= 0).mean(), (diff >= 0).mean())})
pair = pd.DataFrame(pair); pair.to_csv(f"{OUT}/rig_exposure_auc_gain.csv", index=False)

bias = []
ps = df.groupby("sensor").agg(flood_rate=("flooded", "mean"), hist=("hist", "mean"), exposure=("exposure", "mean"),
                              before=("risk_before", "mean"), corrected=("risk_corrected", "mean"), days=("flooded", "size"))
ps = ps[ps["days"] >= 10]
for lab, col in (("as before", "before"), ("exposure-corrected", "corrected")):
    over = ps[col].rank(pct=True) - ps["flood_rate"].rank(pct=True)
    a, b, c = spearmanr(ps[col], ps["flood_rate"]), spearmanr(ps["hist"], over), spearmanr(ps["exposure"], over)
    hi = ps["hist"] >= ps["hist"].median()
    bias.append({"model": lab, "sensors": len(ps), "risk_vs_measured_flood_rate": a.statistic, "p_risk": a.pvalue,
                 "complaint_history_vs_over_ranking": b.statistic, "p_history": b.pvalue,
                 "exposure_vs_over_ranking": c.statistic, "p_exposure": c.pvalue,
                 "over_ranking_complaint_heavy": over[hi].mean(), "over_ranking_quiet": over[~hi].mean()})
bias = pd.DataFrame(bias)
res.to_csv(f"{OUT}/rig_exposure_scores.csv", index=False); bias.to_csv(f"{OUT}/rig_exposure_bias.csv", index=False)
comp.to_csv(f"{OUT}/rig_exposure_complaints.csv", index=False)
pd.set_option("display.width", 220)
print("\n1. Which sensor-days had 1 inch of water or more? (AUC: 0.5 = chance; interval from resampling sensors)")
print(res.round(3).to_string(index=False))
print("\n1b. AUC gain of the corrected model against measured water (paired, resampling sensors):")
print(pair.round(3).to_string(index=False))
print("\n2. Complaint bias per sensor (Spearman): over-ranking = risk rank minus measured-water rank")
print(bias.round(3).to_string(index=False))
print("\n3. Predicting 2026 complaints (what the model was trained for):")
print(comp.round(3).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for k, (lab, col) in enumerate((("as before", "before"), ("exposure-corrected", "corrected"))):
    over = ps[col].rank(pct=True) - ps["flood_rate"].rank(pct=True)
    ax[k].scatter(ps["hist"], over, s=40); ax[k].axhline(0, color="k", lw=.8); ax[k].set_xscale("symlog", linthresh=0.1)
    ax[k].set_xlim(-0.02, ps["hist"].max() * 1.6)                     # no negative complaint counts; keep the last point in view
    b = bias.iloc[k]
    ax[k].set_title(f"Street model {lab}: Spearman {b.complaint_history_vs_over_ranking:.2f} (p {b.p_history:.3f})")
    ax[k].set_xlabel("past flooding complaints near the sensor (per year)"); ax[k].set_ylabel("over-ranking (risk rank − water rank)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_exposure_bias.png", dpi=150)
print("\nsaved rig_exposure_*.csv and fig_exposure_bias.png")
