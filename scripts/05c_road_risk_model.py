"""Step 2c: road risk model. Unit = one street on one storm day. Label = a 311 street-flooding
complaint on that street for that storm. Chronological split; tested on 2026 storms.
Usage: python 05c_road_risk_model.py <dir with 05b outputs> <out_dir>"""
import sys, json, warnings
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
IN, OUT = sys.argv[1:3]
rng = np.random.default_rng(42)
WET_MM = 5.0                                   # storm day = citywide mean rain >= 5 mm
T0 = pd.Timestamp("2020-10-01")
SPLIT = {"burn_in_end": "2021-09-30", "train_end": "2025-06-30", "val_end": "2025-12-31", "test_end": "2026-07-28"}

st = pd.read_csv(f"{IN}/nyc_street_features.csv")
st["sidx"] = np.arange(len(st)); N = len(st)
days = pd.read_csv(f"{IN}/nyc_rain_daily_by_cell.csv", parse_dates=["date"])
cm = pd.read_csv(f"{IN}/nyc_complaints_matched.csv", parse_dates=["date"]).merge(st[["street_id", "sidx"]], on="street_id")

# ---------- storm days (citywide) ----------
w = st["rain_cell"].value_counts(normalize=True)
city = days.assign(wt=days["rain_cell"].map(w)).groupby("date").apply(lambda g: (g["rain_total"] * g["wt"]).sum())
wet = city[(city >= WET_MM) & (city.index > SPLIT["burn_in_end"]) & (city.index <= SPLIT["test_end"])].index
wetset = set(wet)
def period(d): return "train" if d <= pd.Timestamp(SPLIT["train_end"]) else "val" if d <= pd.Timestamp(SPLIT["val_end"]) else "test"
print("Storm days:", pd.Series([period(d) for d in wet]).value_counts().to_dict())

# ---------- labels: complaint -> storm day (same day, else the day before) ----------
sj = cm[cm["kind"] == "street_flood"].copy()
sj["storm"] = np.where(sj["date"].isin(wetset), sj["date"],
              np.where((sj["date"] - pd.Timedelta(days=1)).isin(wetset), sj["date"] - pd.Timedelta(days=1), pd.NaT))
sj["storm"] = pd.to_datetime(sj["storm"])
print(f"Street-flooding complaints tied to a storm day: {sj['storm'].notna().sum()} of {len(sj)}")
pos = sj.dropna(subset=["storm"]).drop_duplicates(["sidx", "storm"])[["sidx", "storm"]]
pos_key = set(zip(pos["sidx"], pos["storm"]))

# ---------- history features: complaints strictly BEFORE the storm day (no leakage) ----------
def ordn(d): return (pd.to_datetime(d) - T0).dt.days.values if isinstance(d, pd.Series) else (d - T0).days
BIG = 10000
gx = (st["x"] // 1500).astype(int); gy = (st["y"] // 1500).astype(int)
st["gidx"] = pd.factorize(gx * 100000 + gy)[0]
grid_n = st.groupby("gidx").size()
def keys(df, col): return np.sort(df[col].values * BIG + ordn(df["date"]))
cm["gidx"] = cm["sidx"].map(st["gidx"])
K = {"hist_flood": keys(cm[cm.kind == "street_flood"], "sidx"), "hist_drain": keys(cm[cm.kind == "drain_clogged"], "sidx"),
     "area_flood": keys(cm[cm.kind == "street_flood"], "gidx")}
def before(k, unit, day): return np.searchsorted(k, unit * BIG + day) - np.searchsorted(k, unit * BIG)

STATIC = ["length_ft", "width_ft", "total_lanes", "travel_lanes", "speed_mph", "one_way", "boro", "drains",
          "drains_per_100ft", "drain_dist_ft", "drains_500ft", "nbr_streets",
          "mod_nuis", "mod_deep", "ext_nuis", "ext_deep", "mod_nbr_share"]
RAINF = ["rain_total", "rain_max1h", "rain_max3h", "rain_max6h", "wet_hours", "rain_prev1", "rain_prev3"]
HIST = ["hist_flood", "hist_drain", "area_flood"]
FEATS = STATIC + RAINF + ["month_sin", "month_cos"] + HIST
S = st[STATIC].fillna(-1).values.astype(np.float32)
cell_code = pd.factorize(st["rain_cell"])[0]; cells = pd.factorize(st["rain_cell"])[1]
rain_lookup = {(c, d): v for (c, d), v in days.set_index(["rain_cell", "date"])[RAINF].apply(tuple, axis=1).items()}

def build(day, idx):
    """feature matrix for streets `idx` on storm `day`"""
    o = ordn(day); yrs = max(o / 365.25, 1.0)
    R = np.array([rain_lookup[(c, day)] for c in cells], dtype=np.float32)[cell_code[idx]]
    mo = np.full((len(idx), 2), [np.sin(2*np.pi*day.month/12), np.cos(2*np.pi*day.month/12)], dtype=np.float32)
    g = st["gidx"].values[idx]
    H = np.column_stack([before(K["hist_flood"], idx, o) / yrs, before(K["hist_drain"], idx, o) / yrs,
                         before(K["area_flood"], g, o) / yrs / grid_n.values[g]]).astype(np.float32)
    y = np.fromiter(((i, day) in pos_key for i in idx), bool, len(idx))
    return np.hstack([S[idx], R, mo, H]), y

def sample(dayset, frac):
    X, Y, W = [], [], []
    for d in dayset:
        p = pos.loc[pos["storm"] == d, "sidx"].values
        neg = rng.choice(N, int(N * frac), replace=False); neg = neg[~np.isin(neg, p)]
        idx = np.concatenate([p, neg]); x, y = build(d, idx)
        X.append(x); Y.append(y); W.append(np.where(y, 1.0, 1.0 / frac))   # weight negatives back up
    return np.vstack(X), np.concatenate(Y), np.concatenate(W)

tr_days = [d for d in wet if period(d) == "train"]; va_days = [d for d in wet if period(d) == "val"]
te_days = [d for d in wet if period(d) == "test"]
Xtr, ytr, wtr = sample(tr_days, 0.03); Xva, yva, wva = sample(va_days, 0.10)
print(f"train rows {len(ytr)} ({ytr.sum()} flooded) | val rows {len(yva)} ({yva.sum()} flooded)")

# ---------- models ----------
F = {f: i for i, f in enumerate(FEATS)}
def fit_xgb(cols):
    ci = [F[c] for c in cols]
    m = xgb.XGBClassifier(n_estimators=1500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=5, n_jobs=2, eval_metric="aucpr", early_stopping_rounds=60, tree_method="hist")
    m.fit(Xtr[:, ci], ytr, sample_weight=wtr, eval_set=[(Xva[:, ci], yva)], sample_weight_eval_set=[wva], verbose=False)
    return m, ci
xgb_full, ci_full = fit_xgb(FEATS)
xgb_nohist, ci_nh = fit_xgb([f for f in FEATS if f not in HIST])
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=1.0))
Xl = lambda X: np.log1p(np.clip(X, 0, None))
lr.fit(Xl(Xtr), ytr, logisticregression__sample_weight=wtr)
print("XGBoost trees:", xgb_full.best_iteration, "| without history:", xgb_nohist.best_iteration)

MODELS = {
    "City flood map only":            lambda X: 3*X[:, F["mod_deep"]] + 2*X[:, F["mod_nuis"]] + 1.5*X[:, F["ext_deep"]] + X[:, F["ext_nuis"]],
    "Past complaints only":           lambda X: X[:, F["hist_flood"]] + 0.01 * X[:, F["area_flood"]],
    "Logistic regression":            lambda X: lr.predict_proba(Xl(X))[:, 1],
    "XGBoost (no complaint history)": lambda X: xgb_nohist.predict_proba(X[:, ci_nh])[:, 1],
    "XGBoost (all features)":         lambda X: xgb_full.predict_proba(X[:, ci_full])[:, 1],
}

# ---------- test: every street on every 2026 storm day ----------
allidx = np.arange(N); ks = [0.01, 0.05, 0.10]
agg = {n: {"y": [], "p": [], "hits": np.zeros(len(ks))} for n in MODELS}
perday = []; total_pos = 0; keep = {}
for d in te_days:
    X, y = build(d, allidx); total_pos += y.sum()
    for n, f in MODELS.items():
        p = f(X).astype(np.float32); order = np.argsort(-p, kind="stable")
        hits = np.array([y[order[:int(N*k)]].sum() for k in ks]); agg[n]["hits"] += hits
        sub = rng.random(N) < 0.05                               # 5% of dry streets for the pooled AUCs
        agg[n]["y"].append(np.concatenate([y[y], y[~y & sub]])); agg[n]["p"].append(np.concatenate([p[y], p[~y & sub]]))
        if n == "XGBoost (all features)":
            perday.append(dict(date=d.date(), city_rain_mm=round(city[d], 1), flooded_streets=int(y.sum()),
                               expected_flooded=round(float(p.sum()), 1), caught_top1pct=int(hits[0]), caught_top5pct=int(hits[1])))
            keep[d] = p
rows = []
for n, a in agg.items():
    y = np.concatenate(a["y"]); p = np.concatenate(a["p"]); wt = np.where(y, 1.0, 20.0)
    rows.append(dict(model=n, ROC_AUC=roc_auc_score(y, p, sample_weight=wt), PR_AUC=average_precision_score(y, p, sample_weight=wt),
                     **{f"recall_top{int(k*100)}pct": h / total_pos for k, h in zip(ks, a["hits"])}))
met = pd.DataFrame(rows); met["lift_top1pct"] = met["recall_top1pct"] / 0.01
met.to_csv(f"{OUT}/road_risk_metrics_test.csv", index=False)
pday = pd.DataFrame(perday); pday.to_csv(f"{OUT}/road_risk_per_storm_test.csv", index=False)
base = total_pos / (N * len(te_days))
print(f"\nTEST: {len(te_days)} storm days x {N} streets, {total_pos} flooded street-days (base rate {base:.4%})")
print(met.round(4).to_string(index=False))
print("\nDay-level check: correlation of expected vs actual flooded streets per storm day:", round(pday["expected_flooded"].corr(pday["flooded_streets"]), 2), "| Spearman:", round(pday["expected_flooded"].corr(pday["flooded_streets"], method="spearman"), 2), "| rain-total-only Spearman:", round(pday["city_rain_mm"].corr(pday["flooded_streets"], method="spearman"), 2))
print("\nBiggest test storms (XGBoost all features):")
print(pday.sort_values("flooded_streets", ascending=False).head(8).to_string(index=False))

# ---------- importance, figures, saved model, scenario scores ----------
imp = pd.Series(xgb_full.get_booster().get_score(importance_type="gain"))
imp.index = [FEATS[int(i[1:])] for i in imp.index]; imp = (imp / imp.sum()).sort_values(ascending=False)
imp.round(4).to_csv(f"{OUT}/road_risk_feature_importance.csv", header=["gain_share"])
print("\nTop features:\n", imp.head(10).round(3).to_string())

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
x = np.arange(len(ks)); wd = 0.16
for i, r in met.iterrows():
    ax[0].bar(x + (i-2)*wd, [r[f"recall_top{int(k*100)}pct"] for k in ks], wd, label=r["model"])
ax[0].plot(x, ks, "k--", lw=1, label="random guess"); ax[0].set_xticks(x); ax[0].set_xticklabels(["top 1%", "top 5%", "top 10%"])
ax[0].set_ylabel("share of flooded streets caught"); ax[0].set_title("Test storms 2026: flooded streets found in highest-risk streets")
ax[0].legend(fontsize=7)
imp.head(12)[::-1].plot.barh(ax=ax[1]); ax[1].set_title("XGBoost: most useful features (share of gain)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_road_risk_results.png", dpi=150)

big = pday.sort_values("flooded_streets", ascending=False).iloc[0]["date"]; bd = pd.Timestamp(big)
p = keep[bd]; yb = np.fromiter(((i, bd) in pos_key for i in allidx), bool, N)
fig, ax = plt.subplots(figsize=(8, 8))
o = np.argsort(p); sc = ax.scatter(st["mid_lon"].values[o], st["mid_lat"].values[o], c=p[o], s=0.6, cmap="YlOrRd",
                                    vmax=np.quantile(p, 0.995))
ax.scatter(st["mid_lon"][yb], st["mid_lat"][yb], s=14, facecolors="none", edgecolors="blue", lw=0.7, label="flooding reported")
ax.set_title(f"Predicted street flood risk, storm of {big} (test)"); ax.legend(); ax.set_aspect(1.3); ax.axis("off")
fig.colorbar(sc, shrink=0.6, label="predicted probability"); fig.tight_layout(); fig.savefig(f"{OUT}/fig_road_risk_map.png", dpi=150)

xgb_full.save_model(f"{OUT}/road_risk_xgb.json")
json.dump({"features": FEATS, "split": SPLIT, "wet_day_mm": WET_MM, "history_origin": str(T0.date())},
          open(f"{OUT}/road_risk_model_info.json", "w"), indent=2)

# risk of every street under three typical storms (history as of the end of the data)
last = pd.Timestamp(SPLIT["test_end"]); cw = days[days["date"].isin(wetset)]
scen = {"light": cw[cw.rain_total.between(5, 15)], "moderate": cw[cw.rain_total.between(15, 40)], "heavy": cw[cw.rain_total > 40]}
outs = st[["street_id", "name", "boro", "mid_lat", "mid_lon"]].copy()
Xb, _ = build(te_days[-1], allidx)
for k, g in scen.items():
    X = Xb.copy(); med = g[RAINF].median()
    for f in RAINF: X[:, F[f]] = med[f]
    X[:, F["month_sin"]], X[:, F["month_cos"]] = np.sin(2*np.pi*8/12), np.cos(2*np.pi*8/12)
    outs[f"risk_{k}"] = xgb_full.predict_proba(X[:, ci_full])[:, 1].round(5)
    print(f"scenario {k}: {dict(med.round(1))}")
outs.to_csv(f"{OUT}/nyc_street_risk_scenarios.csv", index=False)
