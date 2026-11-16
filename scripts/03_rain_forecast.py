"""Phase 3: chronological split + rainfall forecasting (Stage 1 of the paper).
Models: persistence (baseline), ARIMA (AR form), XGBoost, ANN (MLP).
Predicts rain 1..6 hours ahead from information available at hour t."""
import json, sys, time, warnings
import numpy as np, pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

RAIN, OUT = sys.argv[1], sys.argv[2]
H = 6                                   # forecast horizon (hours)
LAGS = 24                               # past hours used as input
SPLIT = {"train_end": "2025-12-31 23:00", "val_start": "2026-01-01 00:00",
         "val_end": "2026-06-30 23:00", "test_start": "2026-07-01 00:00", "test_end": "2026-10-31 23:00"}
WET = 0.5

# ---------- data: one series per distinct rain grid cell ----------
r = pd.read_csv(RAIN); r["time"] = pd.to_datetime(r["time"])
piv = r.pivot(index="time", columns="Sensor ID", values="precipitation").sort_index()
cells, sensor_cell = {}, {}
for sid in piv.columns:
    key = tuple(np.round(piv[sid].values, 3))
    key = hash(key)
    if key not in cells: cells[key] = f"cell_{len(cells)+1}"
    sensor_cell[sid] = cells[key]
first = {c: s for s, c in sensor_cell.items()}
rain = pd.DataFrame({c: piv[s] for c, s in sorted(first.items(), key=lambda kv: kv[0])})
print("Distinct rain cells:", rain.shape[1], "| hours:", len(rain))
city_mean = rain.mean(axis=1)

def period(t):
    return np.where(t <= pd.Timestamp(SPLIT["train_end"]), "train",
           np.where(t <= pd.Timestamp(SPLIT["val_end"]), "val", "test"))

# ---------- supervised table ----------
rows = []
for c in rain.columns:
    s = rain[c]
    d = pd.DataFrame({"time": rain.index, "cell": c})
    for k in range(LAGS):
        d[f"lag{k}"] = s.shift(k).values                 # lag0 = rain in hour ending at t
    for w in (3, 6, 12, 24):
        d[f"sum{w}"] = s.rolling(w).sum().values
    d["city_lag0"] = city_mean.values
    d["city_sum3"] = city_mean.rolling(3).sum().values
    hr, mo = rain.index.hour, rain.index.month
    d["hr_sin"], d["hr_cos"] = np.sin(2*np.pi*hr/24), np.cos(2*np.pi*hr/24)
    d["mo_sin"], d["mo_cos"] = np.sin(2*np.pi*mo/12), np.cos(2*np.pi*mo/12)
    for h in range(1, H+1):
        d[f"y{h}"] = s.shift(-h).values
    rows.append(d)
df = pd.concat(rows, ignore_index=True).dropna().reset_index(drop=True)
df["period"] = period(df["time"])
X_cols = [c for c in df.columns if c.startswith(("lag", "sum", "city", "hr_", "mo_"))]
Y_cols = [f"y{h}" for h in range(1, H+1)]
# "wet" origin = rain in the last 3 h or within the forecast window -> what matters for floods
df["wet"] = (df["sum3"] >= WET) | (df[Y_cols].max(axis=1) >= WET)
for p in ("train", "val", "test"):
    sub = df[df.period == p]; print(f"{p}: {len(sub)} rows, {sub.wet.mean():.1%} wet origins")

tr, va, te = (df[df.period == p] for p in ("train", "val", "test"))
preds = {}

# ---------- 1. persistence ----------
preds["Persistence"] = {p: np.repeat(d[["lag0"]].values, H, axis=1) for p, d in (("val", va), ("test", te))}

# ---------- 2. ARIMA (AR(p) chosen by AIC, fitted per cell on training data) ----------
t0 = time.time(); ar_params = {}
for c in rain.columns:
    y = rain[c][: SPLIT["train_end"]].values[-24*365*2:]          # last 2 training years
    best = None
    for p in (1, 2, 3, 4, 6):
        m = ARIMA(y, order=(p, 0, 0), trend="c").fit()
        if best is None or m.aic < best[0]: best = (m.aic, p, m)
    m = best[2]; const = m.params[0]; phi = m.params[1:1+best[1]]
    mu = const                                                  # statsmodels 'c' = process mean
    ar_params[c] = (best[1], mu, phi)
def ar_forecast(d):
    out = np.zeros((len(d), H))
    for c, (p, mu, phi) in ar_params.items():
        idx = (d["cell"] == c).values
        hist = d.loc[idx, [f"lag{k}" for k in range(p)]].values - mu   # lag0, lag1 ...
        for h in range(H):
            nxt = (hist[:, :p] * phi).sum(axis=1)
            out[idx, h] = nxt + mu
            hist = np.column_stack([nxt, hist[:, :-1]]) if p > 1 else nxt[:, None]
    return np.clip(out, 0, None)
preds["ARIMA"] = {"val": ar_forecast(va), "test": ar_forecast(te)}
print("ARIMA orders:", {c: v[0] for c, v in ar_params.items()}, f"({time.time()-t0:.0f}s)")

# ---------- 3. XGBoost (one model per horizon) ----------
t0 = time.time(); xg = {"val": np.zeros((len(va), H)), "test": np.zeros((len(te), H))}
xgb_models = []
for h in range(H):
    m = xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, objective="reg:squarederror", n_jobs=2,
                         early_stopping_rounds=40, eval_metric="rmse")
    m.fit(tr[X_cols], tr[Y_cols[h]], eval_set=[(va[X_cols], va[Y_cols[h]])], verbose=False)
    xg["val"][:, h] = m.predict(va[X_cols]); xg["test"][:, h] = m.predict(te[X_cols])
    xgb_models.append(m)
preds["XGBoost"] = {k: np.clip(v, 0, None) for k, v in xg.items()}
print(f"XGBoost done ({time.time()-t0:.0f}s)")

# ---------- 4. ANN (MLP, all horizons at once) ----------
t0 = time.time()
sx = StandardScaler().fit(tr[X_cols])
ann = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", learning_rate_init=1e-3,
                   batch_size=256, max_iter=60, early_stopping=True, validation_fraction=0.1,
                   n_iter_no_change=6, random_state=42)
ann.fit(sx.transform(tr[X_cols]), tr[Y_cols].values)
preds["ANN"] = {p: np.clip(ann.predict(sx.transform(d[X_cols])), 0, None) for p, d in (("val", va), ("test", te))}
print(f"ANN done ({time.time()-t0:.0f}s, {ann.n_iter_} epochs)")

# ---------- evaluation ----------
def scores(y, yh):
    return mean_absolute_error(y, yh), np.sqrt(mean_squared_error(y, yh)), r2_score(y, yh)
recs = []
for name, pr in preds.items():
    for p, d in (("val", va), ("test", te)):
        for subset, mask in (("all hours", np.ones(len(d), bool)), ("wet hours", d["wet"].values)):
            for h in range(H):
                mae, rmse, r2 = scores(d[Y_cols[h]].values[mask], pr[p][mask, h])
                recs.append(dict(model=name, period=p, subset=subset, horizon_h=h+1, MAE=mae, RMSE=rmse, R2=r2))
met = pd.DataFrame(recs)
met.to_csv(f"{OUT}/rain_forecast_metrics.csv", index=False)

wet_val = met[(met.period == "val") & (met.subset == "wet hours")].groupby("model")["RMSE"].mean()
best = wet_val.idxmin()
print("\nValidation RMSE on wet hours (mean over 1-6 h):\n", wet_val.round(3).to_string())
print("Best model on validation:", best)

tbl = (met[(met.period == "test") & (met.subset == "wet hours")]
       .pivot_table(index="model", columns="horizon_h", values="RMSE").round(3))
print("\nTEST RMSE (mm/h) on wet hours by horizon:\n", tbl.to_string())
tbl2 = (met[(met.period == "test") & (met.subset == "wet hours")].groupby("model")[["MAE", "RMSE", "R2"]].mean().round(3))
print("\nTEST wet hours, averaged over horizons:\n", tbl2.to_string())

# ---------- forecasts for every hour (input to the flood-depth stage) ----------
allX = df[X_cols]
if best == "XGBoost":
    full = np.column_stack([m.predict(allX) for m in xgb_models])
elif best == "ANN":
    full = ann.predict(sx.transform(allX))
elif best == "ARIMA":
    full = ar_forecast(df)
else:
    full = np.repeat(df[["lag0"]].values, H, axis=1)
fc = df[["time", "cell", "period"]].copy()
for h in range(H): fc[f"rain_fc_h{h+1}"] = np.clip(full[:, h], 0, None).round(3)
fc["model"] = best
fc.to_csv(f"{OUT}/nyc_rain_forecasts.csv", index=False)
pd.Series(sensor_cell, name="cell").rename_axis("Sensor ID").to_csv(f"{OUT}/sensor_rain_cell.csv")
json.dump({**SPLIT, "horizon_hours": H, "input_lags_hours": LAGS, "best_rain_model": best},
          open(f"{OUT}/splits.json", "w"), indent=2)

# ---------- figures ----------
fig, ax = plt.subplots(figsize=(7, 4))
for name in preds:
    sub = met[(met.model == name) & (met.period == "test") & (met.subset == "wet hours")]
    ax.plot(sub.horizon_h, sub.RMSE, marker="o", label=name)
ax.set_xlabel("Forecast horizon (hours)"); ax.set_ylabel("RMSE (mm/h)")
ax.set_title("Rainfall forecast error, test period (wet hours)"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_rain_rmse_by_horizon.png", dpi=150)

c0 = te.groupby("cell")["lag0"].sum().idxmax()
sub = te[te.cell == c0].reset_index(drop=True)
i = sub["sum6"].idxmax(); w = sub.iloc[max(i-30, 0): i+30]
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(w["time"], w["lag0"], width=0.03, color="lightgray", label="observed")
for name in ("Persistence", "ARIMA", "XGBoost", "ANN"):
    pr = preds[name]["test"][(te.cell == c0).values][w.index, 0]
    ax.plot(w["time"] + pd.Timedelta(hours=1), pr, label=f"{name} (1 h ahead)")
ax.set_ylabel("rain (mm/h)"); ax.set_title(f"Biggest test storm, {c0}: 1-hour-ahead forecasts")
ax.legend(fontsize=8); fig.autofmt_xdate(); fig.tight_layout()
fig.savefig(f"{OUT}/fig_rain_test_storm.png", dpi=150)
print("\nSaved outputs to", OUT)
