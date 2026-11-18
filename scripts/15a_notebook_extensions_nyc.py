"""Stage 15a: the "Urban_Flood_Project_Extensions" notebook, run on the real NYC data.

The notebook (written for a simulated single-location dataset) is kept as close to the original as possible:
  - same Keras models: rainfall ANN (Dense 64 -> 32 -> 8, relu), flood LSTM (LSTM 50 relu, Dropout 0.2,
    concatenated with the next step's rainfall, Dense 16 relu, Dense 1), Adam, MSE, EarlyStopping
  - LOOKBACK = 15 five-minute steps, HORIZON = 8 (40 minutes)
  - "original" inputs (rain, depth) and "enhanced" inputs (+ intensity, 15/30/60-min totals, change in rain)
  - MinMax scalers fitted on training data only
  - the LSTM learns ONE step ahead (trained with the real rain at that step, as in the notebook) and is then
    rolled forward 8 steps with the ANN's rain forecast, feeding its own predictions back in
  - risk levels: < 0.10 m low, 0.10 - 0.30 m moderate, > 0.30 m high (here 3.94 and 11.81 inches)
Changes needed for the NYC data (and why):
  - many sensors and storms instead of one long series: features and windows are built inside each storm window
  - the project's time split instead of 80/20: train up to 2025, Jan-Jun 2026 for early stopping, test Jul-Oct 2026
  - depth in inches (FloodNet's unit); scores are also given in the notebook's form (R², RMSE, MAE)
  - every test window is a forecast origin (the notebook used every 8th step); dry training windows are sampled
    at 15% as in the project (07c/07d) so training fits in memory and time
  - the notebook's rainfall-condition thresholds were the 33rd/66th percentiles of ALL steps, which are 0.0 when
    most steps are dry (its "moderate" group came out empty); here they follow the gauge's 0.254 mm tips
    (light = 1 tip, moderate = 2-3, heavy = 4 or more per 5 minutes)
Then it compares the result with the project's models on exactly the same test windows (rig_preds.npz).
Usage: python 15a_notebook_extensions_nyc.py <out_dir>"""
import os, sys, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = sys.argv[1]
LOOKBACK, HORIZON = 15, 8
SPLIT = {"train_end": "2025-12-31 23:59", "val_end": "2026-06-30 23:59"}
DRY_KEEP, FLOOD_IN = 0.15, 1.0
LOW_MAX, MODERATE_MAX = 0.10 / 0.0254, 0.30 / 0.0254          # the notebook's 0.10 m and 0.30 m, in inches
tf.keras.utils.set_random_seed(0)
t_start = time.time()

# ---------------------------------------------------------------- data, and the notebook's features inside each storm
d = pd.read_csv(f"{OUT}/nyc_depth_rain_5min.csv", parse_dates=["time"])
d["rain_mm"] = d["rain_mm"].fillna(0.0)
d = d.sort_values(["storm_id", "time"]).reset_index(drop=True)
g = d.groupby("storm_id")["rain_mm"]
d["rainfall_intensity"] = d["rain_mm"] * 12
d["cumulative_15min"] = g.transform(lambda s: s.rolling(3, min_periods=1).sum())
d["cumulative_30min"] = g.transform(lambda s: s.rolling(6, min_periods=1).sum())
d["cumulative_60min"] = g.transform(lambda s: s.rolling(12, min_periods=1).sum())
d["rainfall_change"] = g.diff().fillna(0)
d["period"] = np.where(d["time"] <= SPLIT["train_end"], "train", np.where(d["time"] <= SPLIT["val_end"], "val", "test"))
ENG = ["rain_mm", "depth_in", "rainfall_intensity", "cumulative_15min", "cumulative_30min", "cumulative_60min", "rainfall_change"]

tr_rows = d["period"] == "train"
rain_scaler = MinMaxScaler().fit(d.loc[tr_rows, ["rain_mm"]])
depth_scaler = MinMaxScaler().fit(d.loc[tr_rows, ["depth_in"]])
feature_scaler = MinMaxScaler().fit(d.loc[tr_rows, ENG])
rain_s = rain_scaler.transform(d[["rain_mm"]]).ravel().astype(np.float32)
depth_s = depth_scaler.transform(d[["depth_in"]]).ravel().astype(np.float32)
base_s = np.column_stack([rain_s, depth_s]).astype(np.float32)
eng_s = feature_scaler.transform(d[ENG]).astype(np.float32)
rain_raw, depth_raw = d["rain_mm"].values.astype(np.float32), d["depth_in"].values.astype(np.float32)

# ---------------------------------------------------------------- forecast origins: position i = first future step
# history = rows i-15..i-1, future = rows i..i+7, all inside one storm window
start = d.groupby("storm_id").cumcount().values; size = d.groupby("storm_id")["time"].transform("size").values
origin = np.where((start >= LOOKBACK) & (start <= size - HORIZON))[0]
per_o = d["period"].values[origin - 1]                                # period of the last observed step
fut_depth = np.stack([depth_raw[origin + k] for k in range(HORIZON)], 1)
hist_depth = np.stack([depth_raw[origin - LOOKBACK + k] for k in range(LOOKBACK)], 1)
floody = (fut_depth.max(1) > 0) | (hist_depth[:, -12:].max(1) > 0)
rng = np.random.default_rng(0)
keep = floody | (rng.random(len(origin)) < DRY_KEEP)
O = {p: origin[(per_o == p) & (keep if p != "test" else True)] for p in ("train", "val", "test")}
for p in O: print(f"{p}: {len(O[p]):,} forecast origins")

def windows(arr, pos):                                                # arr[pos-15 .. pos-1]
    return np.stack([arr[pos - LOOKBACK + k] for k in range(LOOKBACK)], 1)

# ---------------------------------------------------------------- stage 1: the notebook's rainfall ANN
X_rain = {p: windows(rain_s, O[p]) for p in ("train", "val", "test")}
y_rain = {p: np.stack([rain_s[O[p] + k] for k in range(HORIZON)], 1) for p in ("train", "val")}
SAVE = f"{OUT}/nb_models"; os.makedirs(SAVE, exist_ok=True)              # trained models are kept, so a rerun only re-scores
if os.path.exists(f"{SAVE}/rainfall_ann.keras"):
    rainfall_ann = tf.keras.models.load_model(f"{SAVE}/rainfall_ann.keras"); print("rainfall ANN loaded")
else:
    rain_input = Input(shape=(LOOKBACK,)); x = Dense(64, activation="relu")(rain_input); x = Dense(32, activation="relu")(x)
    rainfall_ann = Model(rain_input, Dense(HORIZON, activation="relu")(x))
    rainfall_ann.compile(optimizer="adam", loss="mse", metrics=["mae"])
    rainfall_ann.fit(X_rain["train"], y_rain["train"], epochs=100, batch_size=32, validation_data=(X_rain["val"], y_rain["val"]),
                     shuffle=False, verbose=0, callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)])
    rainfall_ann.save(f"{SAVE}/rainfall_ann.keras"); print(f"rainfall ANN trained ({time.time()-t_start:.0f}s)", flush=True)

# ---------------------------------------------------------------- stage 2: the notebook's flood LSTM, original and enhanced
def build_flood_model(number_of_features):
    history_input = Input(shape=(LOOKBACK, number_of_features), name="historical_features")
    future_rain_input = Input(shape=(1,), name="future_rainfall")
    x = LSTM(50, activation="relu")(history_input); x = Dropout(0.2)(x)
    x = Concatenate()([x, future_rain_input]); x = Dense(16, activation="relu")(x)
    m = Model([history_input, future_rain_input], Dense(1)(x)); m.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return m

def one_step_data(p, feats):                                          # history before step i, real rain at i, depth at i
    return [windows(feats, O[p]), rain_s[O[p]].reshape(-1, 1)], depth_s[O[p]]
models = {}
for name, feats, nf, fn in (("Original-input LSTM", base_s, 2, "baseline_lstm"), ("Feature-enhanced LSTM", eng_s, 7, "enhanced_lstm")):
    if os.path.exists(f"{SAVE}/{fn}.keras"):
        models[name] = tf.keras.models.load_model(f"{SAVE}/{fn}.keras"); print(f"{name} loaded"); continue
    m = build_flood_model(nf); (Xt, yt), (Xv, yv) = one_step_data("train", feats), one_step_data("val", feats)
    h = m.fit(Xt, yt, epochs=100, batch_size=32, validation_data=(Xv, yv), shuffle=False, verbose=0,
              callbacks=[EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)])
    m.save(f"{SAVE}/{fn}.keras"); models[name] = m
    print(f"{name} trained, {len(h.history['loss'])} epochs ({time.time()-t_start:.0f}s)", flush=True)

# ---------------------------------------------------------------- the notebook's 8-step recursive forecast, for every test origin at once
def forecast_eight_steps(pos):
    fr_s = rainfall_ann.predict(windows(rain_s, pos), batch_size=4096, verbose=0)
    fr = np.maximum(rain_scaler.inverse_transform(fr_s.reshape(-1, 1)).reshape(fr_s.shape), 0)
    base_hist, eng_hist = windows(base_s, pos).copy(), windows(eng_s, pos).copy()
    recent = windows(rain_raw, pos)[:, -12:].copy()                   # last 12 raw rain values, for the new features
    out = {k: np.zeros((len(pos), HORIZON), np.float32) for k in models}
    for i in range(HORIZON):
        nr = fr_s[:, i:i + 1]
        b = models["Original-input LSTM"].predict([base_hist, nr], batch_size=4096, verbose=0)[:, 0]
        e = models["Feature-enhanced LSTM"].predict([eng_hist, nr], batch_size=4096, verbose=0)[:, 0]
        bd = np.maximum(depth_scaler.inverse_transform(b.reshape(-1, 1)).ravel(), 0)
        ed = np.maximum(depth_scaler.inverse_transform(e.reshape(-1, 1)).ravel(), 0)
        out["Original-input LSTM"][:, i], out["Feature-enhanced LSTM"][:, i] = bd, ed
        base_hist = np.concatenate([base_hist[:, 1:], np.stack([nr[:, 0], b], 1)[:, None]], 1)
        recent = np.concatenate([recent[:, 1:], fr[:, i:i + 1]], 1)
        cur = recent[:, -1]
        new = np.column_stack([cur, ed, cur * 12, recent[:, -3:].sum(1), recent[:, -6:].sum(1), recent[:, -12:].sum(1), cur - recent[:, -2]])
        eng_hist = np.concatenate([eng_hist[:, 1:], feature_scaler.transform(pd.DataFrame(new, columns=ENG)).astype(np.float32)[:, None]], 1)
    return fr, out

te = O["test"]
if os.path.exists(f"{SAVE}/test_forecasts.npz"):
    zz = np.load(f"{SAVE}/test_forecasts.npz"); fr_te = zz["rain"]; pred = {"Original-input LSTM": zz["original"], "Feature-enhanced LSTM": zz["enhanced"]}
else:
    fr_te, pred = forecast_eight_steps(te)
    np.savez_compressed(f"{SAVE}/test_forecasts.npz", rain=fr_te, original=pred["Original-input LSTM"], enhanced=pred["Feature-enhanced LSTM"])
actual = np.stack([depth_raw[te + k] for k in range(HORIZON)], 1)
now_te = depth_raw[te - 1]
print(f"recursive forecasts done for {len(te):,} test origins ({time.time()-t_start:.0f}s)", flush=True)

# ---------------------------------------------------------------- 1. the notebook's comparison table (all 8 steps pooled)
def calculate_metrics(a, p):
    return {"R2": r2_score(a, p), "RMSE_in": np.sqrt(mean_squared_error(a, p)), "MAE_in": mean_absolute_error(a, p)}
fl_te = (actual.max(1) > 0) | (windows(depth_raw, te)[:, -12:].max(1) > 0)        # flood-related, as in 10b
rows = []
for subset, m in (("all test windows", np.ones(len(te), bool)), ("flood-related test windows", fl_te)):
    for name in models:
        rows.append({"windows": subset, "Model": name, **calculate_metrics(actual[m].ravel(), pred[name][m].ravel())})
    rows.append({"windows": subset, "Model": "Persistence (water stays as it is)", **calculate_metrics(actual[m].ravel(), np.repeat(now_te[m, None], HORIZON, 1).ravel())})
comparison = pd.DataFrame(rows)

# ---------------------------------------------------------------- 2. against the project's models on exactly the same windows
z = np.load(f"{OUT}/rig_preds.npz", allow_pickle=True)
key_nb = pd.MultiIndex.from_arrays([d["storm_id"].values[te - 1], d["time"].values[te - 1]])
key_rig = pd.MultiIndex.from_arrays([z["storm"], pd.to_datetime(z["time"]).values])
pos_rig = pd.Series(np.arange(len(key_rig)), index=key_rig)
j = pos_rig.reindex(key_nb).values; both = ~np.isnan(j); j = j[both].astype(int)
assert np.allclose(z["y"][j], actual[both], atol=1e-4), "windows do not line up"
# flood-related windows exactly as 10b defines them (water in the last hour or in the next 40 minutes)
d["wet"] = d["depth_in"] > 0
past = d.groupby("storm_id")["wet"].transform(lambda s: s.rolling(12, min_periods=1).max()).astype(bool).values
fl = (actual[both].max(1) > 0) | past[te[both] - 1]
names = list(z["names"]); M = {n: z["preds"][names.index(n)][j] for n in ("Persistence", "LSTM (standard)", "ANN-LSTM", "CNN-LSTM")}
M["Notebook: original-input LSTM"] = pred["Original-input LSTM"][both]; M["Notebook: feature-enhanced LSTM"] = pred["Feature-enhanced LSTM"][both]
y_cmp = actual[both][fl]
hz = pd.DataFrame([{"model": n, **{f"{5*(h+1)} min": r2_score(y_cmp[:, h], p[fl][:, h]) for h in range(HORIZON)}} for n, p in M.items()])

# is the notebook's original-input LSTM really different from the project's LSTM? resample whole storms
st_cmp = d["storm_id"].values[te[both] - 1][fl]; ust = np.unique(st_cmp); grp = {u: np.where(st_cmp == u)[0] for u in ust}
rgb = np.random.default_rng(0); r2f = lambda a, p: 1 - ((a - p) ** 2).sum() / max(((a - a.mean()) ** 2).sum(), 1e-9)
pairs = []
for other in ("LSTM (standard)", "ANN-LSTM", "Persistence"):
    for h in (0, 3, 7):
        A, Bp = M["Notebook: original-input LSTM"][fl][:, h], M[other][fl][:, h]; diff = []
        for _ in range(1000):
            i = np.concatenate([grp[u] for u in rgb.choice(ust, len(ust))])
            diff.append(r2f(y_cmp[i, h], A[i]) - r2f(y_cmp[i, h], Bp[i]))
        diff = np.array(diff)
        pairs.append({"notebook original-input LSTM vs": other, "minutes_ahead": 5 * (h + 1), "R2_difference": r2f(y_cmp[:, h], A) - r2f(y_cmp[:, h], Bp),
                      "lo": np.quantile(diff, .025), "hi": np.quantile(diff, .975), "p_boot": 2 * min((diff <= 0).mean(), (diff >= 0).mean())})
pairs = pd.DataFrame(pairs); pairs.to_csv(f"{OUT}/nb_vs_project_bootstrap.csv", index=False)

# ---------------------------------------------------------------- 3. rainfall conditions (thresholds from rainy training steps)
# NYC gauges record rain in tips of 0.254 mm (0.01 in) and about 75% of rainy steps are a single tip, so the 33rd
# and 66th percentiles of rainy steps are both 0.254 mm. The groups therefore follow the tips instead:
# light = one tip, moderate = 2-3 tips, heavy = 4 or more tips (1 mm or more in 5 minutes, over 12 mm per hour)
low_t, high_t = 0.3, 0.8
pc = d.loc[tr_rows, "rainfall_change"]; rapid_t = pc[pc > 0].quantile(0.90)
ar = np.stack([rain_raw[te + k] for k in range(HORIZON)], 1); ch = np.stack([d["rainfall_change"].values[te + k] for k in range(HORIZON)], 1)
conds = {"No rain": ar == 0, f"Light rain (0 to {low_t:.2f} mm)": (ar > 0) & (ar <= low_t),
         f"Moderate rain ({low_t:.2f} to {high_t:.2f} mm)": (ar > low_t) & (ar <= high_t), f"Heavy rain (over {high_t:.2f} mm)": ar > high_t,
         f"Rapidly increasing rain (rise of {rapid_t:.2f} mm or more)": (ar > 0) & (ch >= rapid_t)}
cr = []
for c, m in conds.items():
    m = m & fl_te[:, None]
    if m.sum() < 2:
        print("Insufficient observations for:", c); continue                # as in the notebook
    for name in models:
        cr.append({"Rainfall condition": c, "Model": name, "points": int(m.sum()), **calculate_metrics(actual[m], pred[name][m])})
condition_results = pd.DataFrame(cr)

# ---------------------------------------------------------------- 4. risk levels (notebook thresholds), 20 and 40 minutes ahead
def classify(depth): return np.where(depth < LOW_MAX, "Low Risk", np.where(depth <= MODERATE_MAX, "Moderate Risk", "High Risk"))
rk = []
for name in models:
    for h in (3, 7):
        a, p = classify(actual[:, h]), classify(pred[name][:, h])
        for lvl in ("Moderate Risk", "High Risk"):
            tp, fn, fp = ((a == lvl) & (p == lvl)).sum(), ((a == lvl) & (p != lvl)).sum(), ((a != lvl) & (p == lvl)).sum()
            rk.append({"Model": name, "minutes_ahead": 5 * (h + 1), "level": lvl, "actual_cases": int(tp + fn),
                       "caught": tp / max(tp + fn, 1), "false_share_of_alerts": fp / max(tp + fp, 1)})
risk = pd.DataFrame(rk)

# ---------------------------------------------------------------- save, print, figure
comparison.to_csv(f"{OUT}/nb_model_comparison_nyc.csv", index=False); hz.to_csv(f"{OUT}/nb_r2_by_horizon_vs_project.csv", index=False)
condition_results.to_csv(f"{OUT}/nb_rainfall_condition_analysis_nyc.csv", index=False); risk.to_csv(f"{OUT}/nb_risk_levels_nyc.csv", index=False)
pd.set_option("display.width", 220)
print("\n1. The notebook's comparison table, NYC test storms Jul-Oct 2026 (all 8 steps pooled; depth in inches):")
print(comparison.round(4).to_string(index=False))
print(f"\n2. R² by minutes ahead, flood-related test windows, identical windows for every model ({fl.sum():,} windows):")
print(hz.round(3).to_string(index=False))
print("\n2b. Notebook original-input LSTM minus the other model, R² (95% interval from resampling storms):")
print(pairs.round(3).to_string(index=False))
print("\n3. Rainfall conditions (flood-related windows):"); print(condition_results.round(3).to_string(index=False))
print("\n4. Risk levels (notebook thresholds, 0.10 m = 3.9 in, 0.30 m = 11.8 in):"); print(risk.round(3).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for _, r in hz.iterrows():
    nb = r["model"].startswith("Notebook")
    ax[0].plot(range(5, 45, 5), r.values[1:].astype(float), "o-" if nb else "s--", lw=2.2 if nb else 1.3, label=r["model"])
ax[0].axhline(0, color="k", lw=.8); ax[0].set_ylim(-1, 1); ax[0].set_xlabel("minutes ahead"); ax[0].set_ylabel("R² (flood-related test windows)")
ax[0].set_title("Notebook models on NYC data vs the project's models"); ax[0].legend(fontsize=8)
s = int(np.argmax(actual[:, 0] * (now_te < 1)))                             # a test window where a flood is about to start
lo = max(0, s - 30); idx = np.arange(lo, min(len(te), s + 60))
same = d["storm_id"].values[te[idx] - 1] == d["storm_id"].values[te[s] - 1]; idx = idx[same]
ax[1].plot(np.arange(len(idx)) * 5, actual[idx, 3], "k-", lw=2, label="measured depth")
for name in models: ax[1].plot(np.arange(len(idx)) * 5, pred[name][idx, 3], lw=1.6, label=f"{name}, 20 min ahead")
ax[1].set_xlabel("minutes into the storm"); ax[1].set_ylabel("depth (in)"); ax[1].set_title(f"One test storm: {d['storm_id'].values[te[s]-1][:30]}")
ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_notebook_on_nyc.png", dpi=150)
print(f"\nsaved nb_*.csv and fig_notebook_on_nyc.png ({time.time()-t_start:.0f}s)")
