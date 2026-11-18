"""Stage 16b: the notebook's two flood LSTMs (scripts/15a), made available to the website.

  Improved LSTM          : reads the last 75 minutes (15 steps) of rain and depth
  Feature-enhanced LSTM  : the same, plus the engineered rain features
                           rainfall_intensity (mm/h) = rain x 12, cumulative_15min / 30min / 60min, rainfall_change
Both forecast one step ahead and are rolled forward 8 steps (40 min) with the notebook's rainfall ANN, exactly as in 15a.

Writes
  backend/data/model_lstm_improved.json, model_lstm_features.json   weights + scalers, run in the browser (Model Lab)
  adds to every July-October 2026 storm file, per sensor:
    fe = {"imp": {"5": [...288], "15", "30", "40"}, "enh": {...}}   forecasts made at each 5-minute slot (New York time)
    rn = [...288]                                                    rain at the sensor's gauge (mm per 5 min)
Uses the saved models and test forecasts of 15a (no retraining).
Usage: python3 scripts/16b_feature_lstm_export.py <dir with nyc_depth_rain_5min.csv and nb_models/>
Then:  python3 backend/build_store.py"""
import os, sys, json, glob
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np, pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "processed")
DATA = os.path.join(ROOT, "backend", "data")
SAVE = os.path.join(OUT, "nb_models")
LOOKBACK, HORIZON, NY = 15, 8, "America/New_York"
SPLIT = {"train_end": "2025-12-31 23:59", "val_end": "2026-06-30 23:59"}
STEPS = {5: 0, 15: 2, 30: 5, 40: 7}

# ---------------------------------------------------------------- data, features and scalers exactly as in 15a
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
tr = d["period"] == "train"
rain_sc = MinMaxScaler().fit(d.loc[tr, ["rain_mm"]]); depth_sc = MinMaxScaler().fit(d.loc[tr, ["depth_in"]]); feat_sc = MinMaxScaler().fit(d.loc[tr, ENG])
start = d.groupby("storm_id").cumcount().values; size = d.groupby("storm_id")["time"].transform("size").values
origin = np.where((start >= LOOKBACK) & (start <= size - HORIZON))[0]
te = origin[d["period"].values[origin - 1] == "test"]
z = np.load(f"{SAVE}/test_forecasts.npz"); P = {"imp": z["original"], "enh": z["enhanced"]}
assert len(te) == len(P["imp"]), "test origins do not line up with 15a"
actual = np.stack([d["depth_in"].values[te + k] for k in range(HORIZON)], 1)

# ---------------------------------------------------------------- weights for the browser
L = lambda a: np.asarray(a, dtype=float).round(7).tolist()
ann = tf.keras.models.load_model(f"{SAVE}/rainfall_ann.keras")
ann_w = [{"W": L(l.get_weights()[0]), "b": L(l.get_weights()[1])} for l in ann.layers if l.get_weights()]
sc = lambda s: {"min": L(s.min_), "scale": L(s.scale_)}
fl = (actual.max(1) > 0) | (np.stack([d["depth_in"].values[te - 12 + k] for k in range(12)], 1).max(1) > 0)
def metrics(p):
    r2 = [1 - ((p[fl, h] - actual[fl, h]) ** 2).sum() / ((actual[fl, h] - actual[fl, h].mean()) ** 2).sum() for h in range(HORIZON)]
    rmse = [float(np.sqrt(((p[fl, h] - actual[fl, h]) ** 2).mean())) for h in range(HORIZON)]
    return [round(float(v), 4) for v in r2], [round(v, 4) for v in rmse]
for key, fn, name, feats in (("imp", "baseline_lstm", "Improved LSTM", ["rain_mm", "depth_in"]),
                             ("enh", "enhanced_lstm", "Feature-enhanced LSTM", ENG)):
    m = tf.keras.models.load_model(f"{SAVE}/{fn}.keras"); lw = [l for l in m.layers if l.get_weights()]
    k, rk, b = lw[0].get_weights(); d1, d2 = lw[1].get_weights(), lw[2].get_weights()
    r2, rmse = metrics(P[key])
    j = {"name": name, "kind": "nb", "enhanced": key == "enh", "lookback": LOOKBACK, "H": HORIZON, "features": feats,
         "scale_rain": sc(rain_sc), "scale_depth": sc(depth_sc), "scale_feat": sc(feat_sc) if key == "enh" else None,
         "ann": ann_w, "lstm": {"kernel": L(k), "recurrent": L(rk), "bias": L(b), "units": int(rk.shape[0])},
         "dense1": {"W": L(d1[0]), "b": L(d1[1])}, "dense2": {"W": L(d2[0]), "b": L(d2[1])},
         "test_period": "Jul-Oct 2026, flood-related windows", "test_r2": r2, "test_rmse_in": rmse}
    with open(f"{DATA}/model_lstm_{'improved' if key == 'imp' else 'features'}.json", "w") as f: json.dump(j, f, separators=(",", ":"))
    print(f"{name}: R2 " + " ".join(f"{5*(h+1)}m {r2[h]:.2f}" for h in range(HORIZON)))

# ---------------------------------------------------------------- into the storm files
local = lambda t: pd.DatetimeIndex(t).tz_localize("UTC").tz_convert(NY).tz_localize(None)
made = local(d["time"].values[te - 1]); sens = d["Sensor ID"].values[te - 1]
d["local"] = local(d["time"].values)
slots = pd.timedelta_range("5min", "24h", freq="5min")
r2f = lambda v: round(float(v), 2)
n = 0
for f in sorted(glob.glob(f"{DATA}/storm_*.json")):
    st = json.load(open(f))
    if not st.get("has_pred"): continue
    grid = pd.Timestamp(st["date"]) + slots
    for q in st["sensors"]:
        q.pop("fe", None); q.pop("rn", None)
        if not q.get("nc"): continue
        sel = np.where((sens == q["id"]) & (made >= grid[0]) & (made <= grid[-1]))[0]
        if not len(sel): continue
        pos = np.rint((made[sel] - grid[0]) / pd.Timedelta("5min")).astype(int)
        fe = {key: {str(m): [None] * 288 for m in STEPS} for key in P}
        for i, p in zip(sel, pos):
            for key in P:
                for m, h in STEPS.items(): fe[key][str(m)][p] = r2f(P[key][i, h])
        rr = d[(d["Sensor ID"] == q["id"]) & (d["local"] >= grid[0]) & (d["local"] <= grid[-1])]
        rn = [None] * 288
        for t, v in zip(rr["local"], rr["rain_mm"]): rn[int(round((t - grid[0]) / pd.Timedelta("5min")))] = r2f(v)
        q["fe"], q["rn"] = fe, rn; n += 1
    with open(f, "w") as fh: json.dump(st, fh, separators=(",", ":"))
print(f"added Improved / Feature-enhanced LSTM forecasts to {n} sensor-days")
