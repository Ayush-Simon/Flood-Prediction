"""Stage 16a: the main model (ANN rain forecast -> CNN-LSTM, the one the live site runs) on every past test storm.

For every 5-minute step of every July-October 2026 storm, the model reads the last hour of depth and rain at each
sensor (exactly the inputs it was tested on) and forecasts the depth 5, 15, 30 and 40 minutes ahead. The results are
added to the past-storm files so the website can show "it is raining now: here is the flood in 5, 15, 30, 40 minutes"
next to what really happened. Also writes model_live.json so the Model Lab can run the same model in the browser.

Usage: python3 scripts/16a_rain_now_past.py <dir with nyc_depth_rain_5min.csv>
Then:  python3 backend/build_store.py      (loads the updated files into the store)"""
import sys, os, json, glob
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
from flood_models import DepthModel

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "processed")
DATA = os.path.join(ROOT, "backend", "data")
L, H, NY = 12, 8, "America/New_York"
STEPS = {5: 0, 15: 2, 30: 5, 40: 7}                       # minutes ahead -> output index
TEST_FROM = pd.Timestamp("2026-07-01")                    # test period: July-October 2026 (UTC)

J = json.load(open(os.path.join(ROOT, "models", "live_cnn_lstm_ann.json")))
M = DepthModel(J)

# ---------------------------------------------------------------- windows, built exactly as in training (07d)
d = pd.read_csv(f"{OUT}/nyc_depth_rain_5min.csv", parse_dates=["time"])
d["rain_missing"] = d["rain_mm"].isna().astype(np.float32); d["rain_mm"] = d["rain_mm"].fillna(0)
seq, y, meta = [], [], []
for sid, g in d.groupby("storm_id", sort=False):
    if len(g) < L + H or g["time"].iloc[-1] < TEST_FROM: continue
    dep, rn, ms, T = g["depth_in"].values, g["rain_mm"].values, g["rain_missing"].values, g["time"].values
    for t in range(L - 1, len(g) - H):
        if T[t] < TEST_FROM: continue
        seq.append(np.column_stack([dep[t-L+1:t+1], rn[t-L+1:t+1], ms[t-L+1:t+1]])); y.append(dep[t+1:t+H+1])
        meta.append((g["Sensor ID"].iloc[0], T[t]))
seq, y = np.asarray(seq, np.float64), np.asarray(y, np.float64)
meta = pd.DataFrame(meta, columns=["sensor", "time"])
pred, rain = [], []
for b in range(0, len(seq), 4096):
    p, r = M.predict(seq[b:b+4096], meta["time"].values[b:b+4096], return_rain=True); pred.append(p); rain.append(r)
pred, rain = np.vstack(pred), np.vstack(rain)

# ---------------------------------------------------------------- check: same accuracy as reported (flood-related windows)
fl = (y.max(1) > 0) | (seq[:, :, 0].max(1) > 0)
r2 = [1 - ((pred[fl, h] - y[fl, h]) ** 2).sum() / ((y[fl, h] - y[fl, h].mean()) ** 2).sum() for h in range(H)]
rmse = [float(np.sqrt(((pred[fl, h] - y[fl, h]) ** 2).mean())) for h in range(H)]
print(f"{len(seq):,} test windows ({fl.sum():,} flood-related)")
print("R2   :", " ".join(f"{5*(h+1)}min {r2[h]:.2f}" for h in range(H)))
print("RMSE :", " ".join(f"{5*(h+1)}min {rmse[h]:.2f}" for h in range(H)))

# ---------------------------------------------------------------- into the storm files (New York time, 288 slots per day)
meta["made"] = meta["time"].dt.tz_localize("UTC").dt.tz_convert(NY).dt.tz_localize(None)
slots = pd.timedelta_range("5min", "24h", freq="5min")
r1 = lambda v: None if v is None or np.isnan(v) else round(float(v), 2)
n_files = n_sens = 0
for f in sorted(glob.glob(f"{DATA}/storm_*.json")):
    st = json.load(open(f))
    if not st.get("has_pred"): continue
    day = pd.Timestamp(st["date"]); grid = day + slots
    for q in st["sensors"]:
        q.pop("nc", None); q.pop("ncr", None)
        if not q.get("pred"): continue
        m = meta.index[(meta["sensor"] == q["id"]) & (meta["made"] >= grid[0]) & (meta["made"] <= grid[-1])]
        if not len(m): continue
        pos = ((meta.loc[m, "made"] - grid[0]) / pd.Timedelta("5min")).round().astype(int).values
        nc = {str(k): [None] * 288 for k in STEPS}; ncr = {str(k): [None] * 288 for k in STEPS}
        for i, p in zip(m, pos):                         # indexed by the time the forecast was MADE ("now")
            for k, h in STEPS.items():
                nc[str(k)][p] = r1(pred[i, h]); ncr[str(k)][p] = r1(rain[i, :h + 1].sum())
        q["nc"], q["ncr"] = nc, ncr; n_sens += 1
    st["nc_model"] = "CNN-LSTM + ANN rain forecast (main model)"
    with open(f, "w") as fh: json.dump(st, fh, separators=(",", ":"))
    n_files += 1
print(f"added 5/15/30/40-minute forecasts to {n_files} storm files ({n_sens} sensor-days)")

# ---------------------------------------------------------------- the same model for the Model Lab (browser)
lab = {"name": "Main model (CNN-LSTM + ANN rain)", "cnn": True, "rain_forecast": True, "L": L, "H": H, "step_min": 5,
       "inputs": ["depth_in", "rain_mm", "rain_missing"], **{k: J[k] for k in ("seq_mean", "seq_std", "fut_mean", "fut_std", "y_sd", "ann")},
       "test_period": "Jul-Oct 2026, flood-related windows", "test_r2": [round(v, 4) for v in r2], "test_rmse_in": [round(v, 4) for v in rmse],
       "weights": J["weights"], **({"members": J["members"]} if "members" in J else {})}
with open(f"{DATA}/model_live.json", "w") as fh: json.dump(lab, fh, separators=(",", ":"))
print("wrote backend/data/model_live.json")
