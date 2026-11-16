"""Stage 8b: the project's own models, on the same 5-minute data, split and metrics as 07c.
  ANN-LSTM : stage 1 = ANN (MLP) forecasts the next 40 min of rain at the gauge from the last hour,
             stage 2 = LSTM predicts depth 5..40 min ahead from past depth/rain + the ANN rain forecast.
             This is the reference paper's two-stage design, with nothing taken from the future.
  CNN-LSTM : 1-D convolutions pick out short patterns (bursts, sudden rises) in the last hour before the LSTM.
  CNN-LSTM + ANN rain forecast : both ideas together (proposed model).
Reference rows retrained here with the same seeds: Persistence, LSTM (standard), LSTM + perfect rain forecast,
CNN-LSTM + perfect rain forecast (upper bounds: given the rain that actually fell next).
Usage: python 07d_ann_cnn_lstm.py <out_dir with nyc_depth_rain_5min.csv and nyc_rain_5min_stations.csv>"""
import sys, time, json, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore"); torch.set_num_threads(2)
OUT = sys.argv[1]
L, H, FLOOD_IN, DRY_KEEP, SEEDS = 12, 8, 1.0, 0.15, [0, 1, 2]
SPLIT = {"train_end": "2025-12-31 23:59", "val_end": "2026-06-30 23:59"}
rng = np.random.default_rng(0)

# ============================================================ stage 1: ANN rain forecaster
g5 = pd.read_csv(f"{OUT}/nyc_rain_5min_stations.csv", parse_dates=["time_utc"])
g5.loc[g5["coverage"] < 0.6, "rain_mm"] = np.nan
def rain_feats(hist, t):                    # hist: (n, 12) last hour of 5-min rain (NaN-free), t: timestamps
    h = pd.DatetimeIndex(t).hour + pd.DatetimeIndex(t).minute / 60
    m = pd.DatetimeIndex(t).month
    return np.column_stack([hist, hist.sum(1), hist[:, -3:].sum(1), (hist > 0).sum(1),
                            np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24), np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12)]).astype(np.float32)
Xr, Yr, Tr = [], [], []
for s, g in g5.groupby("station"):
    v = g["rain_mm"].values; t = g["time_utc"].values
    W = np.lib.stride_tricks.sliding_window_view(v, L + H)
    ok = ~np.isnan(W).any(1)
    wet = np.nansum(W, 1) > 0
    pick = ok & (wet | (rng.random(len(W)) < 0.02))
    Xr.append(W[pick, :L]); Yr.append(W[pick, L:]); Tr.append(t[L - 1:][:len(W)][pick])
Xr, Yr, Tr = np.vstack(Xr), np.vstack(Yr), np.concatenate(Tr)
per_r = np.where(Tr <= np.datetime64(SPLIT["train_end"]), "train", np.where(Tr <= np.datetime64(SPLIT["val_end"]), "val", "test"))
Fr = rain_feats(Xr, Tr); mu, sd = Fr[per_r == "train"].mean(0), Fr[per_r == "train"].std(0) + 1e-6
t0 = time.time()
ann = MLPRegressor(hidden_layer_sizes=(64, 32), learning_rate_init=1e-3, batch_size=512, max_iter=80,
                   early_stopping=True, validation_fraction=0.1, n_iter_no_change=6, random_state=0)
ann.fit((Fr[per_r == "train"] - mu) / sd, Yr[per_r == "train"])
ann_rain = lambda hist, t: np.clip(ann.predict((rain_feats(hist, t) - mu) / sd), 0, None).astype(np.float32)
te_r = per_r == "test"; pa = ann_rain(Xr[te_r], Tr[te_r]); pp = np.repeat(Xr[te_r][:, -3:].mean(1, keepdims=True), H, 1)
wet = Yr[te_r].sum(1) > 0
rain_eval = pd.DataFrame([{"model": n, **{f"{5*(h+1)}min": np.sqrt(mean_squared_error(Yr[te_r][wet, h], p[wet, h])) for h in (0, 3, 7)},
                           "R2_40min_total": r2_score(Yr[te_r][wet].sum(1), p[wet].sum(1))}
                          for n, p in (("Persistence nowcast (last 15 min continues)", pp), ("ANN", pa), ("Zero rain", np.zeros_like(pa)))])
rain_eval.to_csv(f"{OUT}/rain5_ann_metrics_test.csv", index=False)
print(f"ANN rain forecaster trained on {int((per_r=='train').sum())} windows in {time.time()-t0:.0f}s")
print("Rain forecast, test period, wet windows - RMSE (mm per 5 min) and R2 of the 40-min total:\n", rain_eval.round(3).to_string(index=False))

# ============================================================ stage 2 data (identical to 07c)
d = pd.read_csv(f"{OUT}/nyc_depth_rain_5min.csv", parse_dates=["time"])
d["rain_missing"] = d["rain_mm"].isna().astype(np.float32); d["rain_mm"] = d["rain_mm"].fillna(0)
seq, fut, meta = [], [], []
for sid, g in d.groupby("storm_id", sort=False):
    if len(g) < L + H: continue
    dep, rn, ms, T = g["depth_in"].values, g["rain_mm"].values, g["rain_missing"].values, g["time"].values
    sensor = g["Sensor ID"].iloc[0]
    for t in range(L - 1, len(g) - H):
        seq.append(np.column_stack([dep[t-L+1:t+1], rn[t-L+1:t+1], ms[t-L+1:t+1]])); fut.append(rn[t+1:t+H+1])
        meta.append((sensor, sid, T[t]))
seq, fut = np.asarray(seq, np.float32), np.asarray(fut, np.float32)
y = np.ascontiguousarray(np.asarray([0]))  # placeholder, rebuilt below
meta = pd.DataFrame(meta, columns=["Sensor ID", "storm_id", "time"])
# targets
ys = []
for sid, g in d.groupby("storm_id", sort=False):
    if len(g) < L + H: continue
    dep = g["depth_in"].values
    ys += [dep[t+1:t+H+1] for t in range(L - 1, len(g) - H)]
y = np.asarray(ys, np.float32)
ann_fc = ann_rain(seq[:, :, 1], meta["time"].values)          # the ANN sees only the last hour
per = np.where(meta["time"] <= SPLIT["train_end"], "train", np.where(meta["time"] <= SPLIT["val_end"], "val", "test"))
floody = (y.max(1) > 0) | (seq[:, :, 0].max(1) > 0)
rng = np.random.default_rng(0); keep = floody | (rng.random(len(y)) < DRY_KEEP)   # same sampling as 07c
idx = {"train": np.where((per == "train") & keep)[0], "val": np.where((per == "val") & keep)[0], "test": np.where(per == "test")[0]}
for p in idx: print(f"{p}: {len(idx[p])} samples ({floody[idx[p]].mean():.1%} flood-related)")
tr = idx["train"]
sm, ss = seq[tr].reshape(-1, 3).mean(0), seq[tr].reshape(-1, 3).std(0) + 1e-6
fm, fs = fut[tr].mean(), fut[tr].std() + 1e-6
ysd = y[tr].std() + 1e-6; wts = np.where(floody, 5.0, 1 / DRY_KEEP).astype(np.float32)
T_s = torch.tensor((seq - sm) / ss); Y = torch.tensor(y / ysd); W = torch.tensor(wts)
FUT = {"none": torch.zeros(len(y), H), "perfect": torch.tensor((fut - fm) / fs), "ann": torch.tensor((ann_fc - fm) / fs)}

class Net(nn.Module):
    def __init__(self, rain_in, cnn, hid=64, ch=32):
        super().__init__(); self.rain_in, self.cnn = rain_in, cnn
        if cnn: self.conv = nn.Sequential(nn.Conv1d(3, ch, 3, padding=1), nn.ReLU(), nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(ch if cnn else 3, hid, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid + (H if rain_in else 0), 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, H))
    def forward(self, s, f):
        if self.cnn: s = self.conv(s.transpose(1, 2)).transpose(1, 2)
        _, (h, _) = self.lstm(s)
        return self.head(torch.cat([h[-1]] + ([f] if self.rain_in else []), 1))

CFG = {"LSTM (standard)": ("none", False), "LSTM + perfect rain forecast": ("perfect", False),
       "ANN-LSTM (LSTM + ANN rain forecast)": ("ann", False), "CNN-LSTM": ("none", True),
       "CNN-LSTM + ANN rain forecast": ("ann", True), "CNN-LSTM + perfect rain forecast": ("perfect", True)}
def train(rain, cnn, seed):
    torch.manual_seed(seed); net = Net(rain != "none", cnn); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    F = FUT[rain]; tri, vai = torch.tensor(tr), torch.tensor(idx["val"]); best, state, bad = 1e9, None, 0
    for ep in range(60):
        net.train(); perm = tri[torch.randperm(len(tri))]
        for b in range(0, len(perm), 512):
            j = perm[b:b+512]; loss = ((net(T_s[j], F[j]) - Y[j])**2 * W[j, None]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): vl = (((net(T_s[vai], F[vai]) - Y[vai])**2) * W[vai, None]).mean().item()
        if vl < best - 1e-5: best, state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8: break
    net.load_state_dict(state); net.eval(); te = torch.tensor(idx["test"])
    with torch.no_grad(): p = np.zeros_like(y); p[idx["test"]] = net(T_s[te], F[te]).numpy() * ysd
    return np.clip(p, 0, None), ep + 1, net
preds = {"Persistence": np.repeat(seq[:, -1:, 0], H, axis=1)}; runs_all = {}
for name, (rain, cnn) in CFG.items():
    t0 = time.time(); runs = [train(rain, cnn, s) for s in SEEDS]
    preds[name] = np.mean([r[0] for r in runs], 0); runs_all[name] = [r[0] for r in runs]
    print(f"{name}: {time.time()-t0:.0f}s, epochs {[r[1] for r in runs]}")
    if name == "CNN-LSTM + ANN rain forecast": torch.save(runs[0][2].state_dict(), f"{OUT}/cnn_lstm_ann.pt")

# ============================================================ evaluation (same metrics as 07c)
te = idx["test"]; yt = y[te]; flood_mask = (yt.max(1) > 0) | (seq[te, :, 0].max(1) > 0); onset = seq[te, -1, 0] == 0
rec = []
for name, p in preds.items():
    pt = p[te]
    for h in range(H):
        a, b = yt[:, h] >= FLOOD_IN, pt[:, h] >= FLOOD_IN; tp, fp, fn = (a & b).sum(), (~a & b).sum(), (a & ~b).sum()
        on = onset & a; m = flood_mask
        rec.append(dict(model=name, minutes_ahead=5*(h+1), RMSE_flood=np.sqrt(mean_squared_error(yt[m, h], pt[m, h])),
                        MAE_flood=mean_absolute_error(yt[m, h], pt[m, h]), R2_flood=r2_score(yt[m, h], pt[m, h]),
                        POD=tp/(tp+fn) if tp+fn else np.nan, FAR=fp/(tp+fp) if tp+fp else np.nan, CSI=tp/(tp+fp+fn) if tp+fp+fn else np.nan,
                        onset_caught=b[on].mean() if on.sum() else np.nan, onsets=int(on.sum())))
met = pd.DataFrame(rec); met.to_csv(f"{OUT}/depth5_own_models_metrics_test.csv", index=False)
order = list(preds)
def tab(col, f="{:.3f}"):
    return met.pivot_table(index="model", columns="minutes_ahead", values=col).loc[order][[5, 10, 20, 30, 40]].map(lambda v: f.format(v)).to_string()
print(f"\nTEST (Jul-Oct 2026): {len(te)} samples, {int(flood_mask.sum())} flood-related")
print("\nRMSE on flood samples (inches):\n" + tab("RMSE_flood")); print("\nR2 on flood samples:\n" + tab("R2_flood", "{:.2f}"))
print("\nCSI for depth >= 1 inch:\n" + tab("CSI", "{:.2f}")); print("\nFlood onsets caught:\n" + tab("onset_caught", "{:.0%}"))
# significance-style check: per-seed flood RMSE (all horizons)
sv = {n: [np.sqrt(mean_squared_error(yt[flood_mask], r[te][flood_mask])) for r in rs] for n, rs in runs_all.items()}
sv_df = pd.DataFrame({n: dict(mean=np.mean(v), std=np.std(v), runs=", ".join(f"{x:.3f}" for x in v)) for n, v in sv.items()}).T
sv_df.to_csv(f"{OUT}/depth5_own_models_seed_variability.csv"); print("\nPer-seed flood RMSE (all horizons):\n", sv_df.to_string())
# per-storm comparison: does CNN-LSTM+ANN beat LSTM (standard) storm by storm?
mt = meta.iloc[te].reset_index(drop=True); mt["m"] = flood_mask
e = {n: ((preds[n][te] - yt) ** 2).mean(1) for n in ("LSTM (standard)", "ANN-LSTM (LSTM + ANN rain forecast)", "CNN-LSTM", "CNN-LSTM + ANN rain forecast")}
st = pd.DataFrame(e)[flood_mask].assign(storm=mt.loc[flood_mask, "storm_id"].values).groupby("storm").mean()
st = st[mt[flood_mask].groupby("storm_id").size().reindex(st.index) >= 6]
wins = {n: f"{(st[n] < st['LSTM (standard)']).mean():.0%} of {len(st)} flooded test storms" for n in st.columns if n != "LSTM (standard)"}
print("\nBeats LSTM (standard) on:", wins); json.dump(wins, open(f"{OUT}/depth5_own_models_storm_wins.json", "w"), indent=2)

# predictions in the same format as 07c, for the dashboard
outp = meta.iloc[te].copy()
for h in (0, 3, 7): outp[f"obs_{5*(h+1)}min"] = yt[:, h]
for n in order:
    for h in (0, 3, 7): outp[f"{n}|{5*(h+1)}min"] = preds[n][te, h].round(3)
outp.to_csv(f"{OUT}/depth5_own_models_predictions_test.csv", index=False)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
for n in order:
    s = met[met.model == n]
    for a, c in zip(ax, ("RMSE_flood", "R2_flood", "onset_caught")): a.plot(s.minutes_ahead, s[c], marker="o", ms=3, label=n)
for a, t in zip(ax, ("RMSE on flood samples (in)", "R² on flood samples", "Flood onsets caught")): a.set(title=t, xlabel="minutes ahead"); a.grid(alpha=.3)
ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{OUT}/fig_depth5_own_models.png", dpi=150)
