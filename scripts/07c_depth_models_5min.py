"""Stage 8: 5-minute flood-depth prediction, 5..40 minutes ahead (like the reference paper).
Usage: python 07c_depth_models_5min.py <out_dir with nyc_depth_rain_5min.csv>
Same chronological split as before: train <= 2025-12-31, validation 2026 Jan-Jun, test 2026 Jul-Oct.

'Future rain' inputs: the paper feeds forecast rain into the LSTM. Here two versions are tested:
  - perfect rain forecast = the rain that actually fell next (an upper bound on what any forecast can give)
  - nowcast              = average rain of the last 15 minutes, assumed to continue (realistic, no extra data)"""
import sys, json, time, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore"); torch.set_num_threads(2)
OUT = sys.argv[1]
L, H = 12, 8                        # 1 hour of history (12 x 5 min) -> 40 minutes ahead (8 x 5 min)
FLOOD_IN = 1.0
DRY_KEEP = 0.15                     # share of 'dry' training samples kept (all flood-related ones are kept)
SEEDS = [0, 1, 2]
SPLIT = {"train_end": "2025-12-31 23:59", "val_end": "2026-06-30 23:59"}
STATIC = ["elevation_m", "slope_deg", "height_above_low_in"]
rng = np.random.default_rng(0)

d = pd.read_csv(f"{OUT}/nyc_depth_rain_5min.csv", parse_dates=["time"])
d["rain_missing"] = d["rain_mm"].isna().astype(np.float32); d["rain_mm"] = d["rain_mm"].fillna(0)

seq, fut, now, sta, y, meta = [], [], [], [], [], []
for sid, g in d.groupby("storm_id", sort=False):
    if len(g) < L + H: continue
    dep, rn, ms = g["depth_in"].values, g["rain_mm"].values, g["rain_missing"].values
    S = g[STATIC].values[0]; T = g["time"].values; sensor = g["Sensor ID"].iloc[0]
    for t in range(L - 1, len(g) - H):
        seq.append(np.column_stack([dep[t-L+1:t+1], rn[t-L+1:t+1], ms[t-L+1:t+1]]))
        fut.append(rn[t+1:t+H+1]); now.append(np.full(H, rn[t-2:t+1].mean()))
        sta.append(S); y.append(dep[t+1:t+H+1]); meta.append((sensor, sid, T[t]))
seq, fut, now, sta, y = (np.asarray(a, np.float32) for a in (seq, fut, now, sta, y))
meta = pd.DataFrame(meta, columns=["Sensor ID", "storm_id", "time"])
per = np.where(meta["time"] <= SPLIT["train_end"], "train", np.where(meta["time"] <= SPLIT["val_end"], "val", "test"))
floody = (y.max(1) > 0) | (seq[:, :, 0].max(1) > 0)
keep = floody | (rng.random(len(y)) < DRY_KEEP)
w_dry = 1 / DRY_KEEP
idx = {"train": np.where((per == "train") & keep)[0], "val": np.where((per == "val") & keep)[0], "test": np.where(per == "test")[0]}
for p in idx:
    print(f"{p}: {len(idx[p])} samples ({floody[idx[p]].mean():.1%} flood-related), {meta.loc[idx[p], 'Sensor ID'].nunique()} sensors")

tr = idx["train"]
def zs(a, axis=0): m, s = a[tr].reshape(-1, a.shape[-1]).mean(0), a[tr].reshape(-1, a.shape[-1]).std(0) + 1e-6; return (a - m) / s
Xs = zs(seq); Xfut = zs(fut[..., None])[..., 0]; Xnow = (now - fut[tr].mean()) / (fut[tr].std() + 1e-6); Xt = zs(sta)
ysd = y[tr].std() + 1e-6
wts = np.where(floody, 5.0, w_dry).astype(np.float32)
preds = {"Persistence": np.repeat(seq[:, -1:, 0], H, axis=1)}

# ---------------- XGBoost ----------------
def flat(fr): return np.column_stack([seq[:, :, 0], seq[:, :, 1], seq[:, :, 1].sum(1, keepdims=True), sta] + ([fr] if fr is not None else []))
for name, fr in (("XGBoost (no future rain)", None), ("XGBoost + perfect rain forecast", fut)):
    t0 = time.time(); F = flat(fr); p = np.zeros_like(y)
    for h in range(H):
        m = xgb.XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             n_jobs=2, early_stopping_rounds=40, tree_method="hist")
        m.fit(F[tr], y[tr, h], sample_weight=wts[tr], eval_set=[(F[idx["val"]], y[idx["val"], h])], verbose=False)
        p[:, h] = m.predict(F)
    preds[name] = np.clip(p, 0, None); print(f"{name}: {time.time()-t0:.0f}s")

# ---------------- deep models ----------------
class Net(nn.Module):
    def __init__(self, rain_in, use_static, attention, hid=64):
        super().__init__()
        self.rain_in, self.use_static, self.attention = rain_in, use_static, attention
        self.lstm = nn.LSTM(3, hid, batch_first=True)
        if attention: self.att = nn.Linear(hid, 1)
        extra = (H if rain_in else 0) + (len(STATIC) if use_static else 0)
        self.head = nn.Sequential(nn.Linear(hid + extra, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, H))
    def forward(self, s, f, t):
        out, (hN, _) = self.lstm(s)
        z = (torch.softmax(self.att(out), 1) * out).sum(1) if self.attention else hN[-1]
        return self.head(torch.cat([z] + ([f] if self.rain_in else []) + ([t] if self.use_static else []), 1))

CFG = {
    "LSTM (standard)":                         dict(rain="none",    static=False, att=False),
    "LSTM + perfect rain forecast":            dict(rain="perfect", static=False, att=False),
    "Att-LSTM + perfect rain forecast":        dict(rain="perfect", static=False, att=True),
    "Att-LSTM + perfect forecast + terrain":   dict(rain="perfect", static=True,  att=True),
    "Att-LSTM + nowcast + terrain (realistic)":dict(rain="nowcast", static=True,  att=True),
}
Y = torch.tensor(y / ysd); W = torch.tensor(wts)
T_s, T_t = torch.tensor(Xs), torch.tensor(Xt.astype(np.float32))
T_f = {"perfect": torch.tensor(Xfut.astype(np.float32)), "nowcast": torch.tensor(Xnow.astype(np.float32)),
       "none": torch.zeros(len(y), H)}
def train(cfg, seed):
    torch.manual_seed(seed)
    net = Net(cfg["rain"] != "none", cfg["static"], cfg["att"]); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    F = T_f[cfg["rain"]]; tri = torch.tensor(tr); vai = torch.tensor(idx["val"])
    best, state, bad = 1e9, None, 0
    for ep in range(60):
        net.train(); perm = tri[torch.randperm(len(tri))]
        for b in range(0, len(perm), 512):
            j = perm[b:b+512]
            loss = ((net(T_s[j], F[j], T_t[j]) - Y[j])**2 * W[j, None]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): vl = (((net(T_s[vai], F[vai], T_t[vai]) - Y[vai])**2) * W[vai, None]).mean().item()
        if vl < best - 1e-5: best, state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8: break
    net.load_state_dict(state); net.eval()
    te = torch.tensor(idx["test"])
    with torch.no_grad(): p = np.zeros_like(y); p[idx["test"]] = net(T_s[te], F[te], T_t[te]).numpy() * ysd
    return np.clip(p, 0, None), ep + 1
seed_rmse = {}
for name, cfg in CFG.items():
    t0 = time.time(); runs = [train(cfg, s) for s in SEEDS]
    preds[name] = np.mean([r[0] for r in runs], 0)
    print(f"{name}: {time.time()-t0:.0f}s, epochs {[r[1] for r in runs]}")
    seed_rmse[name] = [r[0] for r in runs]

# ---------------- evaluation on the test period ----------------
te = idx["test"]; yt = y[te]
flood_mask = (yt.max(1) > 0) | (seq[te, :, 0].max(1) > 0)
onset = (seq[te, -1, 0] == 0)                                    # dry right now ...
rec = []
for name, p in preds.items():
    pt = p[te]
    for h in range(H):
        a, b = yt[:, h] >= FLOOD_IN, pt[:, h] >= FLOOD_IN
        tp, fp, fn = (a & b).sum(), (~a & b).sum(), (a & ~b).sum()
        on = onset & a                                           # ... but flooded h steps later
        m = flood_mask
        rec.append(dict(model=name, minutes_ahead=5*(h+1),
                        RMSE_flood=np.sqrt(mean_squared_error(yt[m, h], pt[m, h])), MAE_flood=mean_absolute_error(yt[m, h], pt[m, h]),
                        R2_flood=r2_score(yt[m, h], pt[m, h]),
                        POD=tp/(tp+fn) if tp+fn else np.nan, FAR=fp/(tp+fp) if tp+fp else np.nan, CSI=tp/(tp+fp+fn) if tp+fp+fn else np.nan,
                        onset_caught=(b[on]).mean() if on.sum() else np.nan, onsets=int(on.sum())))
met = pd.DataFrame(rec); met.to_csv(f"{OUT}/depth5_metrics_test.csv", index=False)
order = list(preds)
def tab(col, f="{:.3f}"):
    t = met.pivot_table(index="model", columns="minutes_ahead", values=col).loc[order][[5, 10, 20, 30, 40]]
    return t.map(lambda v: f.format(v)).to_string()
print(f"\nTEST (Jul-Oct 2026): {len(te)} samples, {int(flood_mask.sum())} flood-related")
print("\nRMSE on flood samples (inches):\n" + tab("RMSE_flood"))
print("\nR2 on flood samples:\n" + tab("R2_flood", "{:.2f}"))
print("\nCSI for 'depth >= 1 inch' (1 = perfect warnings):\n" + tab("CSI", "{:.2f}"))
print("\nFlood ONSETS caught (dry now, >= 1 inch later):\n" + tab("onset_caught", "{:.0%}"))
print("number of onsets per horizon:", met[met.model == "Persistence"].set_index("minutes_ahead")["onsets"].to_dict())
sv = {n: [np.sqrt(mean_squared_error(yt[flood_mask], r[te][flood_mask])) for r in rs] for n, rs in seed_rmse.items()}
pd.DataFrame({n: dict(mean=np.mean(v), std=np.std(v)) for n, v in sv.items()}).T.round(3).to_csv(f"{OUT}/depth5_seed_variability.csv")
print("\nSeed spread (flood-sample RMSE over all horizons):", {n: f"{np.mean(v):.3f} ± {np.std(v):.3f}" for n, v in sv.items()})

# ---------------- outputs ----------------
outp = meta.iloc[te].copy()
for h in (0, 3, 7): outp[f"obs_{5*(h+1)}min"] = yt[:, h]
for n in order:
    for h in (0, 3, 7): outp[f"{n}|{5*(h+1)}min"] = preds[n][te, h].round(3)
outp.to_csv(f"{OUT}/depth5_predictions_test.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
for n in order:
    s = met[met.model == n]; ax[0].plot(s.minutes_ahead, s.RMSE_flood, marker="o", ms=3, label=n)
    ax[1].plot(s.minutes_ahead, s.onset_caught, marker="o", ms=3, label=n)
ax[0].set(xlabel="minutes ahead", ylabel="RMSE (inches)", title="Depth error on test flood samples")
ax[1].set(xlabel="minutes ahead", ylabel="share caught", title="Flood onsets caught (dry now, >= 1 in later)")
ax[0].legend(fontsize=7); [a.grid(alpha=.3) for a in ax]; fig.tight_layout(); fig.savefig(f"{OUT}/fig_depth5_metrics.png", dpi=150)

pk = outp.groupby("storm_id")["obs_5min"].max().sort_values(ascending=False).index[:4]
fig, axes = plt.subplots(2, 2, figsize=(13, 7)); axes = axes.ravel()
show = ["Persistence", "LSTM (standard)", "Att-LSTM + perfect forecast + terrain", "Att-LSTM + nowcast + terrain (realistic)"]
for ax, s in zip(axes, pk):
    g = outp[outp.storm_id == s].sort_values("time"); g = g[(g["obs_20min"] > 0) | (g["obs_5min"] > 0)]
    lo, hi = g["time"].min() - pd.Timedelta(minutes=45), g["time"].max() + pd.Timedelta(minutes=30)
    g = outp[(outp.storm_id == s) & outp["time"].between(lo, hi)].sort_values("time")
    tt = g["time"] + pd.Timedelta(minutes=20)
    ax.plot(tt, g["obs_20min"], "k-", lw=2.2, label="observed")
    for n in show: ax.plot(tt, g[f"{n}|20min"], label=n)
    ax.set_title(f"{g['Sensor ID'].iloc[0]}  {g['time'].iloc[0]:%Y-%m-%d}  (20 min ahead)", fontsize=8)
    ax.set_ylabel("depth (in)"); ax.tick_params(axis="x", labelsize=7)
axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{OUT}/fig_depth5_test_storms.png", dpi=150)
json.dump({"L_steps": L, "H_steps": H, "step_minutes": 5, "split": SPLIT, "flood_threshold_in": FLOOD_IN}, open(f"{OUT}/depth5_setup.json", "w"), indent=2)
