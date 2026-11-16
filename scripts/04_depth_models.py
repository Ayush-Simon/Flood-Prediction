"""Phase 4: flood-depth prediction (Stage 2 of the paper), hourly, 1-3 h ahead.
Models: Persistence, XGBoost, standard LSTM, LSTM + rain forecast (paper-style),
Attention-LSTM + rain forecast, Attention-LSTM + rain forecast + terrain (proposed).
All models use the same chronological split (splits.json) and the same samples."""
import json, sys, time, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore"); torch.set_num_threads(2)

SW, FC, CELL, SPL, OUT = sys.argv[1:6]
L, H = 6, 3                 # 6 h of history -> predict depth 1, 2, 3 h ahead
FLOOD_IN = 1.0              # "flooding" for warning scores: depth >= 1 inch
W_FLOOD = 5.0               # loss weight for samples whose target contains flooding
SEEDS = [0, 1, 2]
sp = json.load(open(SPL))

# ---------------- data ----------------
d = pd.read_csv(SW, parse_dates=["time"])
d = d[~d["suspect_offline"]].copy()
cell = pd.read_csv(CELL).set_index("Sensor ID")["cell"]
fc = pd.read_csv(FC, parse_dates=["time"])
d["cell"] = d["Sensor ID"].map(cell)
d = d.merge(fc[["time", "cell"] + [f"rain_fc_h{h}" for h in range(1, H+1)]], on=["time", "cell"], how="left")
d[[f"rain_fc_h{h}" for h in range(1, H+1)]] = d[[f"rain_fc_h{h}" for h in range(1, H+1)]].fillna(0)
d = d.sort_values(["Sensor ID", "time"]).reset_index(drop=True)

STATIC = ["elevation_m", "slope_deg", "height_above_low_in"]
FCOLS = [f"rain_fc_h{h}" for h in range(1, H+1)]
seq, fcs, sta, y, meta = [], [], [], [], []
for (sid, stid), g in d.groupby(["Sensor ID", "storm_id"], sort=False):
    g = g.reset_index(drop=True)
    if len(g) < L + H: continue
    if (g["time"].diff().dropna() != pd.Timedelta(hours=1)).any(): continue
    dep, rn = g["depth_in"].values, g["rain_mm"].values
    F, S = g[FCOLS].values, g[STATIC].values[0]
    for t in range(L-1, len(g)-H):
        seq.append(np.column_stack([dep[t-L+1:t+1], rn[t-L+1:t+1]]))
        fcs.append(F[t]); sta.append(S); y.append(dep[t+1:t+H+1])
        meta.append((sid, stid, g["time"].iloc[t]))
seq, fcs, sta, y = map(lambda a: np.asarray(a, dtype=np.float32), (seq, fcs, sta, y))
meta = pd.DataFrame(meta, columns=["Sensor ID", "storm_id", "time"])
per = np.where(meta["time"] <= sp["train_end"], "train", np.where(meta["time"] <= sp["val_end"], "val", "test"))
idx = {p: np.where(per == p)[0] for p in ("train", "val", "test")}
flooded_target = (y.max(1) > 0)
for p in idx:
    print(f"{p}: {len(idx[p])} samples, {flooded_target[idx[p]].mean():.1%} with flooding ahead, "
          f"{meta.loc[idx[p], 'Sensor ID'].nunique()} sensors")

# scaling (train statistics only)
tr = idx["train"]
mu_s, sd_s = seq[tr].reshape(-1, 2).mean(0), seq[tr].reshape(-1, 2).std(0) + 1e-6
mu_f, sd_f = fcs[tr].mean(0), fcs[tr].std(0) + 1e-6
mu_t, sd_t = sta[tr].mean(0), sta[tr].std(0) + 1e-6
Xs = (seq - mu_s) / sd_s; Xf = (fcs - mu_f) / sd_f; Xt = (sta - mu_t) / sd_t
ysd = y[tr].std() + 1e-6
wts = np.where(flooded_target, W_FLOOD, 1.0).astype(np.float32)

preds = {}
# ---------------- 1. persistence ----------------
preds["Persistence"] = np.repeat(seq[:, -1:, 0], H, axis=1)

# ---------------- 2. XGBoost ----------------
t0 = time.time()
flat = np.column_stack([seq[:, :, 0], seq[:, :, 1], fcs, sta])
px = np.zeros_like(y)
for h in range(H):
    m = xgb.XGBRegressor(n_estimators=800, max_depth=5, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, n_jobs=2, early_stopping_rounds=40)
    m.fit(flat[tr], y[tr, h], sample_weight=wts[tr],
          eval_set=[(flat[idx["val"]], y[idx["val"], h])], verbose=False)
    px[:, h] = m.predict(flat)
preds["XGBoost"] = np.clip(px, 0, None); print(f"XGBoost {time.time()-t0:.0f}s")

# ---------------- deep models ----------------
class Net(nn.Module):
    def __init__(self, use_fc, use_static, attention, hid=64):
        super().__init__()
        self.use_fc, self.use_static, self.attention = use_fc, use_static, attention
        self.lstm = nn.LSTM(2, hid, batch_first=True)
        if attention: self.att = nn.Linear(hid, 1)
        extra = (H if use_fc else 0) + (len(STATIC) if use_static else 0)
        self.head = nn.Sequential(nn.Linear(hid + extra, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, H))
    def forward(self, s, f, t):
        out, (hN, _) = self.lstm(s)
        if self.attention:
            a = torch.softmax(self.att(out), dim=1)          # weight for each past hour
            z = (a * out).sum(1); self.last_att = a.detach()
        else:
            z = hN[-1]
        parts = [z] + ([f] if self.use_fc else []) + ([t] if self.use_static else [])
        return self.head(torch.cat(parts, 1))

def tens(i): return [torch.tensor(a[i]) for a in (Xs, Xf, Xt, y / ysd, wts)]
TR, VA = tens(idx["train"]), tens(idx["val"])
ALL = [torch.tensor(a) for a in (Xs, Xf, Xt)]

def train(cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net(**cfg); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best, best_state, bad = 1e9, None, 0
    n = len(TR[0])
    for ep in range(100):
        net.train(); perm = torch.randperm(n)
        for b in range(0, n, 256):
            j = perm[b:b+256]
            p = net(TR[0][j], TR[1][j], TR[2][j])
            loss = ((p - TR[3][j])**2 * TR[4][j][:, None]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = (((net(*VA[:3]) - VA[3])**2) * VA[4][:, None]).mean().item()
        if vl < best - 1e-5: best, best_state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 10: break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad(): out = net(*ALL).numpy() * ysd
    return np.clip(out, 0, None), ep + 1, net

CFG = {
    "LSTM (standard)":               dict(use_fc=False, use_static=False, attention=False),
    "LSTM + rain forecast":          dict(use_fc=True,  use_static=False, attention=False),
    "Att-LSTM + rain forecast":      dict(use_fc=True,  use_static=False, attention=True),
    "Att-LSTM + forecast + terrain": dict(use_fc=True,  use_static=True,  attention=True),
}
seed_preds, nets = {}, {}
for name, cfg in CFG.items():
    t0 = time.time(); runs = []
    for s in SEEDS:
        p, ep, net = train(cfg, s); runs.append(p)
    seed_preds[name] = runs; preds[name] = np.mean(runs, 0); nets[name] = net
    print(f"{name}: {time.time()-t0:.0f}s ({len(SEEDS)} seeds, last run {ep} epochs)")

# ---------------- evaluation ----------------
def warn_scores(yt, yp):
    a, b = yt >= FLOOD_IN, yp >= FLOOD_IN
    tp, fp, fn = (a & b).sum(), (~a & b).sum(), (a & ~b).sum()
    pod = tp / (tp + fn) if tp + fn else np.nan          # share of real floods caught
    far = fp / (tp + fp) if tp + fp else np.nan          # share of warnings that were false
    csi = tp / (tp + fp + fn) if tp + fp + fn else np.nan
    return pod, far, csi
te = idx["test"]
flood_mask = (y[te].max(1) > 0) | (seq[te, -1, 0] > 0)
rec = []
for name, p in preds.items():
    for subset, m in (("all storm hours", np.ones(len(te), bool)), ("flood hours", flood_mask)):
        for h in range(H):
            yt, yp = y[te][m, h], p[te][m, h]
            pod, far, csi = warn_scores(yt, yp)
            rec.append(dict(model=name, subset=subset, horizon_h=h+1,
                            MAE=mean_absolute_error(yt, yp), RMSE=np.sqrt(mean_squared_error(yt, yp)),
                            R2=r2_score(yt, yp), POD=pod, FAR=far, CSI=csi))
met = pd.DataFrame(rec); met.to_csv(f"{OUT}/depth_metrics_test.csv", index=False)

# seed variability (RMSE on flood hours, mean over horizons)
var = []
for name, runs in seed_preds.items():
    r = [np.sqrt(mean_squared_error(y[te][flood_mask], p[te][flood_mask])) for p in runs]
    var.append(dict(model=name, RMSE_mean=np.mean(r), RMSE_std=np.std(r)))
pd.DataFrame(var).to_csv(f"{OUT}/depth_seed_variability.csv", index=False)

order = list(preds)
print(f"\nTEST, flood hours ({flood_mask.sum()} samples) - RMSE (inches) by horizon")
print(met[met.subset == "flood hours"].pivot_table(index="model", columns="horizon_h", values="RMSE").loc[order].round(3).to_string())
print("\nTEST, flood hours - R2 by horizon")
print(met[met.subset == "flood hours"].pivot_table(index="model", columns="horizon_h", values="R2").loc[order].round(3).to_string())
print("\nTEST, all storm hours - averaged over horizons")
print(met[met.subset == "all storm hours"].groupby("model")[["MAE", "RMSE", "R2", "POD", "FAR", "CSI"]].mean().loc[order].round(3).to_string())
print("\nSeed variability (flood hours RMSE):"); print(pd.DataFrame(var).round(3).to_string(index=False))

# ---------------- save predictions ----------------
outp = meta.copy(); outp["period"] = per
for h in range(H): outp[f"obs_h{h+1}"] = y[:, h]
for name, p in preds.items():
    for h in range(H): outp[f"{name}|h{h+1}"] = p[:, h].round(3)
outp[outp.period == "test"].to_csv(f"{OUT}/depth_predictions_test.csv", index=False)
torch.save(nets["Att-LSTM + forecast + terrain"].state_dict(), f"{OUT}/att_lstm_terrain.pt")
json.dump({"L": L, "H": H, "static": STATIC, "mu_s": mu_s.tolist(), "sd_s": sd_s.tolist(), "mu_f": mu_f.tolist(),
           "sd_f": sd_f.tolist(), "mu_t": mu_t.tolist(), "sd_t": sd_t.tolist(), "ysd": float(ysd)},
          open(f"{OUT}/att_lstm_terrain_scaling.json", "w"), indent=2)

# ---------------- figures ----------------
fig, ax = plt.subplots(figsize=(7, 4))
for name in order:
    s = met[(met.model == name) & (met.subset == "flood hours")]
    ax.plot(s.horizon_h, s.RMSE, marker="o", label=name)
ax.set_xticks([1, 2, 3]); ax.set_xlabel("Forecast horizon (hours)"); ax.set_ylabel("RMSE (inches)")
ax.set_title("Flood-depth error on test flood hours"); ax.legend(fontsize=7); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_depth_rmse_by_horizon.png", dpi=150)

tp = outp[outp.period == "test"]
top = tp.groupby("storm_id")["obs_h1"].max().sort_values(ascending=False).index[:4]
fig, axes = plt.subplots(2, 2, figsize=(12, 7)); axes = axes.ravel()
for ax, s in zip(axes, top):
    g = tp[tp.storm_id == s].sort_values("time"); tt = g["time"] + pd.Timedelta(hours=1)
    ax.plot(tt, g["obs_h1"], "k-", lw=2, label="observed")
    for name in ("Persistence", "LSTM (standard)", "LSTM + rain forecast", "Att-LSTM + forecast + terrain"):
        ax.plot(tt, g[f"{name}|h1"], label=name)
    ax.set_title(f"{g['Sensor ID'].iloc[0]}  {g['time'].iloc[0]:%Y-%m-%d}", fontsize=8)
    ax.set_ylabel("depth (in)"); ax.tick_params(axis="x", labelsize=7)
axes[0].legend(fontsize=7); fig.suptitle("Test storms: 1-hour-ahead depth predictions"); fig.tight_layout()
fig.savefig(f"{OUT}/fig_depth_test_storms.png", dpi=150)
print("\nSaved to", OUT)
