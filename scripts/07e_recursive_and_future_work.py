"""Stage 8c: the reference paper's own design, plus its future-work experiments.

A. Single-step recursive LSTM (paper, eq. 7): one network predicts depth 5 min ahead from the last hour plus the
   forecast rain for that next step; its prediction and the forecast rain are appended to the window and it is run
   again, 8 times (40 min). One recursive model per 5-minute rain forecaster, as in the paper's Table 1:
   AR-LSTM (linear autoregressive, ARIMA-type), XGBoost-LSTM, ANN-LSTM, and perfect rain (upper bound).
B. Future work 1: how rain-forecast accuracy drives depth accuracy - forecasts blended between ANN and perfect.
C. Future work 4: accuracy on extreme storms (deepest test floods) and in the first 15 minutes.
Same data, split and sampling as 07c/07d.  Usage: python 07e_recursive_and_future_work.py <out_dir>"""
import sys, time, json, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore"); torch.set_num_threads(2)
OUT = sys.argv[1]
src = open(__file__.replace("07e_recursive_and_future_work.py", "07d_ann_cnn_lstm.py")).read().split("CFG = {")[0]
ns = {"__file__": "07d"}; sys.argv = ["07d", OUT]; exec(compile(src, "07d_ann_cnn_lstm.py", "exec"), ns)
L, H, SPLIT = ns["L"], ns["H"], ns["SPLIT"]
seq, fut, y, meta, per, idx, floody = ns["seq"], ns["fut"], ns["y"], ns["meta"], ns["per"], ns["idx"], ns["floody"]
Xr, Yr, Tr, per_r, rain_feats, ann_rain = ns["Xr"], ns["Yr"], ns["Tr"], ns["per_r"], ns["rain_feats"], ns["ann_rain"]
tr, te = idx["train"], idx["test"]
SEEDS = [0, 1, 2]

# ============================================================ A1. three 5-minute rain forecasters (+ perfect)
t0 = time.time(); trm = per_r == "train"
ar = LinearRegression().fit(Xr[trm], Yr[trm])                                  # AR(12): linear autoregression on the last hour
F = rain_feats(Xr, Tr)
xgbs = [xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, n_jobs=2).fit(F[trm], Yr[trm][:, h]) for h in range(H)]
def rain_fc(name, hist, t):
    if name == "AR": return np.clip(ar.predict(hist), 0, None)
    if name == "XGBoost": f = rain_feats(hist, t); return np.clip(np.column_stack([m.predict(f) for m in xgbs]), 0, None)
    return ann_rain(hist, t)
times = meta["time"].values
FC = {n: rain_fc(n, seq[:, :, 1], times).astype(np.float32) for n in ("AR", "XGBoost", "ANN")}
FC["Perfect"] = fut
wet = fut[te].sum(1) > 0
rain_acc = {n: dict(RMSE=float(np.sqrt(mean_squared_error(fut[te][wet], FC[n][te][wet]))),
                    R2_total=float(r2_score(fut[te][wet].sum(1), FC[n][te][wet].sum(1)))) for n in ("AR", "XGBoost", "ANN")}
print(f"rain forecasters ready ({time.time()-t0:.0f}s):", {k: {a: round(b, 3) for a, b in v.items()} for k, v in rain_acc.items()})

# ============================================================ A2. one-step LSTM, run recursively
sm, ss = seq[tr].reshape(-1, 3).mean(0), seq[tr].reshape(-1, 3).std(0) + 1e-6
rm, rs = float(fut[tr].mean()), float(fut[tr].std() + 1e-6); ysd = float(y[tr].std() + 1e-6)
wts = torch.tensor(np.where(floody, 5.0, 1 / ns["DRY_KEEP"]).astype(np.float32))
class OneStep(nn.Module):
    def __init__(self, hid=64):
        super().__init__(); self.lstm = nn.LSTM(3, hid, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid + 1, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, s, r): _, (h, _) = self.lstm(s); return self.head(torch.cat([h[-1], r[:, None]], 1))[:, 0]
S_all = torch.tensor((seq - sm) / ss)
def train_onestep(rain_next, seed):
    """rain_next: (n,) forecast rain for step t+1 used as input"""
    torch.manual_seed(seed); net = OneStep(); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    R = torch.tensor((rain_next - rm) / rs); Y1 = torch.tensor(y[:, 0] / ysd)
    tri, vai = torch.tensor(tr), torch.tensor(idx["val"]); best, state, bad = 1e9, None, 0
    for ep in range(60):
        net.train(); perm = tri[torch.randperm(len(tri))]
        for b in range(0, len(perm), 512):
            j = perm[b:b+512]; loss = ((net(S_all[j], R[j]) - Y1[j])**2 * wts[j]).mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): vl = (((net(S_all[vai], R[vai]) - Y1[vai])**2) * wts[vai]).mean().item()
        if vl < best - 1e-5: best, state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8: break
    net.load_state_dict(state); net.eval(); return net, ep + 1
def run_recursive(net, rain8, rows):
    """paper eq. 7: append each predicted depth and the forecast rain to the window, slide, predict again"""
    w = seq[rows].copy(); out = np.zeros((len(rows), H), np.float32)
    with torch.no_grad():
        for k in range(H):
            d = np.clip(net(torch.tensor((w - sm) / ss), torch.tensor((rain8[rows, k] - rm) / rs)).numpy() * ysd, 0, None)
            out[:, k] = d
            w = np.concatenate([w[:, 1:], np.stack([d, rain8[rows, k], np.zeros(len(rows), np.float32)], 1)[:, None]], 1)
    return out

preds, seedruns = {"Persistence": np.repeat(seq[te][:, -1:, 0], H, axis=1)}, {}
# standard LSTM, direct multi-step (the reference model of 07c/07d), retrained here with the same seeds
Net = ns["Net"]; T_s, Y, W, FUT = ns["T_s"], ns["Y"], ns["W"], ns["FUT"]
def train_direct(seed):
    torch.manual_seed(seed); net = Net(False, False); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Fz = FUT["none"]; tri, vai = torch.tensor(tr), torch.tensor(idx["val"]); best, state, bad = 1e9, None, 0
    for ep in range(60):
        net.train(); perm = tri[torch.randperm(len(tri))]
        for b in range(0, len(perm), 512):
            j = perm[b:b+512]; loss = ((net(T_s[j], Fz[j]) - Y[j])**2 * W[j, None]).mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): vl = (((net(T_s[vai], Fz[vai]) - Y[vai])**2) * W[vai, None]).mean().item()
        if vl < best - 1e-5: best, state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8: break
    net.load_state_dict(state); net.eval()
    with torch.no_grad(): return np.clip(net(T_s[torch.tensor(te)], Fz[torch.tensor(te)]).numpy() * ns["ysd"], 0, None)
t0 = time.time(); runs = [train_direct(s) for s in SEEDS]
preds["LSTM (standard, direct)"] = np.mean(runs, 0); seedruns["LSTM (standard, direct)"] = runs
print(f"direct LSTM {time.time()-t0:.0f}s")
names = {"AR": "AR-LSTM (recursive)", "XGBoost": "XGBoost-LSTM (recursive)", "ANN": "ANN-LSTM (recursive, paper)", "Perfect": "Perfect rain LSTM (recursive, upper bound)"}
saved = {}
for fc_name, label in names.items():
    t0 = time.time(); runs = []
    for s in SEEDS:
        net, ep = train_onestep(FC[fc_name][:, 0], s); runs.append(run_recursive(net, FC[fc_name], te))
        if fc_name == "ANN" and s == 0: saved["net"] = net
    preds[label] = np.mean(runs, 0); seedruns[label] = runs
    print(f"{label}: {time.time()-t0:.0f}s")

# ============================================================ evaluation helpers
yt = y[te]; flood = (yt.max(1) > 0) | (seq[te, :, 0].max(1) > 0); onset = seq[te, -1, 0] == 0
def scores(p, mask):
    r = {}
    for h in range(H):
        a, b = yt[mask, h] >= 1, p[mask, h] >= 1; tp, fp, fn = (a & b).sum(), (~a & b).sum(), (a & ~b).sum()
        on = (onset[mask] & a)
        r[5*(h+1)] = dict(R2=r2_score(yt[mask, h], p[mask, h]), RMSE=np.sqrt(mean_squared_error(yt[mask, h], p[mask, h])),
                          CSI=tp/(tp+fp+fn) if tp+fp+fn else np.nan, caught=b[on].mean() if on.sum() else np.nan)
    return r
rows = []
for n, p in preds.items():
    for h, v in scores(p, flood).items(): rows.append(dict(model=n, subset="all flood samples", minutes_ahead=h, **v))
met = pd.DataFrame(rows)

# ============================================================ C. extreme storms: test storms whose deepest reading is >= 12 in
mt = meta.iloc[te].reset_index(drop=True); peak = pd.Series(np.maximum(yt.max(1), seq[te, :, 0].max(1))).groupby(mt["storm_id"]).transform("max").values
ext = flood & (peak >= 12)
print(f"\nextreme storms: {mt.loc[ext, 'storm_id'].nunique()} storms, {ext.sum()} samples")
for n, p in preds.items():
    for h, v in scores(p, ext).items(): rows.append(dict(model=n, subset="extreme storms (>= 12 in)", minutes_ahead=h, **v))
met = pd.DataFrame(rows); met.to_csv(f"{OUT}/depth5_recursive_metrics_test.csv", index=False)
order = list(preds)
for sub in ("all flood samples", "extreme storms (>= 12 in)"):
    m = met[met.subset == sub]
    print(f"\n== {sub}: R2 by minutes ahead"); print(m.pivot_table(index="model", columns="minutes_ahead", values="R2").loc[order][[5, 10, 15, 20, 30, 40]].round(3).to_string())
    print(f"== {sub}: RMSE (in)"); print(m.pivot_table(index="model", columns="minutes_ahead", values="RMSE").loc[order][[5, 15, 20, 40]].round(3).to_string())
    print(f"== {sub}: floods caught early"); print(m.pivot_table(index="model", columns="minutes_ahead", values="caught").loc[order][[10, 20, 40]].round(2).to_string())
# paper-style improvement of recursive ANN-LSTM over standard LSTM (mean over the 8 steps)
m = met[met.subset == "all flood samples"].groupby("model")[["R2", "RMSE"]].mean()
imp = {k: dict(R2_change_pct=100 * (m.loc[v, "R2"] - m.loc["LSTM (standard, direct)", "R2"]) / abs(m.loc["LSTM (standard, direct)", "R2"]),
               RMSE_change_pct=100 * (m.loc[v, "RMSE"] - m.loc["LSTM (standard, direct)", "RMSE"]) / m.loc["LSTM (standard, direct)", "RMSE"]) for k, v in names.items()}
print("\nchange vs standard LSTM, averaged over 8 steps (paper reports +6.5% R2, -13.8% RMSE):", {k: {a: round(b, 1) for a, b in v.items()} for k, v in imp.items()})
sv = {n: [float(np.sqrt(mean_squared_error(yt[flood], r[flood]))) for r in rs] for n, rs in seedruns.items()}
print("per-seed flood RMSE:", {n: f"{np.mean(v):.3f} ± {np.std(v):.3f}" for n, v in sv.items()})

# ============================================================ B. rain-forecast accuracy -> depth accuracy
sens = []
for n in ("AR", "XGBoost", "ANN"):
    sens.append(dict(forecast=n, rain_RMSE=rain_acc[n]["RMSE"], depth_R2=m.loc[names[n], "R2"], depth_RMSE=m.loc[names[n], "RMSE"], kind="forecaster"))
for a in (0.25, 0.5, 0.75):
    blend = (1 - a) * FC["ANN"] + a * fut; rr = []
    for s in SEEDS[:2]:
        net, _ = train_onestep(blend[:, 0], s); rr.append(run_recursive(net, blend, te))
    p = np.mean(rr, 0); sc = scores(p, flood)
    sens.append(dict(forecast=f"ANN + {int(a*100)}% of the way to perfect", rain_RMSE=float(np.sqrt(mean_squared_error(fut[te][wet], blend[te][wet]))),
                     depth_R2=np.mean([v["R2"] for v in sc.values()]), depth_RMSE=np.mean([v["RMSE"] for v in sc.values()]), kind="blend"))
sens.append(dict(forecast="Perfect", rain_RMSE=0.0, depth_R2=m.loc[names["Perfect"], "R2"], depth_RMSE=m.loc[names["Perfect"], "RMSE"], kind="perfect"))
sens = pd.DataFrame(sens).sort_values("rain_RMSE", ascending=False); sens.to_csv(f"{OUT}/rain_accuracy_sensitivity.csv", index=False)
print("\nrain-forecast accuracy vs depth accuracy (recursive LSTM, mean of 8 steps):\n", sens.round(4).to_string(index=False))
a0, a1 = sens.iloc[0], sens[sens.forecast == "ANN"].iloc[0]
s = np.polyfit(sens["rain_RMSE"], sens["depth_R2"], 1)
print(f"slope: each 0.1 mm lower rain RMSE -> depth R2 +{-s[0]*0.1:.3f}")

json.dump({"rain_forecasters": rain_acc, "improvement_vs_standard_lstm_pct": imp, "seed_rmse": {n: [round(x, 4) for x in v] for n, v in sv.items()},
           "extreme_storms": int(mt.loc[ext, "storm_id"].nunique()), "extreme_samples": int(ext.sum())}, open(f"{OUT}/depth5_recursive_summary.json", "w"), indent=2)
torch.save(saved["net"].state_dict(), f"{OUT}/recursive_ann_lstm.pt")

# figures
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
for n in order:
    q = met[(met.model == n) & (met.subset == "all flood samples")]
    ax[0].plot(q.minutes_ahead, q.R2, marker="o", ms=3, label=n)
ax[0].set(title="R² on test flood samples", xlabel="minutes ahead"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7)
ax[1].plot(sens.rain_RMSE, sens.depth_R2, "o-"); [ax[1].annotate(f, (x, yv), fontsize=7, xytext=(4, 4), textcoords="offset points") for f, x, yv in zip(sens.forecast, sens.rain_RMSE, sens.depth_R2)]
ax[1].set(title="Better rain forecasts -> better depth forecasts", xlabel="rain forecast RMSE (mm per 5 min)", ylabel="depth R² (mean of 8 steps)"); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_depth5_recursive.png", dpi=150)
