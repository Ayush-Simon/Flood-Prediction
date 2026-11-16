"""Stage 10a: retrain the 5-minute depth models exactly as in 07d and SAVE every test prediction,
so the rigour analyses (10b) can bootstrap over storms, test differences, split by gauge distance and
score warnings without retraining. Also fits a physics baseline (bucket model) and runs a rolling-origin
check on earlier years.
Usage: python 10a_train_and_save.py <out_dir with nyc_depth_rain_5min.csv and nyc_rain_5min_stations.csv>"""
import sys, time, json
import numpy as np, pandas as pd, torch
OUT = sys.argv[1]
here = __file__.replace("10a_train_and_save.py", "")
src = open(here + "07d_ann_cnn_lstm.py").read().split("CFG = {")[0]          # data, ANN rain model, Net, scaling: identical to 07d
ns = {"__file__": "07d"}; sys.argv = ["07d", OUT]; exec(compile(src, "07d_ann_cnn_lstm.py", "exec"), ns)
g = ns; L, H, SEEDS = g["L"], g["H"], g["SEEDS"]
seq, fut, y, meta, idx, ann_fc = g["seq"], g["fut"], g["y"], g["meta"], g["idx"], g["ann_fc"]
FUT, T_s, Y, W, Net, ysd = g["FUT"], g["T_s"], g["Y"], g["W"], g["Net"], g["ysd"]
nn = torch.nn


def train(rain, cnn, seed, tr, va, te):
    torch.manual_seed(seed); net = Net(rain != "none", cnn); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    F = FUT[rain]; tri, vai = torch.tensor(tr), torch.tensor(va); best, state, bad = 1e9, None, 0
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
    net.load_state_dict(state); net.eval()
    with torch.no_grad(): p = net(T_s[torch.tensor(te)], F[torch.tensor(te)]).numpy() * ysd
    return np.clip(p, 0, None)


# ------------------------------------------------------------------ physics baseline: a leaky bucket per street
# depth(t+1) = r * depth(t) + k * max(0, rain(t+1) - c)
#   r: share of the water still on the street after 5 minutes (drainage), c: rain the drains swallow per 5 min (mm),
#   k: inches of water per mm of excess rain. Fitted by grid search on training flood samples.
def bucket(d0, rain, r, c, k):
    out, d = [], d0.copy()
    for h in range(H):
        d = r * d + k * np.maximum(0, rain[:, h] - c); out.append(d)
    return np.stack(out, 1)
tr = idx["train"]; fm = (y[tr].max(1) > 0) | (seq[tr, :, 0].max(1) > 0); trf = tr[fm]
best = None
for r in (0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
    for c in (0, 0.25, 0.5, 1.0, 1.5, 2.0):
        for k in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5):
            e = ((bucket(seq[trf, -1, 0], fut[trf], r, c, k) - y[trf]) ** 2).mean()
            if best is None or e < best[0]: best = (e, r, c, k)
_, r, c, k = best
print(f"bucket model fitted: keeps {r:.0%} of water per 5 min, drains swallow {c} mm per 5 min, {k} in per mm of excess rain")

te = idx["test"]
P = {"Persistence": np.repeat(seq[te, -1:, 0], H, 1),
     "Bucket (physics) + ANN rain": bucket(seq[te, -1, 0], ann_fc[te], r, c, k),
     "Bucket (physics) + perfect rain": bucket(seq[te, -1, 0], fut[te], r, c, k)}
CFG = {"LSTM (standard)": ("none", False), "ANN-LSTM": ("ann", False), "CNN-LSTM": ("none", True),
       "CNN-LSTM + ANN rain": ("ann", True), "LSTM + perfect rain (upper bound)": ("perfect", False)}
seeds = {}
for name, (rain, cnn) in CFG.items():
    t0 = time.time(); runs = [train(rain, cnn, s, idx["train"], idx["val"], te) for s in SEEDS]
    P[name] = np.mean(runs, 0); seeds[name] = np.stack(runs)
    print(f"{name}: {time.time()-t0:.0f}s", flush=True)

m = meta.iloc[te].reset_index(drop=True)
np.savez_compressed(f"{OUT}/rig_preds.npz", y=y[te], now=seq[te, -1, 0], rain_now=seq[te, :, 1].sum(1),
                    names=np.array(list(P)), preds=np.stack([P[n] for n in P]),
                    seed_names=np.array(list(seeds)), seed_preds=np.stack([seeds[n] for n in seeds]),
                    sensor=m["Sensor ID"].values.astype(str), storm=m["storm_id"].values.astype(str), time=m["time"].values.astype("datetime64[s]").astype(str))
json.dump({"r": r, "c": c, "k": k}, open(f"{OUT}/rig_bucket_params.json", "w"))

# ------------------------------------------------------------------ rolling origin: same models on earlier test windows
FOLDS = [("2023-12-31 23:59", "2024-06-30 23:59", "2024-12-31 23:59"), ("2024-12-31 23:59", "2025-06-30 23:59", "2025-12-31 23:59")]
per_t = meta["time"].values; floody = (y.max(1) > 0) | (seq[:, :, 0].max(1) > 0); keep = np.zeros(len(y), bool); keep[idx["train"]] = keep[idx["val"]] = True
keep |= floody                                                    # train/val sampling as before; test = everything in its window
roll = []
for a, b, e in FOLDS:
    t1, t2, t3 = np.datetime64(a), np.datetime64(b), np.datetime64(e)
    trn = np.where((per_t <= t1) & keep)[0]; val = np.where((per_t > t1) & (per_t <= t2) & keep)[0]; tst = np.where((per_t > t2) & (per_t <= t3))[0]
    yt = y[tst]; fl = (yt.max(1) > 0) | (seq[tst, :, 0].max(1) > 0)
    for name in ("LSTM (standard)", "LSTM + perfect rain (upper bound)"):   # ANN rain model was trained through 2025, so it is left out here
        rain, cnn = CFG[name]; p = train(rain, cnn, 0, trn, val, tst)
        for h in (0, 3, 7):
            yy, pp = yt[fl, h], p[fl, h]
            roll.append(dict(test_window=f"{str(t2)[:10]} to {str(t3)[:10]}", model=name, minutes_ahead=5 * (h + 1),
                             R2=1 - ((yy - pp) ** 2).sum() / ((yy - yy.mean()) ** 2).sum(), RMSE=np.sqrt(((yy - pp) ** 2).mean()), samples=int(fl.sum())))
    pp = np.repeat(seq[tst, -1:, 0], H, 1)
    for h in (0, 3, 7):
        yy, q = yt[fl, h], pp[fl, h]
        roll.append(dict(test_window=f"{str(t2)[:10]} to {str(t3)[:10]}", model="Persistence", minutes_ahead=5 * (h + 1),
                         R2=1 - ((yy - q) ** 2).sum() / ((yy - yy.mean()) ** 2).sum(), RMSE=np.sqrt(((yy - q) ** 2).mean()), samples=int(fl.sum())))
    print("rolling fold done:", b, e, flush=True)
pd.DataFrame(roll).to_csv(f"{OUT}/rig_rolling_origin.csv", index=False)
print("done")
