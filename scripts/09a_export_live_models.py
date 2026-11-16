"""Live mode: retrain the proposed model (ANN rain forecaster + CNN-LSTM with ANN rain forecast)
exactly as in 07d (seed 0) and export every weight to plain JSON, so the live app can run it with numpy only.
Usage: python 09a_export_live_models.py <dir with nyc_depth_rain_5min.csv, nyc_rain_5min_stations.csv> <out json>"""
import sys, json, runpy, types
import numpy as np, torch
D, OUT = sys.argv[1:3]
sys.argv = ["07d", D]
# run 07d's data + ANN part, but train only the one model we need
src = open(__file__.replace("09a_export_live_models.py", "07d_ann_cnn_lstm.py")).read()
src = src.split("CFG = {")[0]                     # everything up to the model list: data, ANN, Net, tensors
ns = {}; exec(compile(src, "07d_ann_cnn_lstm.py", "exec"), ns)
Net, T_s, FUT, Y, W, tr, idx = ns["Net"], ns["T_s"], ns["FUT"], ns["Y"], ns["W"], ns["tr"], ns["idx"]
torch.manual_seed(0); net = Net(True, True); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
F = FUT["ann"]; tri, vai = torch.tensor(tr), torch.tensor(idx["val"]); best, state, bad = 1e9, None, 0
for ep in range(60):
    net.train(); perm = tri[torch.randperm(len(tri))]
    for b in range(0, len(perm), 512):
        j = perm[b:b+512]; loss = ((net(T_s[j], F[j]) - Y[j])**2 * W[j, None]).mean(); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad(): vl = (((net(T_s[vai], F[vai]) - Y[vai])**2) * W[vai, None]).mean().item()
    if vl < best - 1e-5: best, state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
    else:
        bad += 1
        if bad >= 8: break
net.load_state_dict(state); net.eval()
ann = ns["ann"]
out = {"note": "CNN-LSTM + ANN rain forecast (seed 0) and its ANN rain forecaster; inputs = last 12 five-minute steps",
       "L": ns["L"], "H": ns["H"],
       "ann": {"coefs": [c.tolist() for c in ann.coefs_], "intercepts": [b.tolist() for b in ann.intercepts_], "mu": ns["mu"].tolist(), "sd": ns["sd"].tolist()},
       "seq_mean": ns["sm"].tolist(), "seq_std": ns["ss"].tolist(), "fut_mean": float(ns["fm"]), "fut_std": float(ns["fs"]), "y_sd": float(ns["ysd"]),
       "weights": {k: v.numpy().tolist() for k, v in net.state_dict().items()}}
json.dump(out, open(OUT, "w"))
# self-check: numpy re-implementation must match torch
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(OUT)), "..", "backend"))
from flood_models import DepthModel
m = DepthModel(out); te = idx["test"][:500]
seq = ns["seq"][te]; times = ns["meta"]["time"].values[te]
with torch.no_grad(): ref = np.clip(net(T_s[te], F[te]).numpy() * ns["ysd"], 0, None)
mine = m.predict(seq, times)
print("epochs", ep + 1, "| max difference torch vs numpy:", float(np.abs(ref - mine).max()))
