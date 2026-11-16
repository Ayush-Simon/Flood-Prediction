"""Stage 13a: train and export the two models for the Model Lab page, separately.
  ANN-LSTM : ANN forecasts the next 40 min of rain, LSTM predicts depth from the last hour + that forecast
  CNN-LSTM : 1-D convolutions over the last hour, then an LSTM; no rain forecast
Same data, split and training as 07d (seed 0). Writes backend/data/model_ann_lstm.json, model_cnn_lstm.json
and lab_samples.json (real test storms to try), which build_store.py loads into the store.
Usage: python 13a_export_lab_models.py <out_dir with the 5-minute data> <backend/data dir>"""
import sys, json
import numpy as np, pandas as pd, torch
OUT, DEST = sys.argv[1], sys.argv[2]
src = open(__file__.replace("13a_export_lab_models.py", "07d_ann_cnn_lstm.py")).read().split("CFG = {")[0]
ns = {"__file__": "07d"}; sys.argv = ["07d", OUT]; exec(compile(src, "07d_ann_cnn_lstm.py", "exec"), ns)
g = ns; H, L = g["H"], g["L"]; ann = g["ann"]; y, seq, meta, idx = g["y"], g["seq"], g["meta"], g["idx"]
tr2 = idx["train"]

def train(rain, cnn, seed=0):
    torch.manual_seed(seed); net = g["Net"](rain != "none", cnn); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    F = g["FUT"][rain]; tri, vai = torch.tensor(tr2), torch.tensor(idx["val"]); best, state, bad = 1e9, None, 0
    T_s, Y, W = g["T_s"], g["Y"], g["W"]
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
    net.load_state_dict(state); net.eval(); te = torch.tensor(idx["test"])
    with torch.no_grad(): p = np.clip(net(T_s[te], F[te]).numpy() * g["ysd"], 0, None)
    yt = y[idx["test"]]; fl = (yt.max(1) > 0) | (seq[idx["test"], :, 0].max(1) > 0)
    r2 = [float(1 - ((yt[fl, h] - p[fl, h])**2).sum() / ((yt[fl, h] - yt[fl, h].mean())**2).sum()) for h in range(H)]
    rmse = [float(np.sqrt(((yt[fl, h] - p[fl, h])**2).mean())) for h in range(H)]
    return net, r2, rmse

common = {"L": L, "H": H, "step_min": 5, "inputs": ["depth_in", "rain_mm", "rain_missing"],
          "seq_mean": g["sm"].tolist(), "seq_std": g["ss"].tolist(), "fut_mean": float(g["fm"]), "fut_std": float(g["fs"]), "y_sd": float(g["ysd"]),
          "test_period": "Jul-Oct 2026, flood-related windows"}
for name, rain, cnn, file in (("ANN-LSTM", "ann", False, "model_ann_lstm.json"), ("CNN-LSTM", "none", True, "model_cnn_lstm.json")):
    net, r2, rmse = train(rain, cnn)
    j = {"name": name, "cnn": cnn, "rain_forecast": rain == "ann", **common, "test_r2": r2, "test_rmse_in": rmse,
         "weights": {k: v.numpy().round(7).tolist() for k, v in net.state_dict().items()}}
    if rain == "ann":
        j["ann"] = {"coefs": [c.round(7).tolist() for c in ann.coefs_], "intercepts": [b.round(7).tolist() for b in ann.intercepts_],
                    "mu": g["mu"].tolist(), "sd": g["sd"].tolist()}
    json.dump(j, open(f"{DEST}/{file}", "w"), separators=(",", ":"))
    # reference outputs for checking the browser version
    te = idx["test"][:5]
    with torch.no_grad(): ref = np.clip(net(g["T_s"][torch.tensor(te)], g["FUT"][rain][torch.tensor(te)]).numpy() * g["ysd"], 0, None)
    json.dump({"seq": seq[te].tolist(), "time": [str(t) for t in meta["time"].values[te]], "out": ref.tolist()}, open(f"{OUT}/check_{file}", "w"))
    print(f"{name}: R2 5/20/40 min = {r2[0]:.2f} / {r2[3]:.2f} / {r2[7]:.2f}")

# sample storms: whole test storms at a sensor, with the most water
d = pd.read_csv(f"{OUT}/nyc_depth_rain_5min.csv", parse_dates=["time"])
d = d[d["time"] > "2026-06-30"]
pk = d.groupby("storm_id")["depth_in"].max().sort_values(ascending=False)
names = pd.read_csv(f"{OUT}/floodnet_sensors.csv").drop_duplicates("Sensor ID").set_index("Sensor ID")["Sensor Name"]
samples = []
for sid in pk.index[:40]:
    s = d[d["storm_id"] == sid].sort_values("time")
    if len(s) < 30 or len(samples) >= 6 or any(x["sensor"] == s["Sensor ID"].iloc[0] for x in samples): continue
    samples.append({"id": str(sid), "sensor": s["Sensor ID"].iloc[0], "name": str(names.get(s["Sensor ID"].iloc[0], s["Sensor ID"].iloc[0])),
                    "peak_in": round(float(s["depth_in"].max()), 1),
                    "csv": "time_utc,rain_mm,depth_in\n" + "\n".join(f"{t:%Y-%m-%d %H:%M},{'' if pd.isna(r) else round(r, 2)},{round(v, 2)}" for t, r, v in zip(s["time"], s["rain_mm"], s["depth_in"]))})
json.dump(samples, open(f"{DEST}/lab_samples.json", "w"))
print("samples:", [(x["name"], x["peak_in"]) for x in samples])
