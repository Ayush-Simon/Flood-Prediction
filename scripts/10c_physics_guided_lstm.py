"""Stage 10c: physics-guided LSTM (the project's own model).
The LSTM gets, besides the last hour of depth and rain, the 40-minute forecast of a simple physical
bucket model driven by the ANN rain forecast (depth(t+1) = r*depth(t) + k*max(0, rain - c), fitted in 10a).
Same data, split, seeds and training as 07d / 10a. Adds its test predictions to rig_preds.npz.
Usage: python 10c_physics_guided_lstm.py <out_dir>"""
import sys, json, time
import numpy as np, torch
OUT = sys.argv[1]; here = __file__.replace("10c_physics_guided_lstm.py", "")
src = open(here + "10a_train_and_save.py").read().split("# ------------------------------------------------------------------ physics baseline")[0]
ns = {"__file__": here + "10a_train_and_save.py"}; sys.argv = ["10a", OUT]; exec(compile(src, "10a", "exec"), ns)
g = ns; seq, fut, y, idx, ann_fc, FUT, H = g["seq"], g["fut"], g["y"], g["idx"], g["ann_fc"], g["FUT"], g["H"]
b = json.load(open(f"{OUT}/rig_bucket_params.json"))
def bucket(d0, rain):
    out, d = [], d0.copy()
    for h in range(H): d = b["r"] * d + b["k"] * np.maximum(0, rain[:, h] - b["c"]); out.append(d)
    return np.stack(out, 1).astype(np.float32)
for name, rain in (("ann", ann_fc), ("perfect", fut)):
    pb = bucket(seq[:, -1, 0], rain); tr = idx["train"]; mu, sd = pb[tr].mean(), pb[tr].std() + 1e-6
    FUT["bucket_" + name] = torch.tensor((pb - mu) / sd)
te = idx["test"]; z = dict(np.load(f"{OUT}/rig_preds.npz", allow_pickle=True))
names, preds, sn, sp = list(z["names"]), list(z["preds"]), list(z["seed_names"]), list(z["seed_preds"])
for label, key in (("Physics-guided LSTM (bucket + ANN rain)", "bucket_ann"), ("Physics-guided LSTM + perfect rain (upper bound)", "bucket_perfect")):
    t0 = time.time(); runs = [g["train"](key, False, s, idx["train"], idx["val"], te) for s in g["SEEDS"]]
    if label in names: i = names.index(label); preds[i] = np.mean(runs, 0); sp[sn.index(label)] = np.stack(runs)
    else: names.append(label); preds.append(np.mean(runs, 0)); sn.append(label); sp.append(np.stack(runs))
    print(f"{label}: {time.time()-t0:.0f}s", flush=True)
z.update(names=np.array(names), preds=np.stack(preds), seed_names=np.array(sn), seed_preds=np.stack(sp))
np.savez_compressed(f"{OUT}/rig_preds.npz", **z); print("saved")
