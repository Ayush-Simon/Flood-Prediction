"""Stage 17a: the main model as a 3-run ensemble, exactly as it was scored in 07d.

07d trained "CNN-LSTM + ANN rain forecast" three times (seeds 0, 1, 2) and reported the AVERAGE of the three
forecasts (R2 0.90 at 5 minutes). The live site ran only the seed-0 network (R2 0.80). This script retrains the
three runs with 07d's code, checks that seed 0 matches the saved live model, and writes models/live_cnn_lstm_ann.json
with all three networks ("members"). backend/flood_models.py and frontend/js/models.js average them.

Usage: python3 scripts/17a_ensemble_live_model.py <dir with nyc_depth_rain_5min.csv and nyc_rain_5min_stations.csv>
Then:  python3 scripts/16a_rain_now_past.py <same dir>  and  python3 backend/build_store.py"""
import sys, os, json, time
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1]
src = open(os.path.join(ROOT, "scripts", "07d_ann_cnn_lstm.py")).read()
pre, rest = src.split("CFG = {", 1)
train_src = "def train(" + rest.split("def train(", 1)[1].split("preds = {", 1)[0]       # 07d's own training function
ns = {"__file__": "07d"}; argv = sys.argv; sys.argv = ["07d", OUT]
exec(compile(pre + train_src, "07d_ann_cnn_lstm.py", "exec"), ns); sys.argv = argv

LIVE = os.path.join(ROOT, "models", "live_cnn_lstm_ann.json")
J = json.load(open(LIVE))
members = []
for seed in ns["SEEDS"]:
    t0 = time.time(); _, ep, net = ns["train"]("ann", True, seed)
    w = {k: v.detach().numpy().tolist() for k, v in net.state_dict().items()}
    if seed == 0:
        same = all(np.allclose(np.array(J.get("members", [J["weights"]])[0][k] if "members" in J else J["weights"][k]), np.array(v), atol=1e-6) for k, v in w.items())
        print("seed 0 identical to the saved live model:", same)
    members.append(w); print(f"seed {seed}: {ep} epochs, {time.time() - t0:.0f}s")

J["members"] = members; J["weights"] = members[0]                      # "weights" kept for older readers
J["note"] = (J.get("note", "") + " | 3-run ensemble (seeds 0,1,2) averaged, as scored in 07d (scripts/17a).").strip(" |")
with open(LIVE, "w") as f: json.dump(J, f, separators=(",", ":"))
print("wrote", LIVE, "with", len(members), "members")
