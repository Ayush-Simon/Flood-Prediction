"""Stage 14b: better flood warnings, and a fairer way to count them.

Question (same as 10b): a street is under 1 inch now; will it pass 1 inch within the next 40 minutes?
10b raised an alarm whenever a model's forecast passed a threshold, and counted every 5-minute window separately.
That gave about 1 flood in 3 caught and about 80% false alarms.

This script tries four improvements, all tuned honestly (threshold chosen on half of the test storms, scored on the
other half, then the halves swapped, exactly as in 10b):
  1. two in a row: only alarm if the forecast passes the threshold in two consecutive 5-minute windows
  2. rain gate: only alarm if some rain fell in the last hour (the threshold for "some" is tuned too)
  3. a direct classifier: XGBoost trained on 2020-2025 (stopped on Jan-Jun 2026) to answer the yes/no question
     itself, instead of forecasting depth and then applying a threshold
  4. counting events, not windows: a flood is caught if ANY alarm came in the 40 minutes before it started;
     a false alarm is one continuous alarm (a run of alarm windows) not followed by a flood within 40 minutes.
     This is how people experience warnings, and it stops one long alarm being counted as many.
Usage: python 14b_better_warnings.py <out_dir with rig_preds.npz, nyc_depth_rain_5min.csv, nyc_rain_5min_stations.csv>"""
import sys, time
import numpy as np, pandas as pd, xgboost as xgb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = sys.argv[1]; FLOOD = 1.0
here = __file__.replace("14b_better_warnings.py", "")

# ---------------------------------------------------------------- the same windows as 10a (and the ANN rain forecast)
src = open(here + "07d_ann_cnn_lstm.py").read().split("class Net")[0]
ns = {"__file__": "07d"}; argv = sys.argv; sys.argv = ["07d", OUT]
exec(compile(src.replace("import torch, torch.nn as nn", "import torch").replace("T_s = torch.tensor", "_ = None and torch.tensor"),
             "07d_ann_cnn_lstm.py", "exec"), ns); sys.argv = argv
seq, y_all, meta, idx, ann_fc, H = ns["seq"], ns["y"], ns["meta"], ns["idx"], ns["ann_fc"], ns["H"]
z = np.load(f"{OUT}/rig_preds.npz", allow_pickle=True)
te = idx["test"]
assert np.allclose(z["y"], y_all[te]), "test windows differ from 10a"
y, now, names, P = z["y"], z["now"], list(z["names"]), z["preds"]
storm, ttime = z["storm"], pd.to_datetime(z["time"])
M = {n: P[i] for i, n in enumerate(names)}

# ---------------------------------------------------------------- 3. direct classifier, trained before the test period
def feats(ix):
    s = seq[ix]; dep, rn, ms = s[:, :, 0], s[:, :, 1], s[:, :, 2]; t = pd.DatetimeIndex(meta["time"].values[ix])
    a = ann_fc[ix]; hr = t.hour + t.minute / 60
    return np.column_stack([dep[:, -1], dep[:, -2], dep[:, -4], dep[:, -7], dep.max(1), dep.mean(1), dep[:, -1] - dep[:, -4],
                            rn[:, -1], rn[:, -3:].sum(1), rn[:, -6:].sum(1), rn.sum(1), rn.max(1), (rn > 0).sum(1), ms.sum(1),
                            a.sum(1), a[:, :3].sum(1), a.max(1),
                            np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24),
                            np.sin(2 * np.pi * t.month / 12), np.cos(2 * np.pi * t.month / 12)]).astype(np.float32)
target = lambda ix: (y_all[ix].max(1) >= FLOOD) & (seq[ix, -1, 0] < FLOOD)
dry_tr = idx["train"][seq[idx["train"], -1, 0] < FLOOD]; dry_va = idx["val"][seq[idx["val"], -1, 0] < FLOOD]
# how often each sensor floods, from training windows only (smoothed towards the average)
tr_tab = pd.DataFrame({"s": meta["Sensor ID"].values[dry_tr], "t": target(dry_tr)})
rate = tr_tab.groupby("s")["t"].agg(["sum", "count"]); prior = tr_tab["t"].mean()
srate = ((rate["sum"] + 20 * prior) / (rate["count"] + 20)).to_dict()
sr = lambda ix: np.array([srate.get(s, prior) for s in meta["Sensor ID"].values[ix]], np.float32)
Xtr, Xva, Xte = (np.column_stack([feats(i), sr(i)]) for i in (dry_tr, dry_va, te))
t0 = time.time()
clf = xgb.XGBClassifier(n_estimators=2000, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                        eval_metric="aucpr", early_stopping_rounds=100, tree_method="hist", n_jobs=2, random_state=0)
clf.fit(Xtr, target(dry_tr), eval_set=[(Xva, target(dry_va))], verbose=False)
p_clf = clf.predict_proba(Xte)[:, 1]
print(f"classifier: {len(dry_tr)} training windows ({target(dry_tr).mean():.1%} positive), {clf.best_iteration} trees, {time.time()-t0:.0f}s")

# ---------------------------------------------------------------- order windows within each storm (one sensor per storm, 5-minute steps)
o = np.lexsort((ttime.values, storm)); inv = np.empty_like(o); inv[o] = np.arange(len(o))
st_o = storm[o]; new = np.r_[True, st_o[1:] != st_o[:-1]]                # first window of each storm
dry = now < FLOOD; tgt = dry & (y.max(1) >= FLOOD)
onset = dry & (y[:, 0] >= FLOOD)                                          # the flood starts 5 minutes after this window
half = pd.Series(storm).astype("category").cat.codes.values % 2
rain_hr = z["rain_now"]

def prev(a):                                                             # value in the previous window of the same storm
    b = a[o]; p = np.r_[False, b[:-1]]; p[new] = False; return p[inv]

def event_counts(alarm, m):
    """alarm, m: per window (original order). Returns caught floods, missed floods, false alarms (runs), lead times."""
    a, on, mm = alarm[o] & m[o], onset[o] & m[o], m[o]
    n = len(a); caught = missed = 0; leads = []
    # a flood starting after window i is caught if an alarm came in windows i-7..i of the same storm (40 min before it)
    sid = np.cumsum(new)
    for i in np.where(on)[0]:
        j0 = max(i - 7, 0); js = np.arange(j0, i + 1); js = js[sid[js] == sid[i]]
        hit = js[a[js]]
        if len(hit): caught += 1; leads.append(5 * (i - hit[0] + 1))
        else: missed += 1
    # runs of alarm windows; a run is true if a flood starts within 40 minutes after any of its windows
    soon = np.zeros(n, bool)
    for k in range(8):                                                   # onset in window i+k (flood at i+k+1 steps)
        sh = np.r_[on[k:], np.zeros(k, bool)]; same = np.r_[sid[k:], np.full(k, -1)] == sid; soon |= sh & same
    start = a & ~np.r_[False, a[:-1] & (sid[:-1] == sid[1:])]
    run = np.cumsum(start); run[~a] = 0
    true_runs = set(run[a & soon]); runs = set(run[a]) - {0}
    false = len(runs - true_runs)
    return caught, missed, false, leads

def window_counts(alarm, m):
    w = alarm & m; t = tgt & m
    return (w & t).sum(), (t & ~w).sum(), (w & ~t).sum()

def alarms(score, th, rule, gate):
    a = (score >= th) & dry
    if rule == "two in a row": a = a & prev(a)
    if gate > 0: a = a & (rain_hr >= gate)
    return a

SCORES = {"Persistence": M["Persistence"].max(1), "LSTM": M["LSTM (standard)"].max(1), "ANN-LSTM": M["ANN-LSTM"].max(1),
          "Direct classifier (XGBoost)": p_clf}
GRID = {k: np.round(np.arange(0.05, 2.01, 0.05), 2) for k in SCORES}
GRID["Direct classifier (XGBoost)"] = np.unique(np.round(np.quantile(p_clf[dry], np.linspace(0.80, 0.999, 60)), 4))
GATES = (0.0, 0.25, 0.5, 1.0, 2.0)
METHODS = [("single window (as before)", "one", False), ("two in a row", "two in a row", False),
           ("rain gate", "one", True), ("two in a row + rain gate", "two in a row", True)]

rows = []
for sname, sc in SCORES.items():
    for mname, rule, use_gate in METHODS:
        for counting in ("windows (as in 10b)", "events"):
            TP = FN = FP = 0; leads = []; picked = []
            for h_fit in (0, 1):
                fit, ev = (half == h_fit), (half != h_fit)
                best = None
                for g in (GATES if use_gate else (0.0,)):
                    for th in GRID[sname]:
                        a = alarms(sc, th, rule, g)
                        if counting == "events": c, m_, f, _ = event_counts(a, fit)
                        else: c, m_, f = window_counts(a, fit)
                        csi = c / max(c + m_ + f, 1)
                        if best is None or csi > best[0]: best = (csi, th, g)
                _, th, g = best; picked.append(f"{th:g}" + (f" & rain≥{g:g}mm" if use_gate else ""))
                a = alarms(sc, th, rule, g)
                if counting == "events": c, m_, f, l = event_counts(a, ev); leads += l
                else:
                    c, m_, f = window_counts(a, ev)
                    hit = a & ev & tgt; leads += list(5 * (np.argmax(y[hit] >= FLOOD, 1) + 1))
                TP += c; FN += m_; FP += f
            rows.append({"score": sname, "method": mname, "counting": counting, "settings": " / ".join(picked),
                         "floods": int(TP + FN), "caught": int(TP), "hit_rate": TP / max(TP + FN, 1),
                         "false_alarm_ratio": FP / max(TP + FP, 1), "CSI": TP / max(TP + FN + FP, 1),
                         "median_lead_min": float(np.median(leads)) if leads else np.nan})

# ---------------------------------------------------------------- 5. combined rule: water showing OR the classifier is worried
# "water is showing" catches floods that build up slowly; the classifier catches floods that start on a dry street
for counting in ("windows (as in 10b)", "events"):
    TP = FN = FP = 0; leads = []; picked = []
    for h_fit in (0, 1):
        fit, evm = (half == h_fit), (half != h_fit); best = None
        for wa in (0.05, 0.1, 0.2, 0.35, 0.5, 99):
            for th in GRID["Direct classifier (XGBoost)"]:
                a = dry & ((now >= wa) | (p_clf >= th))
                c, m_, f = (event_counts(a, fit)[:3] if counting == "events" else window_counts(a, fit))
                csi = c / max(c + m_ + f, 1)
                if best is None or csi > best[0]: best = (csi, wa, th)
        _, wa, th = best; picked.append(f"water≥{wa:g}in or p≥{th:g}")
        a = dry & ((now >= wa) | (p_clf >= th))
        if counting == "events": c, m_, f, l = event_counts(a, evm); leads += l
        else:
            c, m_, f = window_counts(a, evm); hit = a & evm & tgt; leads += list(5 * (np.argmax(y[hit] >= FLOOD, 1) + 1))
        TP += c; FN += m_; FP += f
    rows.append({"score": "Water showing OR classifier", "method": "combined", "counting": counting, "settings": " / ".join(picked),
                 "floods": int(TP + FN), "caught": int(TP), "hit_rate": TP / max(TP + FN, 1), "false_alarm_ratio": FP / max(TP + FP, 1),
                 "CSI": TP / max(TP + FN + FP, 1), "median_lead_min": float(np.median(leads)) if leads else np.nan})
print("done: combined", flush=True)
res = pd.DataFrame(rows); res.to_csv(f"{OUT}/rig_warnings_v2.csv", index=False)
pd.set_option("display.width", 220)
for counting in ("windows (as in 10b)", "events"):
    print(f"\nCounting {counting}: threshold tuned on one half of the test storms, scored on the other (both ways)")
    print(res[res.counting == counting].drop(columns="counting").round(3).to_string(index=False))

# ---------------------------------------------------------------- figure: hit rate vs false alarms, before and after
fig, ax = plt.subplots(figsize=(7.5, 5.5))
ev = res[res.counting == "events"]
for sname, mk in zip(list(SCORES) + ["Water showing OR classifier"], "osD^*"):
    g = ev[ev.score == sname]
    ax.scatter(g.false_alarm_ratio, g.hit_rate, marker=mk, s=60, label=sname)
old = res[(res.counting != "events") & (res.method == "single window (as before)") & (res.score == "LSTM")].iloc[0]
ax.scatter(old.false_alarm_ratio, old.hit_rate, marker="x", s=120, c="k", label="LSTM as reported before (windows)")
ax.set_xlabel("false alarm ratio (lower is better)"); ax.set_ylabel("share of floods caught (higher is better)")
ax.set_title("Flood warnings, test storms 2026 (event counting)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_warnings_v2.png", dpi=150)
print("\nsaved rig_warnings_v2.csv and fig_warnings_v2.png")
