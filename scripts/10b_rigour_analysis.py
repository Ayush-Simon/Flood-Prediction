"""Stage 10b: statistics on the saved test predictions from 10a.
  1. 95% confidence intervals by resampling whole storms (a storm's windows move together)
  2. paired tests: is LSTM really better than ANN-LSTM, CNN-LSTM, ...?
  3. gauge distance: does the rain forecast help more where the rain gauge is close to the sensor?
  4. warnings: hit rate, false alarms and lead time for "the street will flood within 40 minutes",
     with the alarm threshold chosen on one half of the test storms and scored on the other half
Usage: python 10b_rigour_analysis.py <out_dir>"""
import sys, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = sys.argv[1]; B = 1000; rng = np.random.default_rng(42); FLOOD = 1.0
z = np.load(f"{OUT}/rig_preds.npz", allow_pickle=True)   # our own file
y, now, names, P, storm, sensor = z["y"], z["now"], list(z["names"]), z["preds"], z["storm"], z["sensor"]
d = pd.read_csv(f"{OUT}/nyc_depth_rain_5min.csv", usecols=["Sensor ID", "storm_id", "time", "depth_in", "gauge_km"])
gkm = d.groupby("Sensor ID")["gauge_km"].first()
# a window counts as flood-related if water was on the street in the last hour or the next 40 minutes (as in 07d)
d["wet"] = d["depth_in"] > 0
past_wet = d.groupby("storm_id")["wet"].transform(lambda s: s.rolling(12, min_periods=1).max()).astype(bool)
key = pd.Series(past_wet.values, index=pd.MultiIndex.from_arrays([d["storm_id"], pd.to_datetime(d["time"])]))
pw = key.reindex(pd.MultiIndex.from_arrays([storm, pd.to_datetime(z["time"])])).fillna(False).values.astype(bool)
fl = (y.max(1) > 0) | pw
M = {n: P[i] for i, n in enumerate(names)}
REAL = [n for n in names if "perfect" not in n]
print(f"test windows {len(y)}, flood-related {fl.sum()}, storms with water {len(set(storm[fl]))}")

# ------------------------------------------------------------------ helpers
st_f = storm[fl]; ust = np.unique(st_f); groups = {s: np.where(st_f == s)[0] for s in ust}
def r2(a, b): return 1 - ((a - b) ** 2).sum() / max(((a - a.mean()) ** 2).sum(), 1e-9)
def stats(yy, pp, nn):
    on = nn == 0
    out = {f"R2_{5*(h+1)}": r2(yy[:, h], pp[:, h]) for h in (0, 3, 7)}
    out["RMSE"] = np.sqrt(((yy - pp) ** 2).mean())
    a, b = yy[:, 7] >= FLOOD, pp[:, 7] >= FLOOD; o = on & a
    out["caught_40"] = b[o].mean() if o.sum() else np.nan
    return out
def boot(fn):
    vals = []
    for _ in range(B):
        pick = np.concatenate([groups[s] for s in rng.choice(ust, len(ust))])
        vals.append(fn(pick))
    return vals
yf, nf = y[fl], now[fl]

# ------------------------------------------------------------------ 1. confidence intervals
rows = []
for n in names:
    pf = M[n][fl]; point = stats(yf, pf, nf); bs = pd.DataFrame(boot(lambda i: stats(yf[i], pf[i], nf[i])))
    rows.append({"model": n, **{f"{k}": point[k] for k in point}, **{f"{k}_lo": bs[k].quantile(.025) for k in point}, **{f"{k}_hi": bs[k].quantile(.975) for k in point}})
ci = pd.DataFrame(rows); ci.to_csv(f"{OUT}/rig_confidence_intervals.csv", index=False)
fmt = lambda r, k, f="{:.2f}": f"{f.format(r[k])} [{f.format(r[k+'_lo'])}, {f.format(r[k+'_hi'])}]"
print("\n1. Test results with 95% confidence intervals (resampling storms):")
print(pd.DataFrame({r["model"]: {"R2 5 min": fmt(r, "R2_5"), "R2 20 min": fmt(r, "R2_20"), "R2 40 min": fmt(r, "R2_40"),
                                  "RMSE in": fmt(r, "RMSE"), "caught 40 min": fmt(r, "caught_40", "{:.0%}")} for _, r in ci.iterrows()}).T.to_string())

# ------------------------------------------------------------------ 2. paired comparisons against the standard LSTM
base = "LSTM (standard)"; comp = []
for n in names:
    if n == base: continue
    e0, e1 = ((M[base][fl] - yf) ** 2).mean(1), ((M[n][fl] - yf) ** 2).mean(1)
    diff = lambda i: np.sqrt(e1[i].mean()) - np.sqrt(e0[i].mean())
    dr = lambda i: r2(yf[i, 3], M[n][fl][i, 3]) - r2(yf[i, 3], M[base][fl][i, 3])
    bd, br = np.array(boot(diff)), np.array(boot(dr))
    ps = pd.DataFrame({"a": e0, "b": e1, "s": st_f}).groupby("s").mean()
    w = wilcoxon(ps["b"], ps["a"]) if (ps["b"] != ps["a"]).any() else None
    comp.append({"model": n, "RMSE_minus_LSTM": diff(np.arange(len(yf))), "lo": np.quantile(bd, .025), "hi": np.quantile(bd, .975),
                 "p_boot": 2 * min((bd <= 0).mean(), (bd >= 0).mean()), "R2_20_minus_LSTM": dr(np.arange(len(yf))),
                 "R2_lo": np.quantile(br, .025), "R2_hi": np.quantile(br, .975),
                 "storms_better": f"{(ps['b'] < ps['a']).sum()} of {len(ps)}", "p_wilcoxon": w.pvalue if w else np.nan})
cmp = pd.DataFrame(comp); cmp.to_csv(f"{OUT}/rig_paired_tests.csv", index=False)
print("\n2. Each model minus the standard LSTM (negative RMSE difference = better than LSTM):")
print(cmp.round(3).to_string(index=False))
sp = z["seed_preds"]; sn = list(z["seed_names"])
seedv = {n: [np.sqrt(((sp[i, s][fl] - yf) ** 2).mean()) for s in range(sp.shape[1])] for i, n in enumerate(sn)}
print("\nRMSE of each training seed:", {n: [round(v, 3) for v in vs] for n, vs in seedv.items()})

# ------------------------------------------------------------------ 3. gauge distance
km = pd.Series(sensor).map(gkm).values; kf = km[fl]
print("\n3. Sensor-to-gauge distance (km): min %.1f, median %.1f, max %.1f" % (np.nanmin(km), np.nanmedian(km), np.nanmax(km)))
cut = float(np.nanmedian(pd.Series(gkm.reindex(np.unique(sensor[fl])))))
drows = []
for label, mask in ((f"near (≤ {cut:.0f} km)", kf <= cut), (f"far (> {cut:.0f} km)", kf > cut)):
    ii = np.where(mask)[0]; ss = st_f[ii]; us = np.unique(ss); gg = {s: ii[ss == s] for s in us}
    def gboot(fn):
        return np.array([fn(np.concatenate([gg[s] for s in rng.choice(us, len(us))])) for _ in range(B)])
    for n in ("ANN-LSTM", "LSTM + perfect rain (upper bound)", "Bucket (physics) + ANN rain"):
        f = lambda i, n=n: r2(yf[i, 3], M[n][fl][i, 3]) - r2(yf[i, 3], M[base][fl][i, 3])
        f40 = lambda i, n=n: r2(yf[i, 7], M[n][fl][i, 7]) - r2(yf[i, 7], M[base][fl][i, 7])
        b20, b40 = gboot(f), gboot(f40)
        drows.append({"group": label, "sensors": len(set(sensor[fl][ii])), "storms": len(us), "model": n,
                      "R2_gain_20": f(ii), "lo20": np.quantile(b20, .025), "hi20": np.quantile(b20, .975),
                      "R2_gain_40": f40(ii), "lo40": np.quantile(b40, .025), "hi40": np.quantile(b40, .975)})
dist = pd.DataFrame(drows); dist.to_csv(f"{OUT}/rig_gauge_distance.csv", index=False)
print("Gain in R² over the standard LSTM, by distance to the gauge [95% CI]:"); print(dist.round(3).to_string(index=False))
ps = []
for s_id in np.unique(sensor[fl]):
    ii = np.where(sensor[fl] == s_id)[0]
    if (yf[ii, 3] > 0).sum() < 30: continue
    ps.append({"sensor": s_id, "km": gkm[s_id], "gain_ann": r2(yf[ii, 3], M["ANN-LSTM"][fl][ii, 3]) - r2(yf[ii, 3], M[base][fl][ii, 3]),
               "gain_perfect": r2(yf[ii, 3], M["LSTM + perfect rain (upper bound)"][fl][ii, 3]) - r2(yf[ii, 3], M[base][fl][ii, 3])})
ps = pd.DataFrame(ps); ps.to_csv(f"{OUT}/rig_gauge_distance_per_sensor.csv", index=False)
rho_a, rho_p = spearmanr(ps["km"], ps["gain_ann"]), spearmanr(ps["km"], ps["gain_perfect"])
print(f"per sensor ({len(ps)} sensors with water): Spearman distance vs gain, ANN rain {rho_a.statistic:.2f} (p {rho_a.pvalue:.2f}), perfect rain {rho_p.statistic:.2f} (p {rho_p.pvalue:.2f})")

# ------------------------------------------------------------------ 4. warnings: "will this dry street flood within 40 minutes?"
dry = now < FLOOD; tgt = (y.max(1) >= FLOOD) & dry
first = np.argmax(y >= FLOOD, 1)
half = np.array([hash(s) % 2 for s in storm]) if False else (pd.Series(storm).astype("category").cat.codes.values % 2)
TAU = np.round(np.arange(0.05, 2.01, 0.05), 2)
def score(sc, th, m):
    w = (sc >= th) & m; a = tgt & m
    tp, fp, fn = (w & a).sum(), (w & ~a).sum(), (a & ~w).sum()
    return tp, fp, fn
wrows, curves = [], {}
for n in names:
    sc = M[n].max(1); curves[n] = []
    for th in TAU:
        tp, fp, fn = score(sc, th, dry); curves[n].append((th, tp / max(tp + fn, 1), fp / max(tp + fp, 1)))
    # cross-fitted: choose the threshold with the best CSI on one half of the storms, score it on the other half
    TP = FP = FN = 0; leads = []; ths = []
    for h_fit in (0, 1):
        fit, ev = dry & (half == h_fit), dry & (half != h_fit)
        csi = [(lambda t: t[0] / max(sum(t), 1))(score(sc, th, fit)) for th in TAU]; th = TAU[int(np.argmax(csi))]; ths.append(th)
        tp, fp, fn = score(sc, th, ev); TP += tp; FP += fp; FN += fn
        hit = (sc >= th) & ev & tgt; leads += list(5 * (first[hit] + 1))
    wrows.append({"model": n, "threshold_in": "/".join(f"{t:g}" for t in ths), "hit_rate": TP / max(TP + FN, 1), "false_alarm_ratio": FP / max(TP + FP, 1),
                  "CSI": TP / max(TP + FP + FN, 1), "median_lead_min": float(np.median(leads)) if leads else np.nan, "floods": int(TP + FN)})
warn = pd.DataFrame(wrows); warn.to_csv(f"{OUT}/rig_warnings.csv", index=False)
print("\n4. Warnings for 'a street under 1 inch now will pass 1 inch within 40 minutes'\n   (threshold tuned on half the storms, scored on the other half):")
print(warn.round(3).to_string(index=False))
# the fixed 1-inch rule used before, for comparison
for n in ("LSTM (standard)", "ANN-LSTM", "CNN-LSTM + ANN rain"):
    tp, fp, fn = score(M[n].max(1), FLOOD, dry); print(f"   {n} with the old 1-inch rule: hit rate {tp/max(tp+fn,1):.0%}, false alarms {fp/max(tp+fp,1):.0%}")

# ------------------------------------------------------------------ figures
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
o = ci.sort_values("R2_20"); yy = np.arange(len(o))
ax[0].errorbar(o["R2_20"], yy, xerr=[o["R2_20"] - o["R2_20_lo"], o["R2_20_hi"] - o["R2_20"]], fmt="o", capsize=3)
ax[0].set_yticks(yy, o["model"], fontsize=8); ax[0].set(title="R² at 20 minutes, 95% CI over storms"); ax[0].grid(alpha=.3)
for lab, g in dist[dist.model != "Bucket (physics) + ANN rain"].groupby("group", sort=False):
    xs = np.arange(len(g)) + (0 if "near" in lab else 0.25)
    ax[1].errorbar(xs, g["R2_gain_20"], yerr=[g["R2_gain_20"] - g["lo20"], g["hi20"] - g["R2_gain_20"]], fmt="o", capsize=3, label=lab)
ax[1].set_xticks([0.12, 1.12], ["ANN rain forecast", "perfect rain"]); ax[1].axhline(0, c="k", lw=.8)
ax[1].set(title="Gain in R² (20 min) over LSTM, by gauge distance"); ax[1].legend(); ax[1].grid(alpha=.3)
for n in ("LSTM (standard)", "ANN-LSTM", "CNN-LSTM + ANN rain", "Bucket (physics) + ANN rain", "LSTM + perfect rain (upper bound)"):
    c = np.array(curves[n]); ax[2].plot(c[:, 2], c[:, 1], marker=".", label=n)
ax[2].set(title="Warnings: hit rate vs false-alarm ratio", xlabel="false-alarm ratio", ylabel="hit rate"); ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_rigour.png", dpi=150)

roll = pd.read_csv(f"{OUT}/rig_rolling_origin.csv")
print("\n5. Rolling origin (R² on flood samples, one seed):")
print(roll.pivot_table(index=["test_window", "model"], columns="minutes_ahead", values="R2").round(2).to_string())
json.dump({"cut_km": cut, "rho_ann": [rho_a.statistic, rho_a.pvalue], "rho_perfect": [rho_p.statistic, rho_p.pvalue]}, open(f"{OUT}/rig_summary.json", "w"))
