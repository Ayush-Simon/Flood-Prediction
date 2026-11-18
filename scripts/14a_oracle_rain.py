"""Stage 14a: why does the 40-minute forecast fail? The oracle-rain experiment.

The models were already trained three ways (10a, 3 seeds each, same data and split):
  - LSTM (standard): no idea what rain is coming
  - ANN-LSTM: told the ANN's rain forecast for the next 40 minutes
  - LSTM + perfect rain: told the rain that REALLY fell in the next 40 minutes (impossible in practice: an upper bound)
If the 40-minute failure were caused by poor rain forecasts, perfect rain would fix most of it.

This script reads their saved test predictions (rig_preds.npz) and reports:
  1. R² at every horizon (5..40 min), with 95% intervals from resampling whole storms, and the gain from perfect rain
  2. where the 40-minute error comes from: streets already wet, floods that START in the window, streets that stay dry
  3. timing versus size: R² of "the highest depth in the next 40 minutes", which ignores WHEN the peak comes
Usage: python 14a_oracle_rain.py <out_dir with rig_preds.npz and nyc_depth_rain_5min.csv>"""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = sys.argv[1]; FLOOD = 1.0; B = 1000; rng = np.random.default_rng(0)
z = np.load(f"{OUT}/rig_preds.npz", allow_pickle=True)
y, now, names, P, storm = z["y"], z["now"], list(z["names"]), z["preds"], z["storm"]
M = {n: P[i] for i, n in enumerate(names)}

# flood-related windows, exactly as in 10b: water now/in the last hour, or in the next 40 minutes
d = pd.read_csv(f"{OUT}/nyc_depth_rain_5min.csv", usecols=["storm_id", "time", "depth_in"])
d["wet"] = d["depth_in"] > 0
past = d.groupby("storm_id")["wet"].transform(lambda s: s.rolling(12, min_periods=1).max()).astype(bool)
key = pd.Series(past.values, index=pd.MultiIndex.from_arrays([d["storm_id"], pd.to_datetime(d["time"])]))
pw = key.reindex(pd.MultiIndex.from_arrays([storm, pd.to_datetime(z["time"])])).fillna(False).values.astype(bool)
fl = (y.max(1) > 0) | pw
yf, nf, sf = y[fl], now[fl], storm[fl]
ust = np.unique(sf); grp = {s: np.where(sf == s)[0] for s in ust}
print(f"{fl.sum()} flood-related test windows from {len(ust)} storms")

SHOW = {"Persistence": "Persistence (water stays as it is)", "LSTM (standard)": "LSTM", "ANN-LSTM": "ANN-LSTM (ANN rain forecast)",
        "LSTM + perfect rain (upper bound)": "LSTM + perfect future rain (oracle)",
        "Bucket (physics) + perfect rain": "Bucket physics + perfect future rain"}
def r2(a, b): return 1 - ((a - b) ** 2).sum() / max(((a - a.mean()) ** 2).sum(), 1e-9)
def boot(fn):
    out = []
    for _ in range(B):
        i = np.concatenate([grp[s] for s in rng.choice(ust, len(ust))]); out.append(fn(i))
    return np.array(out)

# ---------- 1. R² at every horizon, and the gain from perfect rain
rows = []
for n, lab in SHOW.items():
    p = M[n][fl]
    for h in range(8):
        b = boot(lambda i: r2(yf[i, h], p[i, h]))
        rows.append({"model": lab, "minutes_ahead": 5 * (h + 1), "R2": r2(yf[:, h], p[:, h]), "lo": np.quantile(b, .025), "hi": np.quantile(b, .975)})
hz = pd.DataFrame(rows); hz.to_csv(f"{OUT}/rig_oracle_by_horizon.csv", index=False)
print("\n1. R² on flood-related test windows (95% interval over storms)")
print(hz.assign(R2=hz.apply(lambda r: f"{r.R2:5.2f} ({r.lo:.2f} to {r.hi:.2f})", axis=1))
        .pivot(index="model", columns="minutes_ahead", values="R2").loc[list(SHOW.values())].to_string())

base, orac = M["LSTM (standard)"][fl], M["LSTM + perfect rain (upper bound)"][fl]
gain = []
for h in range(8):
    b = boot(lambda i: r2(yf[i, h], orac[i, h]) - r2(yf[i, h], base[i, h]))
    gap = 1 - r2(yf[:, h], base[:, h])
    g = r2(yf[:, h], orac[:, h]) - r2(yf[:, h], base[:, h])
    gain.append({"minutes_ahead": 5 * (h + 1), "gain": g, "lo": np.quantile(b, .025), "hi": np.quantile(b, .975),
                 "share_of_missing_skill_closed": g / gap, "p_boot": 2 * min((b <= 0).mean(), (b >= 0).mean())})
gain = pd.DataFrame(gain); gain.to_csv(f"{OUT}/rig_oracle_gain.csv", index=False)
print("\nGain in R² from perfect future rain over the standard LSTM, and how much of the missing skill (1 - R²) it closes:")
print(gain.round(3).to_string(index=False))

# ---------- 2. where the 40-minute error comes from
wet_now = nf >= FLOOD
onset = (~wet_now) & (yf[:, 7] >= FLOOD)                           # dry now, flooded 40 minutes later
other = ~wet_now & ~onset
parts = []
for lab, n in (("LSTM", "LSTM (standard)"), ("LSTM + perfect rain", "LSTM + perfect rain (upper bound)")):
    e = (M[n][fl][:, 7] - yf[:, 7]) ** 2; tot = e.sum()
    for g, m in (("street already flooded (1 in or more now)", wet_now), ("flood starts within the 40 minutes", onset), ("stays under 1 in", other)):
        parts.append({"model": lab, "situation": g, "windows": int(m.sum()), "share_of_windows": m.mean(), "share_of_squared_error": e[m].sum() / tot,
                      "RMSE_in": np.sqrt(e[m].mean())})
parts = pd.DataFrame(parts); parts.to_csv(f"{OUT}/rig_oracle_error_sources.csv", index=False)
print("\n2. Where the 40-minute error comes from:")
print(parts.round(3).to_string(index=False))

# ---------- 3. size versus timing: the highest depth in the next 40 minutes, whenever it comes
pk = []
for n, lab in SHOW.items():
    p = M[n][fl]; b = boot(lambda i: r2(yf[i].max(1), p[i].max(1)))
    pk.append({"model": lab, "R2_peak_next_40min": r2(yf.max(1), p.max(1)), "lo": np.quantile(b, .025), "hi": np.quantile(b, .975),
               "R2_exact_40min": r2(yf[:, 7], p[:, 7])})
pk = pd.DataFrame(pk); pk.to_csv(f"{OUT}/rig_oracle_peak.csv", index=False)
print("\n3. Forecasting the highest depth in the next 40 minutes (timing ignored) versus the depth at exactly 40 minutes:")
print(pk.round(3).to_string(index=False))

# ---------- figure
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for lab in SHOW.values():
    g = hz[hz.model == lab]
    ln, = ax[0].plot(g.minutes_ahead, g.R2, "o-", lw=2 if "oracle" in lab else 1.4, label=lab)
    if "oracle" in lab or lab == "LSTM": ax[0].fill_between(g.minutes_ahead, g.lo, g.hi, alpha=.12, color=ln.get_color())
ax[0].axhline(0, color="k", lw=.8); ax[0].set_xlabel("minutes ahead"); ax[0].set_ylabel("R² (flood-related test windows)")
ax[0].set_title("Even perfect future rain lifts the 40-minute R² only a little"); ax[0].legend(fontsize=8)
for k, lab in enumerate(("LSTM", "LSTM + perfect rain")):
    g = parts[parts.model == lab]
    ax[1].bar(np.arange(3) + (k - .5) * .38, g.share_of_squared_error, .38, label=lab)
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(["already flooded", "flood starts\nin the 40 min", "stays dry"])
ax[1].set_ylabel("share of squared error at 40 min"); ax[1].set_title("Where the 40-minute error comes from"); ax[1].legend()
fig.tight_layout(); fig.savefig(f"{OUT}/fig_oracle_rain.png", dpi=150)
print("\nsaved rig_oracle_*.csv and fig_oracle_rain.png")
