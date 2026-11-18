"""Stage 17c: tuned tree models for the street flood-risk map (papers: Nguyen et al. 2021; Chaimook et al. 2025).

Same data, labels, sampling and test as 05c (it reuses 05c's own code up to the model section), then:
  1. Time-based cross-validation on the TRAINING storms only: 3 folds, each trains on earlier storms and
     validates on the next block of storms (never the future to predict the past).
  2. Grid search for XGBoost, LightGBM and Random Forest, scored by weighted PR-AUC on the fold.
  3. Each model's best settings are refitted on all training storms (boosters stop early on the 05c validation
     storms) and tested on every street on every 2026 storm day, exactly as in 05c.
The map keeps using the 05c XGBoost; this script only reports whether tuning or another tree model would do better.
Usage: python3 scripts/17c_tree_model_tuning.py <dir with 05b outputs> <reports dir>"""
import sys, os, json, time, itertools, warnings
import numpy as np, pandas as pd, xgboost as xgb, lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN, OUT = sys.argv[1:3]

# ---------- data exactly as in 05c (same seed, same samples), recording each row's storm day ----------
src = open(os.path.join(ROOT, "scripts", "05c_road_risk_model.py")).read().split("# ---------- models ----------")[0]
src = src.replace("X.append(x); Y.append(y); W.append(", "_D.append(np.full(len(y), d.toordinal())); X.append(x); Y.append(y); W.append(")
ns = {"_D": []}; argv = sys.argv; sys.argv = ["05c", IN, OUT]
exec(compile(src, "05c_road_risk_model.py", "exec"), ns); sys.argv = argv
Xtr, ytr, wtr, Xva, yva, wva = (ns[k] for k in ("Xtr", "ytr", "wtr", "Xva", "yva", "wva"))
Dtr = np.concatenate(ns["_D"])[:len(ytr)]
N, build, te_days, rng = ns["N"], ns["build"], ns["te_days"], ns["rng"]

# ---------- time-based folds on the training storms ----------
days = np.sort(np.unique(Dtr)); cuts = np.array_split(days, 4)            # 4 blocks -> 3 expanding folds
folds = [(np.isin(Dtr, np.concatenate(cuts[:k])), np.isin(Dtr, cuts[k])) for k in (1, 2, 3)]
print("folds (train rows / validation rows):", [(int(a.sum()), int(b.sum())) for a, b in folds])
pr = lambda y, p, w: average_precision_score(y, p, sample_weight=w)

def fit(kind, prm, Xa, ya, wa, Xb=None, yb=None, wb=None):
    if kind == "XGBoost":
        m = xgb.XGBClassifier(n_estimators=1500, subsample=0.8, colsample_bytree=prm.pop("colsample_bytree", 0.8), n_jobs=2, eval_metric="aucpr", early_stopping_rounds=60, tree_method="hist", **prm)
        m.fit(Xa, ya, sample_weight=wa, eval_set=[(Xb, yb)], sample_weight_eval_set=[wb], verbose=False)
    elif kind == "LightGBM":
        m = lgb.LGBMClassifier(n_estimators=1500, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=2, verbose=-1, **prm)
        m.fit(Xa, ya, sample_weight=wa, eval_set=[(Xb, yb)], eval_sample_weight=[wb], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(60, verbose=False)])
    else:
        m = RandomForestClassifier(n_estimators=200, max_samples=0.3, max_features="sqrt", n_jobs=2, random_state=0, **prm); m.fit(Xa, ya, sample_weight=wa)
    return m

GRID = {"XGBoost": dict(max_depth=[3, 6, 9], learning_rate=[0.05, 0.1], min_child_weight=[1, 20]),
        "LightGBM": dict(num_leaves=[15, 63], learning_rate=[0.05, 0.1], min_child_samples=[20, 200]),
        "Random Forest": dict(max_depth=[10, 20], min_samples_leaf=[5, 50])}
cv_rows, best = [], {}
for kind, g in GRID.items():
    t0 = time.time()
    for combo in itertools.product(*g.values()):
        prm = dict(zip(g, combo)); sc = []
        for a, b in folds:
            m = fit(kind, prm, Xtr[a], ytr[a], wtr[a], Xtr[b], ytr[b], wtr[b])
            sc.append(pr(ytr[b], m.predict_proba(Xtr[b])[:, 1], wtr[b]))
        print(f"  {kind} {prm}: CV PR-AUC {np.mean(sc):.4f}", flush=True)
        cv_rows.append(dict(model=kind, **{k: str(v) for k, v in prm.items()}, cv_pr_auc=np.mean(sc), cv_sd=np.std(sc)))
        if kind not in best or np.mean(sc) > best[kind][1]: best[kind] = (prm, np.mean(sc))
    print(f"{kind}: {len(list(itertools.product(*g.values())))} settings x 3 folds in {time.time()-t0:.0f}s; best {best[kind][0]} (CV PR-AUC {best[kind][1]:.4f})")
pd.DataFrame(cv_rows).sort_values(["model", "cv_pr_auc"], ascending=[True, False]).to_csv(f"{OUT}/tuning_cv_results.csv", index=False)

# ---------- refit on all training storms, test exactly as 05c ----------
M = {"XGBoost, 05c settings (the map)": fit("XGBoost", dict(max_depth=6, learning_rate=0.05, colsample_bytree=0.8, min_child_weight=5), Xtr, ytr, wtr, Xva, yva, wva)}
for kind, (prm, _) in best.items(): M[f"{kind}, tuned"] = fit(kind, prm, Xtr, ytr, wtr, Xva, yva, wva)
ks = [0.01, 0.05, 0.10]; agg = {n: {"y": [], "p": [], "hits": np.zeros(3)} for n in M}; total = 0; allidx = np.arange(N)
for d in te_days:
    X, y = build(d, allidx); total += y.sum(); sub = rng.random(N) < 0.05
    for n, m in M.items():
        p = m.predict_proba(X)[:, 1].astype(np.float32); order = np.argsort(-p, kind="stable")
        agg[n]["hits"] += [y[order[:int(N * k)]].sum() for k in ks]
        agg[n]["y"].append(np.concatenate([y[y], y[~y & sub]])); agg[n]["p"].append(np.concatenate([p[y], p[~y & sub]]))
rows = []
for n, a in agg.items():
    y = np.concatenate(a["y"]); p = np.concatenate(a["p"]); wt = np.where(y, 1.0, 20.0)
    rows.append(dict(model=n, settings=json.dumps(best.get(n.split(",")[0], ({},))[0]) if "tuned" in n else "max_depth 6, learning_rate 0.05, min_child_weight 5, colsample 0.8",
                     ROC_AUC=roc_auc_score(y, p, sample_weight=wt), PR_AUC=average_precision_score(y, p, sample_weight=wt),
                     **{f"recall_top{int(k*100)}pct": h / total for k, h in zip(ks, a["hits"])}))
res = pd.DataFrame(rows); res.to_csv(f"{OUT}/tuning_tree_models_test.csv", index=False)
print("\nTEST (2026 storms, every street):\n" + res.drop(columns="settings").round(4).to_string(index=False))
