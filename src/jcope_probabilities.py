#!/usr/bin/env python3
"""
jcope_probabilities.py — turn the improved classifier into a trustworthy PROBABILISTIC
collapse forecast: calibrated P(collapse), P(normal), P(bloom) per cell-day, validated.

Pipeline:
  1. Load the cached feature table (built by jcope_holdout.py -> JCOPE Data/percell_table.npz).
  2. Out-of-fold (year-blocked) probabilities so each cell-day is scored by a model that never
     saw its year. Each fold's RF is wrapped in ISOTONIC calibration (CalibratedClassifierCV)
     so "70% collapse" really means ~70% observed collapse.
  3. Validate: reliability diagram (predicted vs observed frequency), Brier score, log-loss.
  4. Save the calibrated model (fit on all data) -> models/jcope_collapse_rf.joblib, and an
     example probability table -> JCOPE Data/collapse_probabilities_sample.csv.

Runs fast on the Mac once the cache exists (seconds to load, ~few min to fit).
    python src/jcope_holdout.py        # first, to build the cache (once)
    python src/jcope_probabilities.py
"""
from __future__ import annotations
import os, warnings, numpy as np, joblib
os.environ["PYTHONWARNINGS"] = "ignore"; warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, brier_score_loss

CACHE = "JCOPE Data/percell_table.npz"
FEATS = ["tS","tS_l7","tS_l14","tS_l28","dS","dS_l7","dS_l14","dS_l28","d50","strat","sinD","cosD"]
EDGES = [-0.75, 0.75]; NAMES = {0:"collapse", 1:"normal", 2:"bloom"}
if not os.path.exists(CACHE):
    raise SystemExit(f"ERROR: {CACHE} not found. Run `python src/jcope_holdout.py` first to build it.")

T = dict(np.load(CACHE))
zc = lambda a: a/np.nanstd(a)
X = np.column_stack([zc(T[f]) for f in FEATS]).astype(np.float32)
y = np.digitize(zc(T["phy"]), EDGES).astype(int)
yr = T["yr"].astype(int)
print(f"loaded {y.size:,} cell-days | class balance " + " ".join(f"{NAMES[n]} {(y==n).mean():.0%}" for n in (0,1,2)))

def base_rf():
    return RandomForestClassifier(n_estimators=120, min_samples_leaf=5,
        class_weight="balanced_subsample", n_jobs=-1, random_state=0)

# ---- out-of-fold CALIBRATED probabilities (year-blocked) ----
classes = np.array([0,1,2])
proba = np.zeros((y.size, 3), np.float32)
for k, (tr, te) in enumerate(GroupKFold(5).split(X, y, yr), 1):
    cal = CalibratedClassifierCV(base_rf(), method="isotonic", cv=3)
    cal.fit(X[tr], y[tr])
    p = cal.predict_proba(X[te])
    for j, c in enumerate(cal.classes_): proba[te, c] = p[:, j]
    print(f"  fold {k}: calibrated on {tr.size:,}, scored {te.size:,}")
proba = np.clip(proba, 1e-6, None); proba /= proba.sum(1, keepdims=True)

# ---- validation ----
print(f"\nlog-loss {log_loss(y, proba, labels=classes):.3f}")
for c in (0,1,2):
    bs = brier_score_loss((y==c).astype(int), proba[:, c])
    print(f"  Brier {NAMES[c]:8s} {bs:.4f}  (lower=better; base rate {(y==c).mean():.3f})")

# ---- reliability diagram ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
bins = np.linspace(0, 1, 11)
for c, col, axi in [(0, "#c1121f", ax[0]), (2, "#1a7f37", ax[1])]:
    pc = proba[:, c]; obs = (y == c).astype(float)
    idx = np.digitize(pc, bins) - 1
    mp, mo = [], []
    for b in range(10):
        m = idx == b
        if m.sum() > 50: mp.append(pc[m].mean()); mo.append(obs[m].mean())
    axi.plot([0,1],[0,1], "k--", lw=1, label="perfect")
    axi.plot(mp, mo, "-o", color=col, label=NAMES[c])
    axi.set(title=f"Reliability — P({NAMES[c]})", xlabel="predicted probability",
            ylabel="observed frequency", xlim=(0,1), ylim=(0,1)); axi.legend(frameon=False)
fig.suptitle("JCOPE calibrated collapse/bloom probabilities — reliability (on diagonal = trustworthy)",
             fontsize=11)
fig.tight_layout(rect=[0,0,1,0.95]); os.makedirs("figures", exist_ok=True)
fig.savefig("figures/jcope_reliability.png", dpi=140)
print("\nsaved -> figures/jcope_reliability.png")

# ---- save calibrated model + example probability table ----
final = CalibratedClassifierCV(base_rf(), method="isotonic", cv=3).fit(X, y)
os.makedirs("models", exist_ok=True)
joblib.dump({"model": final, "features": FEATS, "classes": NAMES, "edges": EDGES},
            "models/jcope_collapse_rf.joblib")
print("saved -> models/jcope_collapse_rf.joblib")
import csv
samp = np.random.RandomState(0).choice(y.size, 20, replace=False)
with open("JCOPE Data/collapse_probabilities_sample.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["year","true_class","P_collapse","P_normal","P_bloom"])
    for i in samp:
        w.writerow([yr[i], NAMES[y[i]], f"{proba[i,0]:.3f}", f"{proba[i,1]:.3f}", f"{proba[i,2]:.3f}"])
print("saved -> JCOPE Data/collapse_probabilities_sample.csv")
