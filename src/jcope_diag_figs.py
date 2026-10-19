#!/usr/bin/env python3
"""
jcope_diag_figs.py — two poster figures from the forecast cache:
  1. figures/jcope_reliability_season.png — reliability of P(collapse) by season (month labels).
  2. figures/jcope_threshold.png          — recall & precision vs. the warning threshold (fine).
Recomputes calibrated out-of-fold probabilities (year-blocked). Runs from the cache (~15 min).
    python src/jcope_diag_figs.py
"""
from __future__ import annotations
import os, warnings, numpy as np
os.environ["PYTHONWARNINGS"] = "ignore"; warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold

CACHE = "JCOPE Data/forecast_table_L7.npz"
FEATS = ["tS","tS_l7","tS_l14","tS_l28","dS","dS_l7","dS_l14","dS_l28","d50","strat","sinD","cosD"]
EDGES = [-0.75, 0.75]
SEAS_LABEL = {0:"Dec–Feb", 1:"Mar–May", 2:"Jun–Aug", 3:"Sep–Nov"}
SEAS_COL   = {0:"#1c4f80", 1:"#1a7f37", 2:"#c1121f", 3:"#e08a1e"}
if not os.path.exists(CACHE): raise SystemExit(f"ERROR: {CACHE} not found — run src/jcope_forecast.py first.")

T = dict(np.load(CACHE)); zc = lambda a: a/np.nanstd(a)
X = np.column_stack([zc(T[f]) for f in FEATS]).astype(np.float32)
y = np.digitize(zc(T["phyF"]), EDGES).astype(int); yr = T["yr"].astype(int); se = T["se"].astype(int)
def rf(): return RandomForestClassifier(n_estimators=120, min_samples_leaf=5,
        class_weight="balanced_subsample", n_jobs=-1, random_state=0)
proba = np.zeros((y.size, 3))
for k,(tr,te) in enumerate(GroupKFold(5).split(X,y,yr),1):
    cal = CalibratedClassifierCV(rf(), method="isotonic", cv=3).fit(X[tr], y[tr])
    p = cal.predict_proba(X[te])
    for j,c in enumerate(cal.classes_): proba[te,c]=p[:,j]
    print(f"  fold {k} done", flush=True)
pcol = np.clip(proba[:,0], 0, 1); ycol = (y==0).astype(int)
np.savez("JCOPE Data/diag_arrays.npz", pcol=pcol, ycol=ycol, se=se)   # cache for instant figure re-tweaks

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":12})
bins = np.linspace(0,1,11)

# ---- Figure 1: reliability by season ----
fig, ax = plt.subplots(figsize=(7,6.5))
ax.plot([0,1],[0,1],"k--",lw=1,label="perfect")
print("\nreliability by season (predicted -> observed collapse frequency):")
NB = 6   # equal-count (quantile) bins -> every season gets the same number of points
for s in range(4):
    m = se==s; ps=pcol[m]; os_=ycol[m]
    edges = np.quantile(ps, np.linspace(0,1,NB+1)); edges[-1] += 1e-9
    mp,mo=[],[]
    for b in range(NB):
        bm=(ps>=edges[b])&(ps<edges[b+1])
        if bm.sum()>0: mp.append(ps[bm].mean()); mo.append(os_[bm].mean())
    ax.plot(mp,mo,"-o",ms=5,color=SEAS_COL[s],label=SEAS_LABEL[s])
    print(f"  {SEAS_LABEL[s]}: " + " ".join(f"{a:.2f}->{b:.2f}" for a,b in zip(mp,mo)))
ax.set(title="Reliability of the collapse forecast, by season",
       xlabel="predicted P(collapse)", ylabel="observed collapse frequency", xlim=(0,0.8), ylim=(0,0.8))
ax.legend(frameon=False, title="season")
fig.tight_layout(); fig.savefig("figures/jcope_reliability_season.png", dpi=150)
print("saved -> figures/jcope_reliability_season.png")

# ---- Figure 2: threshold sweep (fine) ----
thr = np.arange(0.10, 0.71, 0.05); rec=[]; pre=[]
for t in thr:
    f = pcol > t; tp=(f & (ycol==1)).sum()
    rec.append(tp/max(ycol.sum(),1)); pre.append(tp/max(f.sum(),1))
fig2, ax = plt.subplots(figsize=(8,5.5))
ax.plot(thr,rec,"-o",ms=4,color="#1c4f80",lw=2,label="Collapses caught (recall)")
ax.plot(thr,pre,"-o",ms=4,color="#c1121f",lw=2,label="Warnings correct (precision)")
ax.axhline(ycol.mean(),color="0.6",ls="--",lw=1); ax.text(0.68,ycol.mean()+0.02,f"random guess {ycol.mean():.2f}",fontsize=9,color="0.4",ha="right")
ax.set(title="Effect of the warning threshold", xlabel="collapse-probability threshold  P(collapse) >",
       ylabel="fraction", ylim=(0,1)); ax.legend(frameon=False,loc="upper center")
fig2.tight_layout(); fig2.savefig("figures/jcope_threshold.png", dpi=150)
print("saved -> figures/jcope_threshold.png")
print("\nthreshold  recall  precision"); [print(f"  {t:.2f}    {r:.3f}   {p:.3f}") for t,r,p in zip(thr,rec,pre)]
