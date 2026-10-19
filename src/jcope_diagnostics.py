#!/usr/bin/env python3
"""
jcope_diagnostics.py — verify what ACCURACY hides for the 7-day forecast: does it actually
catch collapses, and are the probabilities meaningful?

Loads the forecast table (JCOPE Data/forecast_table_L7.npz), recomputes year-blocked,
isotonic-calibrated out-of-fold probabilities, then reports:
  * class balance + majority-class baseline (context for accuracy)
  * confusion matrix (row-normalized) + per-class precision/recall/F1
  * COLLAPSE discrimination: ROC-AUC, PR-AUC, and recall/precision at warning thresholds
  * per-season reliability, sharpness, and a Brier-score decomposition (reliability/resolution)
Figure: figures/jcope_diagnostics.png (confusion, PR curve, reliability, sharpness).

Runs fast from the cache (no file re-reads).
    python src/jcope_diagnostics.py
"""
from __future__ import annotations
import os, warnings, numpy as np
os.environ["PYTHONWARNINGS"] = "ignore"; warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score,
                             average_precision_score, precision_recall_curve, brier_score_loss)

CACHE = "JCOPE Data/forecast_table_L7.npz"
FEATS = ["tS","tS_l7","tS_l14","tS_l28","dS","dS_l7","dS_l14","dS_l28","d50","strat","sinD","cosD"]
EDGES = [-0.75, 0.75]; NAMES = ["collapse","normal","bloom"]; SNAME=["DJF","MAM","JJA","SON"]
if not os.path.exists(CACHE):
    raise SystemExit(f"ERROR: {CACHE} not found. Run src/jcope_forecast.py first.")

T = dict(np.load(CACHE)); zc = lambda a: a/np.nanstd(a)
X = np.column_stack([zc(T[f]) for f in FEATS]).astype(np.float32)
y = np.digitize(zc(T["phyF"]), EDGES).astype(int); yr = T["yr"].astype(int); se = T["se"].astype(int)
base = np.bincount(y)/y.size
print("class balance:", {NAMES[i]: f"{base[i]:.1%}" for i in range(3)})
print(f"majority-class ('normal') baseline accuracy = {base[1]:.3f}  (any model must beat THIS)")

def base_rf(): return RandomForestClassifier(n_estimators=120, min_samples_leaf=5,
        class_weight="balanced_subsample", n_jobs=-1, random_state=0)
proba = np.zeros((y.size, 3))
for k,(tr,te) in enumerate(GroupKFold(5).split(X,y,yr),1):
    cal = CalibratedClassifierCV(base_rf(), method="isotonic", cv=3).fit(X[tr], y[tr])
    p = cal.predict_proba(X[te])
    for j,c in enumerate(cal.classes_): proba[te,c]=p[:,j]
    print(f"  fold {k} done", flush=True)
proba = np.clip(proba,1e-9,None); proba/=proba.sum(1,keepdims=True); pred = proba.argmax(1)

# ---- confusion + per-class ----
cm = confusion_matrix(y, pred, labels=[0,1,2]); cmn = cm/cm.sum(1,keepdims=True)
print("\nconfusion matrix (row-normalized; rows=true, cols=pred):")
print("           " + "  ".join(f"{n:>9s}" for n in NAMES))
for i in range(3): print(f"  {NAMES[i]:9s} " + "  ".join(f"{cmn[i,j]:9.3f}" for j in range(3)))
print("\nper-class precision / recall / f1:")
print(classification_report(y, pred, labels=[0,1,2], target_names=NAMES, digits=3, zero_division=0))

# ---- collapse discrimination ----
ycol = (y==0).astype(int); pcol = proba[:,0]
print(f"COLLAPSE discrimination:  ROC-AUC {roc_auc_score(ycol,pcol):.3f} | "
      f"PR-AUC {average_precision_score(ycol,pcol):.3f}  (base rate {ycol.mean():.3f})")
print("early-warning trade-off — flag collapse when P(collapse) > threshold:")
print(f"  {'thr':>5s} {'recall(caught)':>14s} {'precision':>10s} {'%flagged':>9s}")
for thr in (0.2,0.3,0.4,0.5):
    f = pcol > thr; tp = (f & (ycol==1)).sum()
    rec = tp/max(ycol.sum(),1); prec = tp/max(f.sum(),1)
    print(f"  {thr:5.1f} {rec:14.3f} {prec:10.3f} {f.mean():9.1%}")

# ---- Brier decomposition (Murphy) for collapse, 10 bins ----
bins = np.linspace(0,1,11); obar = ycol.mean(); rel=res=0.0; N=y.size
for b in range(10):
    m = (pcol>=bins[b]) & (pcol<bins[b+1]) if b<9 else (pcol>=bins[b])
    if m.sum()==0: continue
    pk, ok, nk = pcol[m].mean(), ycol[m].mean(), m.sum()
    rel += nk/N*(pk-ok)**2; res += nk/N*(ok-obar)**2
print(f"\nBrier decomposition (collapse): reliability {rel:.4f} (low=well-calibrated) | "
      f"resolution {res:.4f} (high=informative) | uncertainty {obar*(1-obar):.4f}")
print(f"  Brier = {brier_score_loss(ycol,pcol):.4f}")

# ---- figure ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 2, figsize=(13, 11))
im = ax[0,0].imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax[0,0].set(xticks=[0,1,2], yticks=[0,1,2], xticklabels=NAMES, yticklabels=NAMES,
            title="A. Confusion matrix (row-normalized)", xlabel="predicted", ylabel="true")
for i in range(3):
    for j in range(3): ax[0,0].text(j,i,f"{cmn[i,j]:.2f}",ha="center",va="center",
                                    color="white" if cmn[i,j]>.5 else "black")
prec, rec, _ = precision_recall_curve(ycol, pcol)
ax[0,1].plot(rec, prec, color="#c1121f"); ax[0,1].axhline(ycol.mean(), color="0.6", ls="--", label=f"base rate {ycol.mean():.2f}")
ax[0,1].set(title=f"B. Collapse precision–recall (PR-AUC {average_precision_score(ycol,pcol):.3f})",
            xlabel="recall (collapses caught)", ylabel="precision", xlim=(0,1), ylim=(0,1)); ax[0,1].legend(frameon=False)
for s in range(4):
    m = se==s; ps=pcol[m]; os_=ycol[m]; mp,mo=[],[]
    for b in range(10):
        bm=(ps>=bins[b])&(ps<bins[b+1]) if b<9 else (ps>=bins[b])
        if bm.sum()>50: mp.append(ps[bm].mean()); mo.append(os_[bm].mean())
    ax[1,0].plot(mp,mo,"-o",ms=3,label=SNAME[s])
ax[1,0].plot([0,1],[0,1],"k--"); ax[1,0].set(title="C. Reliability by season — P(collapse)",
            xlabel="predicted", ylabel="observed", xlim=(0,1), ylim=(0,1)); ax[1,0].legend(frameon=False)
ax[1,1].hist(pcol[ycol==0], bins=30, alpha=.6, density=True, label="no collapse", color="#1c4f80")
ax[1,1].hist(pcol[ycol==1], bins=30, alpha=.6, density=True, label="collapse", color="#c1121f")
ax[1,1].set(title="D. Sharpness — P(collapse) by outcome", xlabel="predicted P(collapse)", ylabel="density"); ax[1,1].legend(frameon=False)
fig.suptitle("7-day collapse forecast — verification diagnostics", fontsize=13)
fig.tight_layout(rect=[0,0,1,0.97]); os.makedirs("figures",exist_ok=True)
fig.savefig("figures/jcope_diagnostics.png", dpi=140)
print("\nsaved -> figures/jcope_diagnostics.png")
