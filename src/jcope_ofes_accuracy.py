#!/usr/bin/env python3
"""Matched classifier accuracy: JCOPE (obs) vs OFES (model), Oct 2000-2004, same RF, same
CV, same classes. Features: temp-only (non-circular heatwave test) and temp+nutrients.
Predict phytoplankton collapse/normal/bloom. Random baseline log-loss ~1.10, acc ~0.33."""
import glob, numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, accuracy_score

EDGES = [-0.75, 0.75]
def rf(): return RandomForestClassifier(n_estimators=45, min_samples_leaf=5,
        class_weight="balanced_subsample", n_jobs=-1, random_state=0)
def evaluate(X, y, grp, tag, cap=90_000):
    if len(y) > cap:
        i = np.random.RandomState(0).choice(len(y), cap, replace=False); X, y, grp = X[i], y[i], grp[i]
    classes = np.unique(y); proba = np.zeros((len(y), len(classes))); pred = np.empty(len(y), int)
    for tr, te in GroupKFold(3).split(X, y, grp):
        m = rf().fit(X[tr], y[tr]); p = m.predict_proba(X[te])
        for j, c in enumerate(m.classes_): proba[te, np.searchsorted(classes, c)] = p[:, j]
        pred[te] = m.predict(X[te])
    p = np.clip(proba, 1e-6, None); p /= p.sum(1, keepdims=True)
    print(f"  {tag:30s} log-loss {log_loss(y,p,labels=classes):.3f}  acc {accuracy_score(y,pred):.3f}  n={len(y):,}")

# ---------- JCOPE (laptop pilot, surface, Oct 2000-2004) ----------
J = "JCOPE Data/jcope_pilot"
def load_oct(var):
    c = [np.stack([np.load(f).squeeze() for f in sorted(glob.glob(f"{J}/{var}/{var}.{y}10??.npy"))])
         for y in range(2000, 2005)]
    return np.stack(c)                                   # (5,31,H,W)
tJ, pJ, dJ = load_oct("t"), load_oct("phy"), load_oct("din")
H, W = tJ.shape[-2:]
z = lambda c: (c - c.mean(0, keepdims=True)).reshape(-1, H, W)      # per day-of-Oct anomaly
zs = lambda a: a/np.nanstd(a)
tJz, pJz, dJz = zs(z(tJ)), zs(z(pJ)), zs(z(dJ))
yr = np.repeat(np.arange(2000, 2005), 31).repeat(H*W)
mask = np.isfinite(tJz)&np.isfinite(pJz)&np.isfinite(dJz)
tJz, pJz, dJz, yr = tJz.ravel(), pJz.ravel(), dJz.ravel(), yr
m = mask.ravel()
yj = np.digitize(pJz.ravel()[m], EDGES); gj = yr[m]
Xt = tJz.ravel()[m][:,None]; Xtn = np.c_[tJz.ravel()[m], dJz.ravel()[m]]
print("=== JCOPE (observation-constrained) ===")
evaluate(Xt, yj, gj, "temp only -> phy")
evaluate(Xtn, yj, gj, "temp + nutrients -> phy")

# ---------- OFES (parquet, Oct 2000-2004) ----------
df = pq.read_table("OFES Data/model_table_corrected.parquet",
                   columns=["t","temp_z","ntr_z","phy_z","year"], filters=[("t","<",155)]).to_pandas()
df = df.dropna(subset=["temp_z","ntr_z","phy_z"])
yo = np.digitize(df.phy_z.values, EDGES); go = df.year.values
print("=== OFES-BGC (free-running model) ===")
evaluate(df[["temp_z"]].values, yo, go, "temp only -> phy")
evaluate(df[["temp_z","ntr_z"]].values, yo, go, "temp + nutrients -> phy")
print("\n(temp-only is the non-circular heatwave test; temp+nutrients is inflated for OFES by")
print(" model circularity — OFES computes phytoplankton FROM nutrients.)")
