#!/usr/bin/env python3
"""
jcope_classifier.py — the ALL-SEASON JCOPE accuracy the matched Oct-2000-04 test couldn't give.
Trains the same random forest (collapse/normal/bloom) on observation-constrained JCOPE across
all seasons, and reports log-loss + accuracy OVERALL and BY SEASON, for:
  - temp only            (non-circular heatwave test)
  - temp + nutrients     (add DIN)
Year-blocked CV. Random baseline log-loss ~1.10.

Builds per-day-of-year anomalies (temp/din/phy), z-scores region-wide, subsamples ~1.5M
cell-days (bounded memory via latitude chunking + mmap), then trains. Runs on the Mac.
    python src/jcope_classifier.py
"""
from __future__ import annotations
import os, glob, time, datetime as dt, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, accuracy_score

def find_data():
    for c in sorted(glob.glob("/Volumes/*/JCOPE_full")):
        if glob.glob(f"{c}/t/t.*.npy"): return c
    raise SystemExit("ERROR: JCOPE_full not found — is HD-PGF-A connected?")
DATA = find_data()
NLAT, NLON = 356, 362
HALFWIN = 5; NCHUNK = 6; EDGES = [-0.75, 0.75]
TARGET_ROWS = 1_500_000
SEAS = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}; SNAME = ["DJF","MAM","JJA","SON"]
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

tfiles = sorted(glob.glob(f"{DATA}/t/t.*.npy"))
pfiles = sorted(glob.glob(f"{DATA}/phy/phy.*.npy"))
dfiles = sorted(glob.glob(f"{DATA}/din/din.*.npy"))
dates = [dt.datetime.strptime(os.path.basename(f).split(".")[1], "%Y%m%d") for f in tfiles]
D = len(dates)
doy = np.array([min(d.timetuple().tm_yday, 365) for d in dates])
year = np.array([d.year for d in dates]); sea = np.array([SEAS[d.month] for d in dates])
log(f"{D} days {dates[0].date()}..{dates[-1].date()}")

def band(files, rows):
    out = np.empty((D, rows.stop-rows.start, NLON), np.float32)
    for i, f in enumerate(files): out[i] = np.load(f, mmap_mode="r")[-1, rows, :]
    return out
def anomaly(X):
    """per-(day-of-year, cell) climatology (+-HALFWIN pooled) -> anomaly."""
    clim = np.full((366,)+X.shape[1:], np.nan, np.float32)
    for dd in range(1, 366):
        w = np.abs(((doy-dd+182) % 365)-182) <= HALFWIN
        clim[dd] = np.nanmean(X[w], axis=0)
    return X - clim[np.clip(doy, 1, 365)]

cols = {k: [] for k in ("t","d","p","yr","se")}
rng = np.random.RandomState(0)
bounds = np.linspace(0, NLAT, NCHUNK+1).astype(int)
for ci in range(NCHUNK):
    rows = slice(bounds[ci], bounds[ci+1]); nr = rows.stop-rows.start
    aT = anomaly(band(tfiles, rows)); aD = anomaly(band(dfiles, rows)); aP = anomaly(band(pfiles, rows))
    fin = np.isfinite(aT) & np.isfinite(aD) & np.isfinite(aP)
    yr3 = np.broadcast_to(year[:,None,None], aT.shape); se3 = np.broadcast_to(sea[:,None,None], aT.shape)
    idx = np.where(fin.ravel())[0]
    take = rng.choice(idx, min(TARGET_ROWS//NCHUNK, idx.size), replace=False)
    cols["t"].append(aT.ravel()[take]); cols["d"].append(aD.ravel()[take]); cols["p"].append(aP.ravel()[take])
    cols["yr"].append(yr3.ravel()[take]); cols["se"].append(se3.ravel()[take])
    log(f"chunk {ci+1}/{NCHUNK}: +{take.size:,} rows")

T = {k: np.concatenate(v) for k, v in cols.items()}
zt = T["t"]/np.std(T["t"]); zd = T["d"]/np.std(T["d"]); zp = T["p"]/np.std(T["p"])
y = np.digitize(zp, EDGES); grp = T["yr"]; se = T["se"]
log(f"table {y.size:,} rows | class balance "
    + " ".join(f"{n}:{(y==n).mean():.0%}" for n in (0,1,2)))

def rf(): return RandomForestClassifier(n_estimators=120, min_samples_leaf=5,
        class_weight="balanced_subsample", n_jobs=-1, random_state=0)
def oof(X):
    classes = np.unique(y); proba = np.zeros((len(y), len(classes))); pred = np.empty(len(y), int)
    for tr, te in GroupKFold(5).split(X, y, grp):
        m = rf().fit(X[tr], y[tr]); p = m.predict_proba(X[te])
        for j, c in enumerate(m.classes_): proba[te, np.searchsorted(classes, c)] = p[:, j]
        pred[te] = m.predict(X[te])
    p = np.clip(proba, 1e-6, None); p /= p.sum(1, keepdims=True)
    return classes, p, pred

def report(name, X):
    cl, p, pred = oof(X)
    print(f"\n{name}")
    print(f"  {'overall':6s}  log-loss {log_loss(y,p,labels=cl):.3f}  acc {accuracy_score(y,pred):.3f}")
    for s in range(4):
        m = se == s
        print(f"  {SNAME[s]:6s}  log-loss {log_loss(y[m],p[m],labels=cl):.3f}  acc {accuracy_score(y[m],pred[m]):.3f}")

Xt = zt[:, None]; Xtn = np.c_[zt, zd]
print("\n=== ALL-SEASON JCOPE classifier (obs-constrained) · random baseline log-loss ~1.10 ===")
report("temp only -> phy", Xt)
report("temp + nutrients -> phy", Xtn)
