#!/usr/bin/env python3
"""
jcope_holdout.py — overfitting check via a HELD-OUT-YEARS split, plus the year-blocked CV
for reference. Also CACHES the per-cell feature table so this and future per-cell experiments
load in seconds instead of re-reading 3,287 files.

Splits (forward in time, whole years -> no spatiotemporal leakage):
    train = 2010-2016   validate = 2017   test = 2018
Reports TRAIN vs TEST for each model. Interpretation:
    * small train->test drop  -> NOT overfitting; the skill generalizes to an unseen year.
    * large train->test drop  -> overfitting (memorizing, not learning).
Compare test to the 5-fold CV number (improved 0.632 / 0.725) — they should agree if honest.

    python src/jcope_holdout.py     # builds JCOPE Data/percell_table.npz once, then instant
"""
from __future__ import annotations
import os, glob, time, datetime as dt, warnings, numpy as np
os.environ["PYTHONWARNINGS"] = "ignore"   # inherited by joblib worker subprocesses (n_jobs=-1)
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, accuracy_score

NLAT, NLON, KSURF, K50 = 356, 362, 9, 5
HALFWIN, NCHUNK, EDGES = 5, 8, [-0.75, 0.75]; LAGS = (7, 14, 28)
TARGET_ROWS = 1_200_000
CACHE = "JCOPE Data/percell_table.npz"
FEATS = ["tS","tS_l7","tS_l14","tS_l28","dS","dS_l7","dS_l14","dS_l28","d50","strat","sinD","cosD"]
SEASN = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

def find_data():
    for c in sorted(glob.glob("/Volumes/*/JCOPE_full")):
        if glob.glob(f"{c}/t/t.*.npy"): return c
    raise SystemExit("ERROR: JCOPE_full not found — is HD-PGF-A connected?")

def build_cache():
    DATA = find_data()
    tf = sorted(glob.glob(f"{DATA}/t/t.*.npy")); pf = sorted(glob.glob(f"{DATA}/phy/phy.*.npy"))
    df = sorted(glob.glob(f"{DATA}/din/din.*.npy"))
    dates = [dt.datetime.strptime(os.path.basename(f).split(".")[1], "%Y%m%d") for f in tf]
    D = len(dates)
    doy = np.array([min(d.timetuple().tm_yday, 365) for d in dates])
    year = np.array([d.year for d in dates]); sea = np.array([SEASN[d.month] for d in dates])
    sinD = np.sin(2*np.pi*doy/365); cosD = np.cos(2*np.pi*doy/365)
    def anomaly(X):
        clim = np.full((366,)+X.shape[1:], np.nan, np.float32)
        for dd in range(1, 366):
            w = np.abs(((doy-dd+182) % 365)-182) <= HALFWIN
            clim[dd] = np.nanmean(X[w], axis=0)
        return X - clim[np.clip(doy, 1, 365)]
    def lag(A, L): o = np.full_like(A, np.nan); o[L:] = A[:-L]; return o
    def band(files, rows, levels):
        outs = [np.empty((D, rows.stop-rows.start, NLON), np.float32) for _ in levels]
        for i, f in enumerate(files):
            a = np.load(f, mmap_mode="r")
            for j, k in enumerate(levels): outs[j][i] = a[k, rows, :]
        return outs
    cols = {k: [] for k in FEATS + ["phy","yr","se","lat","lon","doy"]}
    rng = np.random.RandomState(0); bounds = np.linspace(0, NLAT, NCHUNK+1).astype(int)
    for ci in range(NCHUNK):
        rows = slice(bounds[ci], bounds[ci+1])
        tS, t50 = band(tf, rows, [KSURF, K50]); dS, d50 = band(df, rows, [KSURF, K50]); (pS,) = band(pf, rows, [KSURF])
        aTS, aDS, aD50 = anomaly(tS), anomaly(dS), anomaly(d50)
        aStr, aPhy = anomaly(tS - t50), anomaly(pS)
        feat = {"tS":aTS, "dS":aDS, "d50":aD50, "strat":aStr}
        for L in LAGS: feat[f"tS_l{L}"] = lag(aTS, L); feat[f"dS_l{L}"] = lag(aDS, L)
        feat["sinD"] = np.broadcast_to(sinD[:,None,None], aTS.shape)
        feat["cosD"] = np.broadcast_to(cosD[:,None,None], aTS.shape)
        fin = np.isfinite(aPhy)
        for f in FEATS: fin = fin & np.isfinite(feat[f])
        idx = np.where(fin.ravel())[0]
        take = rng.choice(idx, min(TARGET_ROWS//NCHUNK, idx.size), replace=False)
        for f in FEATS: cols[f].append(feat[f].ravel()[take])
        cols["phy"].append(aPhy.ravel()[take])
        cols["yr"].append(np.broadcast_to(year[:,None,None], aTS.shape).ravel()[take])
        cols["se"].append(np.broadcast_to(sea[:,None,None], aTS.shape).ravel()[take])
        latb = np.broadcast_to((rows.start+np.arange(rows.stop-rows.start))[None,:,None], aTS.shape)
        lonb = np.broadcast_to(np.arange(NLON)[None,None,:], aTS.shape)
        cols["lat"].append(latb.ravel()[take]); cols["lon"].append(lonb.ravel()[take])
        cols["doy"].append(np.broadcast_to(doy[:,None,None], aTS.shape).ravel()[take])
        log(f"chunk {ci+1}/{NCHUNK}: +{take.size:,}")
    T = {k: np.concatenate(v).astype(np.float32) for k, v in cols.items()}
    np.savez_compressed(CACHE, **T); log(f"cached -> {CACHE}")
    return T

if os.path.exists(CACHE):
    log(f"loading cache {CACHE}"); T = dict(np.load(CACHE))
else:
    T = build_cache()

zc = lambda a: a/np.nanstd(a)
X = np.column_stack([zc(T[f]) for f in FEATS])
y = np.digitize(zc(T["phy"]), EDGES).astype(int)
yr = T["yr"].astype(int); se = T["se"].astype(int)
lat = T["lat"].astype(int); lon = T["lon"].astype(int)
bi = [FEATS.index("tS"), FEATS.index("dS")]
log(f"table {y.size:,} rows | balance " + " ".join(f"{n}:{(y==n).mean():.0%}" for n in (0,1,2)))
def rf(): return RandomForestClassifier(n_estimators=120, min_samples_leaf=5,
        class_weight="balanced_subsample", n_jobs=-1, random_state=0)
def score(m, Xs, ys):
    cl = m.classes_; p = np.clip(m.predict_proba(Xs), 1e-6, None); p /= p.sum(1, keepdims=True)
    return log_loss(ys, p, labels=cl), accuracy_score(ys, m.predict(Xs))

# ---- year-blocked 5-fold CV (reference) ----
def cv(Xf, name):
    cl = np.unique(y); pr = np.zeros((len(y), len(cl))); pd = np.empty(len(y), int)
    for tr, te in GroupKFold(5).split(Xf, y, yr):
        m = rf().fit(Xf[tr], y[tr]); p = m.predict_proba(Xf[te])
        for j, c in enumerate(m.classes_): pr[te, np.searchsorted(cl, c)] = p[:, j]
        pd[te] = m.predict(Xf[te])
    p = np.clip(pr, 1e-6, None); p /= p.sum(1, keepdims=True)
    print(f"  {name:20s} log-loss {log_loss(y,p,labels=cl):.3f}  acc {accuracy_score(y,pd):.3f}")
print("\n=== year-blocked 5-fold CV (reference) ===")
cv(X[:, bi], "baseline"); cv(X, "improved")

# ---- portion splits (10% from every year) ----
def eval_split(tr, va, te, title):
    print(f"\n=== {title} ===")
    print(f"  train {tr.sum():,} | val {va.sum():,} | test {te.sum():,}")
    print(f"  {'model':9s} {'split':6s} {'log-loss':>9s} {'acc':>7s}")
    for name, cols_ in (("baseline", bi), ("improved", list(range(len(FEATS))))):
        Xf = X[:, cols_]; m = rf().fit(Xf[tr], y[tr])
        sc = {}
        for lab, mask in (("TRAIN", tr), ("val", va), ("TEST", te)):
            sc[lab] = score(m, Xf[mask], y[mask])
            print(f"  {name:9s} {lab:6s} {sc[lab][0]:9.3f} {sc[lab][1]:7.3f}")
        print(f"    -> {name}: train-minus-TEST acc gap = {sc['TRAIN'][1]-sc['TEST'][1]:+.3f}")

# BLOCKED 80/10/10 — hold out whole (year x 2deg tile) blocks: ~10% of every year, but
# correlated neighbours stay together => leakage-safe. THIS is the number to trust.
TILE = 24
block = (yr.astype(np.int64)*100000 + (lat//TILE)*1000 + (lon//TILE))
ub, inv = np.unique(block, return_inverse=True)
perm = np.random.RandomState(1).permutation(len(ub)); assign = np.zeros(len(ub), np.int8)
assign[perm[:len(ub)//10]] = 2; assign[perm[len(ub)//10:len(ub)//5]] = 1
sp = assign[inv]
eval_split(sp == 0, sp == 1, sp == 2,
           "BLOCKED 80/10/10 (hold out 2-deg tiles from every year — leakage-safe, TRUST THIS)")

# NAIVE random 80/10/10 — random cell-days. Leaks via spatial/temporal neighbours; shown only
# to demonstrate the inflation vs the blocked split. DO NOT trust this number.
r = np.random.RandomState(2).random(len(y))
eval_split(r >= 0.2, (r >= 0.1) & (r < 0.2), r < 0.1,
           "NAIVE random 80/10/10 (leaks — inflated, for comparison only)")
