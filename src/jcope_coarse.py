#!/usr/bin/env python3
"""
jcope_coarse.py — the SCALE test (Experiment 2), standalone + cached. Coarse-grains JCOPE to
~1 deg boxes and weekly means, then runs the baseline vs improved classifier. Comparing its
accuracy to the per-cell run (log-loss 0.632 / acc 0.725) decomposes the remaining error:
  * big jump  -> native-resolution error was mostly unpredictable MESOSCALE NOISE (irreducible
                 at pixel scale; real forecastability lives at the regional/weekly scale).
  * little change -> the ceiling is MISSING PHYSICS/drivers, not noise.

Reads all files ONCE, block-averages, and caches the tiny 1-deg weekly cubes to
`JCOPE Data/coarse_cache.npz`. Re-runs load the cache instantly (no file re-reading).
    python src/jcope_coarse.py
"""
from __future__ import annotations
import os, glob, time, datetime as dt, warnings, numpy as np
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, accuracy_score

def find_data():
    for c in sorted(glob.glob("/Volumes/*/JCOPE_full")):
        if glob.glob(f"{c}/t/t.*.npy"): return c
    raise SystemExit("ERROR: JCOPE_full not found — is HD-PGF-A connected?")
NLAT, NLON, KSURF, K50, B = 356, 362, 9, 5, 12
EDGES = [-0.75, 0.75]; SNAME = ["DJF","MAM","JJA","SON"]
CACHE = "JCOPE Data/coarse_cache.npz"
FEATS = ["tS","tS_l7","tS_l14","tS_l28","dS","dS_l7","dS_l14","dS_l28","d50","strat","sinD","cosD"]
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

def build_cache():
    DATA = find_data()
    tfiles = sorted(glob.glob(f"{DATA}/t/t.*.npy"))
    pfiles = sorted(glob.glob(f"{DATA}/phy/phy.*.npy"))
    dfiles = sorted(glob.glob(f"{DATA}/din/din.*.npy"))
    dates = [dt.datetime.strptime(os.path.basename(f).split(".")[1], "%Y%m%d") for f in tfiles]
    D = len(dates); CH, CW = NLAT//B, NLON//B
    doy = np.array([min(d.timetuple().tm_yday, 365) for d in dates]); year = np.array([d.year for d in dates])
    def coarsen(a):
        ny, nx = (NLAT//B)*B, (NLON//B)*B
        return np.nanmean(a[:ny,:nx].reshape(ny//B, B, nx//B, B), axis=(1,3))
    def load(files, levels):
        outs = [np.empty((D, CH, CW), np.float32) for _ in levels]
        for i, f in enumerate(files):
            a = np.load(f, mmap_mode="r")
            for j, k in enumerate(levels): outs[j][i] = coarsen(a[k])
        return outs
    log(f"reading {D} days x3 vars (block-avg to {CH}x{CW}) ...")
    tS, t50 = load(tfiles, [KSURF, K50]); log("  temp done")
    dS, d50 = load(dfiles, [KSURF, K50]); log("  din done")
    (pS,) = load(pfiles, [KSURF]);        log("  phy done")
    np.savez_compressed(CACHE, tS=tS, t50=t50, dS=dS, d50=d50, pS=pS,
                        doy=doy, year=year)
    log(f"cached -> {CACHE}")
    return dict(tS=tS, t50=t50, dS=dS, d50=d50, pS=pS, doy=doy, year=year)

if os.path.exists(CACHE):
    log(f"loading cache {CACHE}"); Z = dict(np.load(CACHE))
else:
    Z = build_cache()
tS, t50, dS, d50, pS = Z["tS"], Z["t50"], Z["dS"], Z["d50"], Z["pS"]
doy, year = Z["doy"], Z["year"]
D = len(doy); nw = D//7

def weekly(a): return np.nanmean(a[:nw*7].reshape(nw, 7, *a.shape[1:]), axis=1)
wt, wt50, wd, wd50, wp = map(weekly, (tS, t50, dS, d50, pS))
wdoy = doy[:nw*7].reshape(nw,7)[:,0]; wyear = year[:nw*7].reshape(nw,7)[:,0]
def dseas(d): return 0 if (d < 60 or d >= 335) else 1 if d < 152 else 2 if d < 244 else 3
wsea = np.array([dseas(d) for d in wdoy])
woy = ((wdoy-1)//7) % 52
def wanom(A):
    clim = np.stack([np.nanmean(A[woy==k], 0) for k in range(52)])
    return A - clim[woy]
def wlag(A, L): o = np.full_like(A, np.nan); o[L:] = A[:-L]; return o
aT, aD, aD50, aStr, aP = wanom(wt), wanom(wd), wanom(wd50), wanom(wt-wt50), wanom(wp)
FC = {"tS":aT,"tS_l7":wlag(aT,1),"tS_l14":wlag(aT,2),"tS_l28":wlag(aT,4),
      "dS":aD,"dS_l7":wlag(aD,1),"dS_l14":wlag(aD,2),"dS_l28":wlag(aD,4),
      "d50":aD50,"strat":aStr,
      "sinD":np.broadcast_to(np.sin(2*np.pi*wdoy/365)[:,None,None], aT.shape),
      "cosD":np.broadcast_to(np.cos(2*np.pi*wdoy/365)[:,None,None], aT.shape)}
fin = np.isfinite(aP)
for f in FEATS: fin = fin & np.isfinite(FC[f])
ff = fin.ravel()
zc = lambda a: a/np.nanstd(a)
X = np.column_stack([zc(FC[f].ravel()[ff]) for f in FEATS])
y = np.digitize(zc(aP.ravel()[ff]), EDGES)
grp = np.broadcast_to(wyear[:,None,None], aP.shape).ravel()[ff]
se = np.broadcast_to(wsea[:,None,None], aP.shape).ravel()[ff]
log(f"coarse table {y.size:,} rows | balance " + " ".join(f"{n}:{(y==n).mean():.0%}" for n in (0,1,2)))

def evaluate(Xf, name):
    cl = np.unique(y); pr = np.zeros((len(y), len(cl))); pd = np.empty(len(y), int)
    def rf(): return RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
            class_weight="balanced_subsample", n_jobs=-1, random_state=0)
    for tr, te in GroupKFold(5).split(Xf, y, grp):
        m = rf().fit(Xf[tr], y[tr]); p = m.predict_proba(Xf[te])
        for j, c in enumerate(m.classes_): pr[te, np.searchsorted(cl, c)] = p[:, j]
        pd[te] = m.predict(Xf[te])
    p = np.clip(pr, 1e-6, None); p /= p.sum(1, keepdims=True)
    print(f"\n{name}")
    print(f"  {'overall':6s}  log-loss {log_loss(y,p,labels=cl):.3f}  acc {accuracy_score(y,pd):.3f}")
    for s in range(4):
        m = se == s
        print(f"  {SNAME[s]:6s}  log-loss {log_loss(y[m],p[m],labels=cl):.3f}  acc {accuracy_score(y[m],pd[m]):.3f}")

print("\n============= COARSE-GRAINED (~1 deg, weekly) — compare to per-cell 0.632/0.725 =============")
bi = [FEATS.index("tS"), FEATS.index("dS")]
evaluate(X[:, bi], "baseline: temp + nutrients")
evaluate(X, "improved: + lags + subsurface + stratification + season")
