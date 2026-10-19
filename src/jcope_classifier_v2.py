#!/usr/bin/env python3
"""
jcope_classifier_v2.py — IMPROVED phytoplankton classifier on JCOPE, with in-depth
feature engineering, tested against the surface-only baseline (temp+nutrients, log-loss
0.934 / acc 0.54 all-season).

WHY each feature (see the thesis notes / COLLATED_RESULTS §3.6):
  * LAGS (temp & nutrient anomalies at -7,-14,-28 d): phytoplankton biomass is time-
    integrated — blooms/collapses accumulate over days-weeks. Same-day snapshot has no memory.
  * SUBSURFACE nutrients (DIN at -50 m): surface DIN is the *depleted* tail (partly an effect
    of the phytoplankton we predict); the -50 m reservoir is the *cause* that can be mixed up.
  * STRATIFICATION (T_surface - T_-50m): the gate on nutrient supply. Strong stratification =
    a lid starving the surface. Our depth result showed the action is subsurface.
  * SEASON (sin/cos of day-of-year): predictability is seasonal; let the model specialize.

Two experiments:
  (1) PER-CELL (1/12 deg, daily): baseline vs improved feature set. Isolates missing-physics gain.
  (2) COARSE-GRAINED (~1 deg, weekly): improved features on aggregated fields. Averaging out
      mesoscale eddies isolates how much error is irreducible noise vs missing physics.

Runs on the Mac (reads t/phy/din depth files off HD-PGF-A). Memory bounded by lat chunking.
    python src/jcope_classifier_v2.py
"""
from __future__ import annotations
import os, glob, time, datetime as dt, warnings, numpy as np
warnings.filterwarnings("ignore")     # silence the benign sklearn/joblib parallel warning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, accuracy_score

def find_data():
    for c in sorted(glob.glob("/Volumes/*/JCOPE_full")):
        if glob.glob(f"{c}/t/t.*.npy"): return c
    raise SystemExit("ERROR: JCOPE_full not found — is HD-PGF-A connected?")
DATA = find_data()
NLAT, NLON = 356, 362
KSURF, K50 = 9, 5                 # depth indices: 9 = 0 m (surface), 5 = -50 m
HALFWIN = 5; NCHUNK = 8; EDGES = [-0.75, 0.75]; LAGS = (7, 14, 28)
TARGET_ROWS = 1_200_000
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
sinD = np.sin(2*np.pi*doy/365); cosD = np.cos(2*np.pi*doy/365)
log(f"{D} days {dates[0].date()}..{dates[-1].date()}")

def anomaly(X):
    """per-(day-of-year, cell) climatology (+-HALFWIN pooled) -> anomaly. Removes seasonal
    cycle so the model sees departures, not the climatological bloom timing."""
    clim = np.full((366,)+X.shape[1:], np.nan, np.float32)
    for dd in range(1, 366):
        w = np.abs(((doy-dd+182) % 365)-182) <= HALFWIN
        clim[dd] = np.nanmean(X[w], axis=0)
    return X - clim[np.clip(doy, 1, 365)]

def lag(A, L):
    """shift anomaly cube by L days along time: out[t] = A[t-L] (first L days -> nan)."""
    out = np.full_like(A, np.nan); out[L:] = A[:-L]; return out

def evaluate(X, y, grp, se, name, feat_names=None):
    classes = np.unique(y); proba = np.zeros((len(y), len(classes))); pred = np.empty(len(y), int)
    def rf(): return RandomForestClassifier(n_estimators=120, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=-1, random_state=0)
    for tr, te in GroupKFold(5).split(X, y, grp):
        m = rf().fit(X[tr], y[tr]); p = m.predict_proba(X[te])
        for j, c in enumerate(m.classes_): proba[te, np.searchsorted(classes, c)] = p[:, j]
        pred[te] = m.predict(X[te])
    p = np.clip(proba, 1e-6, None); p /= p.sum(1, keepdims=True)
    print(f"\n{name}")
    print(f"  {'overall':6s}  log-loss {log_loss(y,p,labels=classes):.3f}  acc {accuracy_score(y,pred):.3f}")
    for s in range(4):
        m = se == s
        print(f"  {SNAME[s]:6s}  log-loss {log_loss(y[m],p[m],labels=classes):.3f}  acc {accuracy_score(y[m],pred[m]):.3f}")
    if feat_names is not None:
        imp = RandomForestClassifier(n_estimators=120, min_samples_leaf=5,
              class_weight="balanced_subsample", n_jobs=-1, random_state=0).fit(X, y).feature_importances_
        order = np.argsort(imp)[::-1]
        print("  importances: " + ", ".join(f"{feat_names[i]} {imp[i]:.2f}" for i in order))

# =================== EXPERIMENT 1 — per-cell (1/12 deg, daily) ===================
FEATS = ["tS","tS_l7","tS_l14","tS_l28","dS","dS_l7","dS_l14","dS_l28","d50","strat","sinD","cosD"]
cols = {k: [] for k in FEATS + ["phy","yr","se"]}
rng = np.random.RandomState(0)
bounds = np.linspace(0, NLAT, NCHUNK+1).astype(int)
def band_levels(files, rows, levels):
    """read several depth levels from each file in ONE open (halves I/O vs per-level)."""
    outs = [np.empty((D, rows.stop-rows.start, NLON), np.float32) for _ in levels]
    for i, f in enumerate(files):
        a = np.load(f, mmap_mode="r")
        for j, k in enumerate(levels): outs[j][i] = a[k, rows, :]
    return outs
for ci in range(NCHUNK):
    rows = slice(bounds[ci], bounds[ci+1]); nr = rows.stop-rows.start
    tS, t50 = band_levels(tfiles, rows, [KSURF, K50])
    dS, d50 = band_levels(dfiles, rows, [KSURF, K50])
    (pS,) = band_levels(pfiles, rows, [KSURF])
    aTS = anomaly(tS); aDS = anomaly(dS); aD50 = anomaly(d50)
    aStrat = anomaly(tS - t50); aPhy = anomaly(pS)
    feat = {"tS":aTS, "dS":aDS, "d50":aD50, "strat":aStrat}
    for L in LAGS: feat[f"tS_l{L}"] = lag(aTS, L); feat[f"dS_l{L}"] = lag(aDS, L)
    S = np.broadcast_to(sinD[:,None,None], aTS.shape); C = np.broadcast_to(cosD[:,None,None], aTS.shape)
    feat["sinD"] = S; feat["cosD"] = C
    fin = np.isfinite(aPhy)
    for f in FEATS: fin = fin & np.isfinite(feat[f])
    idx = np.where(fin.ravel())[0]
    take = rng.choice(idx, min(TARGET_ROWS//NCHUNK, idx.size), replace=False)
    for f in FEATS: cols[f].append(feat[f].ravel()[take])
    cols["phy"].append(aPhy.ravel()[take])
    cols["yr"].append(np.broadcast_to(year[:,None,None], aTS.shape).ravel()[take])
    cols["se"].append(np.broadcast_to(sea[:,None,None], aTS.shape).ravel()[take])
    log(f"chunk {ci+1}/{NCHUNK}: +{take.size:,}")

T = {k: np.concatenate(v) for k, v in cols.items()}
def zc(a): return a/np.nanstd(a)
X = np.column_stack([zc(T[f]) for f in FEATS])
y = np.digitize(zc(T["phy"]), EDGES); grp = T["yr"]; se = T["se"]
log(f"per-cell table {y.size:,} rows | balance " + " ".join(f"{n}:{(y==n).mean():.0%}" for n in (0,1,2)))
print("\n================= EXPERIMENT 1: PER-CELL (1/12 deg, daily) =================")
bi = [FEATS.index("tS"), FEATS.index("dS")]
evaluate(X[:, bi], y, grp, se, "baseline: temp + nutrients (2 features)")
evaluate(X, y, grp, se, "IMPROVED: + lags + subsurface + stratification + season", FEATS)

# =================== EXPERIMENT 2 — coarse-grained (~1 deg, weekly) ===================
B = 12                              # 12 cells at 1/12 deg ~= 1 deg boxes
def coarsen(a):
    ny = (NLAT//B)*B; nx = (NLON//B)*B
    return np.nanmean(a[:ny,:nx].reshape(ny//B, B, nx//B, B), axis=(1,3))
CH, CW = NLAT//B, NLON//B
def load_coarse_levels(files, levels):
    outs = [np.empty((D, CH, CW), np.float32) for _ in levels]
    for i, f in enumerate(files):
        a = np.load(f, mmap_mode="r")
        for j, k in enumerate(levels): outs[j][i] = coarsen(a[k])
    return outs
log("coarse-grain: loading 1-deg fields...")
ctS, ct50 = load_coarse_levels(tfiles, [KSURF, K50])
cdS, cd50 = load_coarse_levels(dfiles, [KSURF, K50]); (cpS,) = load_coarse_levels(pfiles, [KSURF])
# weekly means (group consecutive 7-day blocks)
nw = D//7
def weekly(a):
    return np.nanmean(a[:nw*7].reshape(nw, 7, *a.shape[1:]), axis=1)
wt, wt50, wd, wd50, wp = map(weekly, (ctS, ct50, cdS, cd50, cpS))
wyear = year[:nw*7].reshape(nw,7)[:,0]; wdoy = doy[:nw*7].reshape(nw,7)[:,0]
def doy_season(d):   # standard meteorological seasons by day-of-year
    return 0 if (d < 60 or d >= 335) else 1 if d < 152 else 2 if d < 244 else 3
wsea = np.array([doy_season(d) for d in wdoy])
# anomalies per week-of-year, weekly lags (1,2,4 weeks ~ 7,14,28 d)
def wanom(A):
    woy = ((wdoy-1)//7) % 52
    clim = np.stack([np.nanmean(A[woy==k], 0) for k in range(52)])
    return A - clim[woy]
def wlag(A, L): o=np.full_like(A,np.nan); o[L:]=A[:-L]; return o
aWt, aWd, aWd50, aWstrat, aWp = wanom(wt), wanom(wd), wanom(wd50), wanom(wt-wt50), wanom(wp)
FC = {"tS":aWt,"tS_l7":wlag(aWt,1),"tS_l14":wlag(aWt,2),"tS_l28":wlag(aWt,4),
      "dS":aWd,"dS_l7":wlag(aWd,1),"dS_l14":wlag(aWd,2),"dS_l28":wlag(aWd,4),
      "d50":aWd50,"strat":aWstrat,
      "sinD":np.broadcast_to(np.sin(2*np.pi*wdoy/365)[:,None,None],aWt.shape),
      "cosD":np.broadcast_to(np.cos(2*np.pi*wdoy/365)[:,None,None],aWt.shape)}
finc = np.isfinite(aWp)
for f in FEATS: finc = finc & np.isfinite(FC[f])
ff = finc.ravel()
Xc = np.column_stack([zc(FC[f].ravel()[ff]) for f in FEATS])
yc = np.digitize(zc(aWp.ravel()[ff]), EDGES)
grc = np.broadcast_to(wyear[:,None,None], aWp.shape).ravel()[ff]
sec = np.broadcast_to(wsea[:,None,None], aWp.shape).ravel()[ff]
log(f"coarse table {yc.size:,} rows")
print("\n============= EXPERIMENT 2: COARSE-GRAINED (~1 deg, weekly) =============")
evaluate(Xc[:, bi], yc, grc, sec, "baseline: temp + nutrients")
evaluate(Xc, yc, grc, sec, "IMPROVED features", FEATS)
