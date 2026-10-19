#!/usr/bin/env python3
"""
jcope_forecast_maps.py — model-based spatial products from the saved 7-day forecast:
  A. figures/jcope_pcollapse_map.png   — P(collapse within 7 days) over the grid for TARGET_DATE
     (+ the SST anomaly that day, for context). The operational forecast product.
  B. figures/jcope_skill_map.png        — per-2° tile forecast accuracy: WHERE the forecast works.
  C. figures/jcope_risk_amplification.png — per-tile mean P(collapse) after warm vs calm days,
     and their difference: WHERE heatwaves most raise collapse risk.

Applies models/jcope_forecast_7d.joblib. Features are z-scored with the SAME per-feature std as
training (recovered from JCOPE Data/forecast_table_L7.npz). Reads the record once; skill/risk use
a day STRIDE to bound compute; the date map uses the full grid for TARGET_DATE.
    python src/jcope_forecast_maps.py
"""
from __future__ import annotations
import os, glob, time, datetime as dt, warnings, numpy as np, joblib
os.environ["PYTHONWARNINGS"] = "ignore"; warnings.filterwarnings("ignore")

NLAT, NLON, KSURF, K50 = 356, 362, 9, 5
HALFWIN, NCHUNK, EDGES = 5, 16, [-0.75, 0.75]; LAGS = (7, 14, 28); LEAD = 7
TILE = 24; STRIDE = 5; TARGET_DATE = "20160815"        # a summer 2016 heatwave day
FEATS = ["tS","tS_l7","tS_l14","tS_l28","dS","dS_l7","dS_l14","dS_l28","d50","strat","sinD","cosD"]
MODEL = "models/jcope_forecast_7d.joblib"; FCACHE = "JCOPE Data/forecast_table_L7.npz"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

M = joblib.load(MODEL); model = M["model"]; cidx = list(model.classes_).index(0)   # collapse col
Z = np.load(FCACHE); fstd = {f: float(np.nanstd(Z[f])) for f in FEATS}              # training scale
log("model + feature scale loaded")

def find_data():
    for c in sorted(glob.glob("/Volumes/*/JCOPE_full")):
        if glob.glob(f"{c}/t/t.*.npy"): return c
    raise SystemExit("ERROR: JCOPE_full not found.")
DATA = find_data()
tf = sorted(glob.glob(f"{DATA}/t/t.*.npy")); pf = sorted(glob.glob(f"{DATA}/phy/phy.*.npy"))
df = sorted(glob.glob(f"{DATA}/din/din.*.npy"))
dates = [os.path.basename(f).split(".")[1] for f in tf]
D = len(dates); doy = np.array([min(dt.datetime.strptime(d,"%Y%m%d").timetuple().tm_yday,365) for d in dates])
sinD, cosD = np.sin(2*np.pi*doy/365), np.cos(2*np.pi*doy/365)
tgt = dates.index(TARGET_DATE) if TARGET_DATE in dates else D//2
log(f"{D} days; target {dates[tgt]}")

def band(files, rows, levels):
    outs = [np.empty((D, rows.stop-rows.start, NLON), np.float32) for _ in levels]
    for i, f in enumerate(files):
        a = np.load(f, mmap_mode="r")
        for j, k in enumerate(levels): outs[j][i] = a[k, rows, :]
    return outs
def anom(X):
    clim = np.full((366,)+X.shape[1:], np.nan, np.float32)
    for dd in range(1, 366):
        w = np.abs(((doy-dd+182) % 365)-182) <= HALFWIN
        clim[dd] = np.nanmean(X[w], axis=0)
    return X - clim[np.clip(doy, 1, 365)]
def lag(A, L): o = np.full_like(A, np.nan); o[L:] = A[:-L]; return o

CH, CW = NLAT//TILE + 1, NLON//TILE + 1; NT = CH*CW
n_f=np.zeros(NT); ok_f=np.zeros(NT)                              # skill (flat tiles)
pw_f=np.zeros(NT); nw_f=np.zeros(NT); pc_f=np.zeros(NT); nc_f=np.zeros(NT)  # risk
pmap = np.full((NLAT, NLON), np.nan)                             # date P(collapse) map
tmap = np.full((NLAT, NLON), np.nan)                            # date SST anomaly map
# phy std for target class: use the forecast cache's phyF std (same standardization)
phyF_std = float(np.nanstd(Z["phyF"]))

bounds = np.linspace(0, NLAT, NCHUNK+1).astype(int)
for ci in range(NCHUNK):
    rows = slice(bounds[ci], bounds[ci+1]); nr = rows.stop-rows.start
    tS, t50 = band(tf, rows, [KSURF, K50]); dS, d50 = band(df, rows, [KSURF, K50]); (pS,) = band(pf, rows, [KSURF])
    aTS, aDS, aD50, aStr, aPhy = anom(tS), anom(dS), anom(d50), anom(tS-t50), anom(pS)
    feat = {"tS":aTS, "dS":aDS, "d50":aD50, "strat":aStr}
    for L in LAGS: feat[f"tS_l{L}"] = lag(aTS, L); feat[f"dS_l{L}"] = lag(aDS, L)
    feat["sinD"] = np.broadcast_to(sinD[:,None,None], aTS.shape); feat["cosD"] = np.broadcast_to(cosD[:,None,None], aTS.shape)
    aPhyF = np.full_like(aPhy, np.nan); aPhyF[:-LEAD] = aPhy[LEAD:]
    warm = aTS > np.nanpercentile(aTS, 90)
    latidx = rows.start + np.arange(nr)
    tl = ((latidx//TILE)[:, None]*CW + (np.arange(NLON)//TILE)[None, :]).ravel()   # tile id per cell
    def predict_day(si):
        cols = [feat[f][si].ravel()/fstd[f] for f in FEATS]
        Xd = np.column_stack(cols); good = np.isfinite(Xd).all(1)
        p = np.full(nr*NLON, np.nan)
        if good.any(): p[good] = model.predict_proba(Xd[good])[:, cidx]
        return p.reshape(nr, NLON), good.reshape(nr, NLON)

    # skill + risk over strided days (vectorized tile accumulation)
    for si in range(0, D-LEAD, STRIDE):
        yf = aPhyF[si]; fin = np.isfinite(yf)
        if not fin.any(): continue
        P, good = predict_day(si); m = (good & fin).ravel()
        if not m.any(): continue
        pcol = P.ravel(); truth = (np.digitize(yf.ravel()/phyF_std, EDGES) == 0)
        correct = (pcol >= 0.5) == truth; wm = warm[si].ravel()
        tm = tl[m]
        n_f  += np.bincount(tm, minlength=NT)
        ok_f += np.bincount(tm, weights=correct[m].astype(float), minlength=NT)
        mw = m & wm; mc = m & ~wm
        pw_f += np.bincount(tl[mw], weights=pcol[mw], minlength=NT); nw_f += np.bincount(tl[mw], minlength=NT)
        pc_f += np.bincount(tl[mc], weights=pcol[mc], minlength=NT); nc_f += np.bincount(tl[mc], minlength=NT)
    # target-date maps
    P, good = predict_day(tgt); pmap[rows] = np.where(good, P, np.nan)
    tmap[rows] = np.where(np.isfinite(aTS[tgt]), aTS[tgt], np.nan)
    log(f"chunk {ci+1}/{NCHUNK} done")

# ---- figures ----
lonD = lambda i: 107.9583 + (144+i)*0.083333; latD = lambda j: 10.4583 + j*0.083333
EXT = [lonD(0), lonD(NLON-1), latD(0), latD(NLAT-1)]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# A. P(collapse) date map + SST anomaly
figA, ax = plt.subplots(1, 2, figsize=(14, 6))
for a in ax: a.set_facecolor("0.82")
im0 = ax[0].imshow(tmap, origin="lower", extent=EXT, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
ax[0].set_title(f"SST anomaly, {dates[tgt]}", loc="left", weight="bold"); plt.colorbar(im0, ax=ax[0], shrink=.8, label="°C anomaly")
im1 = ax[1].imshow(pmap, origin="lower", extent=EXT, cmap="inferno", vmin=0, vmax=0.6, aspect="auto")
ax[1].set_title(f"Forecast P(collapse within 7 days) from {dates[tgt]}", loc="left", weight="bold")
plt.colorbar(im1, ax=ax[1], shrink=.8, label="P(collapse)")
for a in ax: a.axhline(28, color="k", ls="--", lw=.8); a.set_xlabel("lon °E"); a.set_ylabel("lat °N")
figA.suptitle("JCOPE 7-day collapse forecast — operational product", fontsize=12)
figA.tight_layout(rect=[0,0,1,0.95]); figA.savefig("figures/jcope_pcollapse_map.png", dpi=130); log("saved P(collapse) map")

# tile grids -> figures B, C (reshape flat accumulators to CHxCW)
tEXT = EXT
n_t = n_f.reshape(CH, CW); ok_t = ok_f.reshape(CH, CW)
nw = nw_f.reshape(CH, CW); pw_s = pw_f.reshape(CH, CW)
nc = nc_f.reshape(CH, CW); pc_s = pc_f.reshape(CH, CW)
skill = np.where(n_t > 200, ok_t/np.maximum(n_t, 1), np.nan)
pw = np.where(nw > 50, pw_s/np.maximum(nw, 1), np.nan)
pc = np.where(nc > 50, pc_s/np.maximum(nc, 1), np.nan)
figB, a = plt.subplots(figsize=(7.5,6)); a.set_facecolor("0.82")
im = a.imshow(skill, origin="lower", extent=tEXT, cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")
a.axhline(28,color="w",ls="--",lw=.8); a.set(title="Forecast accuracy (collapse vs not) per 2° tile", xlabel="lon °E", ylabel="lat °N")
plt.colorbar(im, ax=a, shrink=.85, label="accuracy"); figB.tight_layout(); figB.savefig("figures/jcope_skill_map.png", dpi=130); log("saved skill map")

figC, ax = plt.subplots(1, 3, figsize=(16, 5.2))
for a, g, ttl, vlim, cm in [(ax[0], pc, "P(collapse) after CALM", 0.4, "inferno"),
                            (ax[1], pw, "P(collapse) after WARM", 0.4, "inferno"),
                            (ax[2], pw-pc, "amplification (warm − calm)", 0.15, "Reds")]:
    a.set_facecolor("0.82")
    im = a.imshow(g, origin="lower", extent=tEXT, cmap=cm, vmin=(0 if cm=="inferno" else -vlim), vmax=vlim, aspect="auto")
    a.axhline(28,color="w",ls="--",lw=.8); a.set(title=ttl, xlabel="lon °E", ylabel="lat °N"); plt.colorbar(im, ax=a, shrink=.8)
figC.suptitle("JCOPE · where marine heatwaves most raise 7-day collapse probability", fontsize=12)
figC.tight_layout(rect=[0,0,1,0.94]); figC.savefig("figures/jcope_risk_amplification.png", dpi=130); log("saved risk-amplification map")
