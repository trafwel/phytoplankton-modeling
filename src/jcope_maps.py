#!/usr/bin/env python3
"""
jcope_maps.py — spatial MHW-response maps (no ML model needed): where and when heatwaves hit
phytoplankton hardest, and whether the impact is intensifying.

Figures:
  A. figures/jcope_suppression_seasonal.png — per-cell phy_z(MHW) - phy_z(calm), 4 panels
     (DJF/MAM/JJA/SON): the seasonality of suppression, geographically.
  B. figures/jcope_suppression_decadal.png  — early (1993-2005) vs late (2006-2018) vs the
     DIFFERENCE (late-early): where the heatwave impact is intensifying.

Runs on the Mac (reads t + phy surface off HD-PGF-A once). Memory bounded by lat chunking.
    python src/jcope_maps.py
"""
from __future__ import annotations
import os, glob, time, datetime as dt, warnings, numpy as np
warnings.filterwarnings("ignore")

NLAT, NLON, KSURF = 356, 362, 9
HALFWIN, SMOOTH, PCT = 5, 31, 90; MIN_DUR, MAX_GAP = 5, 2; NCHUNK = 8
SEAS = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}; SNAME=["DJF","MAM","JJA","SON"]
SPLIT_YEAR = 2006
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

def find_data():
    for c in sorted(glob.glob("/Volumes/*/JCOPE_full")):
        if glob.glob(f"{c}/t/t.*.npy"): return c
    raise SystemExit("ERROR: JCOPE_full not found — is HD-PGF-A connected?")
DATA = find_data()
tfiles = sorted(glob.glob(f"{DATA}/t/t.*.npy")); pfiles = sorted(glob.glob(f"{DATA}/phy/phy.*.npy"))
dates = [dt.datetime.strptime(os.path.basename(f).split(".")[1], "%Y%m%d") for f in tfiles]
D = len(dates)
doy = np.array([min(d.timetuple().tm_yday,365) for d in dates])
year = np.array([d.year for d in dates]); sea = np.array([SEAS[d.month] for d in dates])
log(f"{D} days {dates[0].date()}..{dates[-1].date()}")

def band(files, rows):
    out = np.empty((D, rows.stop-rows.start, NLON), np.float32)
    for i, f in enumerate(files): out[i] = np.load(f, mmap_mode="r")[KSURF, rows, :]
    return out
def smooth_doy(a):
    pad = SMOOTH//2; ap = np.concatenate([a[-pad:], a, a[:pad]], 0); k = np.ones(SMOOTH)/SMOOTH
    return np.apply_along_axis(lambda m: np.convolve(m, k, "valid"), 0, ap)
def detect_mhw(exceed):
    T, C = exceed.shape; INF = 10**7
    dp = np.empty((T, C)); d = np.full(C, INF, float)
    for t in range(T): d = np.where(exceed[t], 0, d+1); dp[t] = d
    dn = np.empty((T, C)); d = np.full(C, INF, float)
    for t in range(T-1, -1, -1): d = np.where(exceed[t], 0, d+1); dn[t] = d
    mhw = exceed | ((~exceed) & (dp+dn-1 <= MAX_GAP) & (dp < INF) & (dn < INF))
    rl = np.empty((T, C)); r = np.zeros(C)
    for t in range(T): r = np.where(mhw[t], r+1, 0); rl[t] = r
    valid = np.zeros((T, C), bool); carry = np.zeros(C, bool)
    for t in range(T-1, -1, -1):
        end = mhw[t] & (True if t == T-1 else ~mhw[t+1])
        carry = np.where(end, rl[t] >= MIN_DUR, carry); carry = np.where(~mhw[t], False, carry)
        valid[t] = carry
    return valid

# accumulators: per-cell sums of phy anomaly for MHW/calm, split by season and by period
sumA2 = 0.0; nA = 0
sm_seas = np.zeros((4, NLAT, NLON)); nm_seas = np.zeros((4, NLAT, NLON))
sc_seas = np.zeros((4, NLAT, NLON)); nc_seas = np.zeros((4, NLAT, NLON))
sm_per = np.zeros((2, NLAT, NLON)); nm_per = np.zeros((2, NLAT, NLON))
sc_per = np.zeros((2, NLAT, NLON)); nc_per = np.zeros((2, NLAT, NLON))
period = (year >= SPLIT_YEAR).astype(int)                      # 0=early, 1=late

bounds = np.linspace(0, NLAT, NCHUNK+1).astype(int)
for ci in range(NCHUNK):
    rows = slice(bounds[ci], bounds[ci+1]); nr = rows.stop-rows.start
    T = band(tfiles, rows); P = band(pfiles, rows)
    thr = np.full((366, nr, NLON), np.nan, np.float32)
    pclim = np.full((366, nr, NLON), np.nan, np.float32)
    for dd in range(1, 366):
        w = np.abs(((doy-dd+182) % 365)-182) <= HALFWIN
        thr[dd] = np.nanpercentile(T[w], PCT, axis=0); pclim[dd] = np.nanmean(P[w], axis=0)
    thr = smooth_doy(thr[1:])
    valid = detect_mhw(((T > thr[np.clip(doy,1,365)-1]) & np.isfinite(T)).reshape(D, nr*NLON)).reshape(D, nr, NLON)
    A = P - pclim[np.clip(doy,1,365)]
    fin = np.isfinite(A) & np.isfinite(T)
    sumA2 += np.nansum(A[fin]**2); nA += fin.sum()
    for si in range(D):
        s = sea[si]; pr = period[si]; a = A[si]; v = valid[si] & fin[si]; c = (~valid[si]) & fin[si]
        sm_seas[s, rows] += np.where(v, a, 0); nm_seas[s, rows] += v
        sc_seas[s, rows] += np.where(c, a, 0); nc_seas[s, rows] += c
        sm_per[pr, rows] += np.where(v, a, 0); nm_per[pr, rows] += v
        sc_per[pr, rows] += np.where(c, a, 0); nc_per[pr, rows] += c
    log(f"chunk {ci+1}/{NCHUNK} done")

std = np.sqrt(sumA2/nA)
def supp(sm, nm, sc, nc):
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (sm/nm - sc/nc)/std
    r[(nm < 20) | (nc < 20)] = np.nan; return r

lonD = lambda i: 107.9583 + (144+i)*0.083333; latD = lambda j: 10.4583 + j*0.083333
EXT = [lonD(0), lonD(NLON-1), latD(0), latD(NLAT-1)]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# Figure A — seasonal suppression
figA, ax = plt.subplots(1, 4, figsize=(19, 5))
for s in range(4):
    m = supp(sm_seas[s], nm_seas[s], sc_seas[s], nc_seas[s])
    ax[s].set_facecolor("0.82")
    im = ax[s].imshow(m, origin="lower", extent=EXT, cmap="RdBu_r", vmin=-.6, vmax=.6, aspect="auto")
    ax[s].axhline(28, color="k", ls="--", lw=.8); ax[s].set_title(SNAME[s], weight="bold")
    ax[s].set_xlabel("lon °E"); ax[s].set_ylabel("lat °N" if s==0 else "")
figA.colorbar(im, ax=ax, shrink=.7, label="phy_z(MHW) − phy_z(calm)  (blue=suppressed)")
figA.suptitle("JCOPE · marine-heatwave phytoplankton suppression by season · 1993–2018", fontsize=12)
figA.savefig("figures/jcope_suppression_seasonal.png", dpi=130, bbox_inches="tight")
log("saved figures/jcope_suppression_seasonal.png")

# Figure B — decadal
early = supp(sm_per[0], nm_per[0], sc_per[0], nc_per[0])
late = supp(sm_per[1], nm_per[1], sc_per[1], nc_per[1])
figB, ax = plt.subplots(1, 3, figsize=(15, 5.2))
for a, m, ttl, cm, vlim in [(ax[0], early, f"early (1993–{SPLIT_YEAR-1})", "RdBu_r", .6),
                            (ax[1], late, f"late ({SPLIT_YEAR}–2018)", "RdBu_r", .6),
                            (ax[2], late-early, "difference (late − early)", "PuOr_r", .4)]:
    a.set_facecolor("0.82")
    im = a.imshow(m, origin="lower", extent=EXT, cmap=cm, vmin=-vlim, vmax=vlim, aspect="auto")
    a.axhline(28, color="k", ls="--", lw=.8); a.set_title(ttl, weight="bold")
    a.set_xlabel("lon °E"); a.set_ylabel("lat °N")
    plt.colorbar(im, ax=a, shrink=.8)
figB.suptitle("JCOPE · is heatwave suppression intensifying? (more negative in 'difference' = stronger suppression now)", fontsize=11)
figB.tight_layout(rect=[0,0,1,0.94]); figB.savefig("figures/jcope_suppression_decadal.png", dpi=130)
log("saved figures/jcope_suppression_decadal.png")
