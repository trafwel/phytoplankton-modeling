#!/usr/bin/env python3
"""
jcope_depth.py — VERTICAL structure of the marine-heatwave -> phytoplankton response.
Marine heatwaves are surface-intensified; this asks whether the phytoplankton suppression
is surface-trapped or extends with depth, and whether the deep chlorophyll maximum
compensates (suppression weakening or reversing below the surface).

Method: detect MHW on SURFACE temperature (Hobday, as in jcope_hobday.py). Then, at EACH of
the 10 extracted depth levels (0 to -200 m), compute the phytoplankton composite
phy_z(MHW) - phy_z(calm) -> a suppression PROFILE, split north/south and by season.

Runs on the Mac (reads the depth files off HD-PGF-A). Memory bounded by latitude chunking.
    python src/jcope_depth.py
"""
from __future__ import annotations
import os, glob, time, datetime as dt, numpy as np
from collections import defaultdict

def find_data():
    for c in sorted(glob.glob("/Volumes/*/JCOPE_full")):
        if glob.glob(f"{c}/t/t.*.npy"): return c
    raise SystemExit("ERROR: JCOPE_full not found — is HD-PGF-A connected?")
DATA = find_data()
NLAT, NLON, NZ = 356, 362, 10
DEPTHS = np.array([-200,-150,-125,-100,-75,-50,-30,-20,-10,0])   # level idx 0..9 (surf=9)
LAT0_DEG, DLAT, NORTH_DEG = 10.4583, 0.083333, 28.0
HALFWIN, SMOOTH, PCT = 5, 31, 90
MIN_DUR, MAX_GAP = 5, 2
NCHUNK = 8
SEAS = {12:"DJF",1:"DJF",2:"DJF",3:"MAM",4:"MAM",5:"MAM",6:"JJA",7:"JJA",8:"JJA",9:"SON",10:"SON",11:"SON"}
SEAS_LIST = ["DJF","MAM","JJA","SON"]
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

tfiles = sorted(glob.glob(f"{DATA}/t/t.*.npy"))
pfiles = sorted(glob.glob(f"{DATA}/phy/phy.*.npy"))
dates = [dt.datetime.strptime(os.path.basename(f).split(".")[1], "%Y%m%d") for f in tfiles]
D = len(dates)
doy = np.array([min(d.timetuple().tm_yday, 365) for d in dates])
seas = np.array([SEAS[d.month] for d in dates])
lat_deg = LAT0_DEG + np.arange(NLAT)*DLAT
north_row = lat_deg >= NORTH_DEG
log(f"{D} days {dates[0].date()}..{dates[-1].date()} | depths {list(DEPTHS)}")

def band_tsurf(rows):
    out = np.empty((D, rows.stop-rows.start, NLON), np.float32)
    for i, f in enumerate(tfiles): out[i] = np.load(f, mmap_mode="r")[-1, rows, :]
    return out
def band_pdepth(rows):
    out = np.empty((D, NZ, rows.stop-rows.start, NLON), np.float32)
    for i, f in enumerate(pfiles): out[i] = np.load(f, mmap_mode="r")[:, rows, :]
    return out

def smooth_doy(a):
    pad = SMOOTH//2
    ap = np.concatenate([a[-pad:], a, a[:pad]], 0); k = np.ones(SMOOTH)/SMOOTH
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

acc = defaultdict(lambda: [0.0, 0])          # (k, band, season, state) -> [sumA, n]
sA2 = np.zeros(NZ); nA = np.zeros(NZ)         # per-depth anomaly variance -> std

bounds = np.linspace(0, NLAT, NCHUNK+1).astype(int)
for ci in range(NCHUNK):
    rows = slice(bounds[ci], bounds[ci+1]); nr = rows.stop-rows.start
    T = band_tsurf(rows); P = band_pdepth(rows)
    # MHW on surface temp
    thr = np.full((366, nr, NLON), np.nan, np.float32)
    for dd in range(1, 366):
        w = np.abs(((doy-dd+182) % 365)-182) <= HALFWIN
        thr[dd] = np.nanpercentile(T[w], PCT, axis=0)
    thr = smooth_doy(thr[1:])
    exceed = (T > thr[np.clip(doy,1,365)-1]) & np.isfinite(T)
    valid = detect_mhw(exceed.reshape(D, nr*NLON)).reshape(D, nr, NLON)
    bandrow = north_row[rows]; band2d = np.broadcast_to(bandrow[:,None], (nr, NLON))
    for k in range(NZ):
        Pk = P[:, k]
        pclim = np.full((366, nr, NLON), np.nan, np.float32)
        for dd in range(1, 366):
            w = np.abs(((doy-dd+182) % 365)-182) <= HALFWIN
            pclim[dd] = np.nanmean(Pk[w], axis=0)
        A = Pk - pclim[np.clip(doy,1,365)]
        fin = np.isfinite(A) & np.isfinite(T)
        sA2[k] += np.nansum(A[fin]**2); nA[k] += fin.sum()
        for bname, bmask in (("north", band2d), ("south", ~band2d)):
            for s in SEAS_LIST:
                dm = (seas == s)
                base = fin & bmask[None] & dm[:,None,None]
                vm = base & valid; cm = base & ~valid
                acc[(k,bname,s,"mhw")][0] += np.nansum(np.where(vm, A, 0)); acc[(k,bname,s,"mhw")][1] += vm.sum()
                acc[(k,bname,s,"calm")][0] += np.nansum(np.where(cm, A, 0)); acc[(k,bname,s,"calm")][1] += cm.sum()
    log(f"chunk {ci+1}/{NCHUNK} done")

std = np.sqrt(sA2/np.maximum(nA,1))
def drop(k, band, season):
    bands = ("north", "south") if band == "ALL" else (band,)
    seasons = SEAS_LIST if season == "ALL" else (season,)
    def m(state):
        s = sum(acc[(k,b,ss,state)][0] for b in bands for ss in seasons)
        n = sum(acc[(k,b,ss,state)][1] for b in bands for ss in seasons)
        return s/n if n else np.nan
    return (m("mhw") - m("calm"))/std[k]

print("\n=== MHW-day phytoplankton suppression (σ) vs DEPTH ===")
print(f"{'depth(m)':>9s} {'ALL':>8s} {'north':>8s} {'south':>8s}")
for k in range(NZ-1, -1, -1):
    print(f"{DEPTHS[k]:>9d} {drop(k,'ALL','ALL'):>+8.3f} {drop(k,'north','ALL'):>+8.3f} {drop(k,'south','ALL'):>+8.3f}")
print("(negative = phytoplankton lower during MHWs; watch for weakening/reversal with depth)")

# ---- figures ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.makedirs("figures", exist_ok=True)
Y0, Y1 = dates[0].year, dates[-1].year; TAG = f"{Y0}-{Y1}"
fig, ax = plt.subplots(1, 2, figsize=(13, 6))
# A: profile ALL/north/south
for band, col in [("ALL","0.2"),("north","#1c4f80"),("south","#c1121f")]:
    ax[0].plot([drop(k,band,"ALL") for k in range(NZ)], DEPTHS, "-o", color=col, label=band)
ax[0].axvline(0, color="0.6", lw=.8); ax[0].legend(frameon=False)
ax[0].set(title="A. Suppression profile", xlabel="phy change on MHW days (σ)", ylabel="depth (m)")
# B: depth x season (ALL bands)
M = np.array([[drop(k,"ALL",s) for s in SEAS_LIST] for k in range(NZ)])
im = ax[1].imshow(M, origin="upper", aspect="auto", cmap="RdBu_r", vmin=-.6, vmax=.6,
                  extent=[0,4,DEPTHS[0],DEPTHS[-1]])
ax[1].set_xticks(np.arange(4)+.5); ax[1].set_xticklabels(SEAS_LIST)
ax[1].set(title="B. Suppression by depth × season (σ)", ylabel="depth (m)")
plt.colorbar(im, ax=ax[1], shrink=.85, label="phy change on MHW days (σ)  (blue=suppressed)")
fig.suptitle(f"JCOPE · vertical structure of the heatwave→phytoplankton response · {TAG}", fontsize=11)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig("figures/jcope_mhw_depth.png", dpi=140)
print("\nsaved -> figures/jcope_mhw_depth.png")
