#!/usr/bin/env python3
"""
jcope_hobday.py — the payoff: ALL-SEASON marine-heatwave -> phytoplankton analysis on the
observation-constrained JCOPE record. Auto-covers WHATEVER is in JCOPE_full/ (so it becomes
the full 1993-2018 once the remaining tars are extracted). Proper version of what the
October-only OFES crop could only approximate.

Steps:
  1. Load surface temperature + phytoplankton for every extracted day (continuous, all seasons).
  2. Hobday climatology + 90th-pct threshold per day-of-year (+-5 day window, 31-day smooth);
     detect MHW = temp > threshold for >=5 consecutive days (gaps <=2 merged).
  3. Phytoplankton anomaly = phy - day-of-year climatology, z-scored region-wide.
  4. Measure suppression: mean phy_z on MHW days vs calm days, OVERALL, BY SEASON, and
     NORTH (>=28N) vs SOUTH. Also MHW frequency by month. Saves a figure.

Runs on the Mac (reads ~6,600 files off HD-PGF-A). Memory bounded by latitude chunking.
    cd "/Users/lukequasar/Documents/Ocean Project"
    source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate ocean
    python src/jcope_hobday.py
"""
from __future__ import annotations
import os, glob, time, datetime as dt, numpy as np

def find_data():
    """Locate JCOPE_full across /Volumes (drive may remount as HD-PGF-A-1, etc.)."""
    for c in sorted(glob.glob("/Volumes/*/JCOPE_full")):
        if glob.glob(f"{c}/t/t.*.npy"):
            return c
    import os as _os
    vols = ", ".join(sorted(_os.listdir("/Volumes")))
    raise SystemExit(f"ERROR: JCOPE_full with data not found. /Volumes has: {vols}\n"
                     "  Is HD-PGF-A connected? (it may have remounted under a new name.)")
DATA = find_data()
NLAT, NLON = 356, 362
LAT0_DEG, DLAT = 10.4583, 0.083333          # JCOPE lat of crop row 0, spacing
NORTH_DEG = 28.0
HALFWIN, SMOOTH, PCT = 5, 31, 90
MIN_DUR, MAX_GAP = 5, 2
NCHUNK = 6
SEASONS = {12:"DJF",1:"DJF",2:"DJF",3:"MAM",4:"MAM",5:"MAM",
           6:"JJA",7:"JJA",8:"JJA",9:"SON",10:"SON",11:"SON"}
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

# ---- dates (from temp files) ----
tfiles = sorted(glob.glob(f"{DATA}/t/t.*.npy"))
dates = [dt.datetime.strptime(os.path.basename(f).split(".")[1], "%Y%m%d") for f in tfiles]
D = len(dates)
doy = np.array([min(d.timetuple().tm_yday, 365) for d in dates])   # clamp leap 366->365
mon = np.array([d.month for d in dates])
seas = np.array([SEASONS[m] for m in mon])
log(f"{D} days: {dates[0].date()}..{dates[-1].date()}")
lat_deg = LAT0_DEG + np.arange(NLAT)*DLAT
north_row = lat_deg >= NORTH_DEG

pfiles = sorted(glob.glob(f"{DATA}/phy/phy.*.npy"))
def load_band(files, rows):
    """surface (last depth idx) for all days, latitude band `rows`, via mmap so only the
    needed rows are read -> (D, nrow, NLON). Memory stays bounded for the full record."""
    out = np.empty((D, rows.stop-rows.start, NLON), np.float32)
    for i, f in enumerate(files):
        out[i] = np.load(f, mmap_mode="r")[-1, rows, :]
    return out

def smooth_doy(a):
    pad = SMOOTH//2
    ap = np.concatenate([a[-pad:], a, a[:pad]], 0)                  # circular over year
    k = np.ones(SMOOTH)/SMOOTH
    return np.apply_along_axis(lambda m: np.convolve(m, k, "valid"), 0, ap)

def detect_mhw(exceed):
    """(D,C) bool exceedance -> (D,C) MHW bool (>=MIN_DUR run, gaps<=MAX_GAP merged)."""
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
        carry = np.where(end, rl[t] >= MIN_DUR, carry)
        carry = np.where(~mhw[t], False, carry)
        valid[t] = carry
    return valid

# ---- accumulators ----
from collections import defaultdict
acc = defaultdict(lambda: [0.0, 0])          # (season, band, state) -> [sumA, n]
sumA2 = 0.0; nA = 0                          # for global phy-anomaly std
mhw_month = np.zeros((12, 2))               # [month, band] -> [mhw_days, total] via two cols
mhw_cnt = np.zeros((12, 2)); tot_cnt = np.zeros((12, 2))
freqmap = np.zeros((NLAT, NLON)); nmap = np.zeros((NLAT, NLON))
sA_mhw = np.zeros((NLAT, NLON)); nC_mhw = np.zeros((NLAT, NLON))   # per-cell suppression map
sA_calm = np.zeros((NLAT, NLON)); nC_calm = np.zeros((NLAT, NLON))

bounds = np.linspace(0, NLAT, NCHUNK+1).astype(int)
for ci in range(NCHUNK):
    rows = slice(bounds[ci], bounds[ci+1]); nr = rows.stop-rows.start
    T = load_band(tfiles, rows); P = load_band(pfiles, rows)
    C = nr*NLON
    log(f"chunk {ci+1}/{NCHUNK} rows[{rows.start}:{rows.stop}] loaded")
    # day-of-year climatology + threshold (per cell), from all years
    thr = np.full((366, nr, NLON), np.nan, np.float32)
    pclim = np.full((366, nr, NLON), np.nan, np.float32)
    for dd in range(1, 366):
        w = (np.abs(((doy-dd+182) % 365)-182) <= HALFWIN)          # +-HALFWIN circular
        thr[dd] = np.nanpercentile(T[w], PCT, axis=0)
        pclim[dd] = np.nanmean(P[w], axis=0)
    thr[0] = thr[365-1]
    thr = smooth_doy(thr[1:])                                       # (365,nr,NLON)
    # MHW detection
    thr_day = thr[np.clip(doy, 1, 365)-1]                          # (D,nr,NLON)
    exceed = (T > thr_day) & np.isfinite(T)
    valid = detect_mhw(exceed.reshape(D, C)).reshape(D, nr, NLON)
    # phy anomaly (z later)
    A = P - pclim[np.clip(doy, 1, 365)]                            # (D,nr,NLON)
    fin = np.isfinite(A) & np.isfinite(T)
    sumA2 += np.nansum(A[fin]**2); nA += fin.sum()
    # accumulate composites
    bandrow = north_row[rows]                                       # (nr,)
    band2d = np.broadcast_to(bandrow[:, None], (nr, NLON))
    for si in range(D):
        s = seas[si]; m = mon[si]-1
        a = A[si]; v = valid[si]; f = fin[si]
        for bname, bmask in (("north", band2d), ("south", ~band2d)):
            fm = f & bmask
            mm = fm & v; cm = fm & ~v
            acc[(s, bname, "mhw")][0] += np.nansum(a[mm]); acc[(s, bname, "mhw")][1] += mm.sum()
            acc[(s, bname, "calm")][0] += np.nansum(a[cm]); acc[(s, bname, "calm")][1] += cm.sum()
            acc[("ALL", bname, "mhw")][0] += np.nansum(a[mm]); acc[("ALL", bname, "mhw")][1] += mm.sum()
            acc[("ALL", bname, "calm")][0] += np.nansum(a[cm]); acc[("ALL", bname, "calm")][1] += cm.sum()
            bi = 0 if bname == "north" else 1
            mhw_cnt[m, bi] += mm.sum(); tot_cnt[m, bi] += fm.sum()
    freqmap[rows] += valid.sum(0); nmap[rows] += fin.sum(0)
    vm = valid & fin; cm = (~valid) & fin
    sA_mhw[rows]  += np.nansum(np.where(vm, A, 0), 0); nC_mhw[rows]  += vm.sum(0)
    sA_calm[rows] += np.nansum(np.where(cm, A, 0), 0); nC_calm[rows] += cm.sum(0)
    log(f"chunk {ci+1}/{NCHUNK} done")

std = np.sqrt(sumA2/nA)
def sig(key):
    s, n = acc[key]; return (s/n)/std if n else np.nan
Y0, Y1 = dates[0].year, dates[-1].year; TAG = f"{Y0}-{Y1}"
log(f"phy-anomaly std = {std:.4f} mmol N/m3  (results below in sigma units)")
print(f"\n=== MHW-day phytoplankton suppression (phy_z), {TAG}, all-season ===")
print(f"{'season':6s} {'north drop':>12s} {'south drop':>12s}")
for s in ["ALL","DJF","MAM","JJA","SON"]:
    dn = sig((s,"north","mhw")) - sig((s,"north","calm"))
    ds = sig((s,"south","mhw")) - sig((s,"south","calm"))
    print(f"{s:6s} {dn:>+12.3f} {ds:>+12.3f}")
print("(negative = phytoplankton LOWER during marine heatwaves)")
print("\nMHW frequency by month (% of cell-days):")
freq = 100*mhw_cnt/np.maximum(tot_cnt,1)
print("  month:", " ".join(f"{m:02d}" for m in range(1,13)))
print("  north:", " ".join(f"{freq[m,0]:4.1f}" for m in range(12)))
print("  south:", " ".join(f"{freq[m,1]:4.1f}" for m in range(12)))

# ---- spatial maps ----
with np.errstate(invalid="ignore", divide="ignore"):
    supp_map = ((sA_mhw/nC_mhw) - (sA_calm/nC_calm))/std
    freq_map = 100*freqmap/np.maximum(nmap, 1)
supp_map[nC_mhw < 20] = np.nan; freq_map[nmap < 20] = np.nan
lonD = lambda i: 107.9583 + (144+i)*0.083333
latD = lambda j: 10.4583 + j*0.083333
EXT = [lonD(0), lonD(NLON-1), latD(0), latD(NLAT-1)]

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.makedirs("figures", exist_ok=True)
mo = np.arange(1, 13); seas_list = ["DJF","MAM","JJA","SON"]
dnB = [sig((s,"north","mhw"))-sig((s,"north","calm")) for s in seas_list]
dsB = [sig((s,"south","mhw"))-sig((s,"south","calm")) for s in seas_list]

# Figure 1 — seasonal summary
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].plot(mo, freq[:,0], "-o", color="#1c4f80", label="north ≥28°N")
ax[0].plot(mo, freq[:,1], "-o", color="#c1121f", label="south <28°N")
ax[0].set(title="A. Marine-heatwave frequency by month", xlabel="month",
          ylabel="% of cell-days in a MHW"); ax[0].set_xticks(mo); ax[0].legend(frameon=False)
x = np.arange(4); w = .38
ax[1].bar(x-w/2, dnB, w, color="#1c4f80", label="north")
ax[1].bar(x+w/2, dsB, w, color="#c1121f", label="south")
ax[1].axhline(0, color="0.5", lw=.8); ax[1].set_xticks(x); ax[1].set_xticklabels(seas_list)
ax[1].set(title="B. Phytoplankton change on MHW days (σ), by season",
          ylabel="phy_z(MHW) − phy_z(calm)"); ax[1].legend(frameon=False)
fig.suptitle(f"JCOPE (obs-constrained) · marine heatwaves vs phytoplankton · {TAG}, all seasons "
             "· 10–40°N, 120–150°E", fontsize=11)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig("figures/jcope_mhw_seasonal.png", dpi=140)

# Figure 2 — spatial maps
fig2, ax2 = plt.subplots(1, 2, figsize=(14, 6))
for a in ax2: a.set_facecolor("0.82")
im0 = ax2[0].imshow(freq_map, origin="lower", extent=EXT, cmap="inferno", aspect="auto")
ax2[0].set_title("A. Marine-heatwave frequency (% of days)", loc="left", weight="bold")
plt.colorbar(im0, ax=ax2[0], shrink=.85, label="% days in a MHW")
im1 = ax2[1].imshow(supp_map, origin="lower", extent=EXT, cmap="RdBu_r", vmin=-.6, vmax=.6, aspect="auto")
ax2[1].set_title("B. Phytoplankton change on MHW days (σ)", loc="left", weight="bold")
plt.colorbar(im1, ax=ax2[1], shrink=.85, label="phy_z(MHW) − phy_z(calm)   (blue = suppressed, red = boosted)")
for a in ax2:
    a.axhline(28, color="k", ls="--", lw=1); a.set_xlabel("longitude (°E)"); a.set_ylabel("latitude (°N)")
fig2.suptitle(f"JCOPE · where marine heatwaves strike and how phytoplankton responds · {TAG}", fontsize=11)
fig2.tight_layout(rect=[0,0,1,0.95]); fig2.savefig("figures/jcope_mhw_maps.png", dpi=140)
print("\nsaved -> figures/jcope_mhw_seasonal.png  and  figures/jcope_mhw_maps.png")
