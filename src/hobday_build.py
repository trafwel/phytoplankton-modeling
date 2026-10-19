#!/usr/bin/env python3
"""
hobday_build.py — detect marine heatwaves (Hobday et al. 2016) over the crop region
from the FULL-YEAR OFES2_NP physics temperature, then cache the MHW state aligned to the
biology dates (October 2000-2019 + September 2018).

Why OFES2_NP: the coupled crop's temp is bit-identical to OFES2_NP[0,300:600,200:500]
(verified), so its continuous daily record IS the temperature the biology experienced.
The biology crop is October-only, but OFES2_NP is all-season -> we can build a proper
seasonal climatology + 90th-percentile threshold instead of an October-only hack.

Scope: we read Sep 1 - Nov 30 each year (2000-2019). That buffer around October is all a
seasonally-varying threshold for October days needs (+-5-day window) and lets >=5-day
events be detected with correct onset/decline around the month boundaries.

Method (Hobday):
  - climatology mean + 90th-pct threshold per calendar day, pooled over an 11-day window
    (+-5 days) across all 20 years, then smoothed with a 31-day running mean.
  - MHW day = temp > threshold; runs of >=5 consecutive days are events; gaps of <=2 days
    between events are merged.

Output: OFES Data/hobday_mhw.npz  with, for October (620 days, ordered year then day) and
for Sep 2018 (30 days): mhw flag (int8), intensity=temp-threshold (float32), and the
climatology/threshold maps. Aligns cell-for-cell with model_table_corrected.parquet
(lat/lon indices 0..299).

Run on the Mac (drive local):
    cd "/Users/lukequasar/Documents/Ocean Project"
    source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate ocean
    python src/hobday_build.py
Reads are the cost (~1820 files). READ-ONLY on the drive.
"""
from __future__ import annotations
import os, glob, time, numpy as np

# ---- locate the physics temp dir (drive may remount under a new /Volumes name) ----
def find_np():
    for c in sorted(glob.glob("/Volumes/*/OFES2_NP/DATA_3D/daily")):
        if glob.glob(f"{c}/y2000/temp/temp.*.dta"):
            return c
    raise SystemExit("ERROR: OFES2_NP/DATA_3D/daily not found — is HD-SHU3 connected?")

NPDIR = find_np()
NZ, NYf, NXf = 105, 880, 1900          # full-domain shape of OFES2_NP files
LAT0, LAT1, LON0, LON1 = 300, 600, 200, 500   # the crop window (10-40N, 120-150E)
NY = NX = 300
YEARS = list(range(2000, 2020))
MONTHS = ["09", "10", "11"]            # Sep-Nov buffer; 30+31+30 = 91 days, no leap day
NDAY = 91
OCT0, OCT1 = 30, 61                    # ordinal range of October within Sep1..Nov30 (30..60)
FILL = -5e33
HALFWIN = 5                            # +-5 day percentile window (11-day)
SMOOTH = 31                            # running-mean smoothing of clim/threshold
PCT = 90
MIN_DUR, MAX_GAP = 5, 2               # Hobday event rules
NCHUNK = 6                             # latitude chunks to bound memory (~1 GB/chunk)
OUT = "OFES Data/hobday_mhw.npz"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

def day_list(year):
    """Ordered (ordinal, path) for Sep1..Nov30 of `year` (91 contiguous days)."""
    dpm = {"09": 30, "10": 31, "11": 30}
    out, k = [], 0
    for mo in MONTHS:
        for da in range(1, dpm[mo] + 1):
            out.append((k, f"{NPDIR}/y{year}/temp/temp.{mo}.{da:02d}.{year}.dta"))
            k += 1
    return out

def read_window(path, lat_slice):
    """Surface temp over the crop lon window, for a latitude sub-slice, via memmap
    (reads only the needed rows, not the whole 700 MB file)."""
    mm = np.memmap(path, dtype=">f4", mode="r", shape=(NZ, NYf, NXf))
    a = np.array(mm[0, LAT0 + lat_slice.start: LAT0 + lat_slice.stop, LON0:LON1], dtype=np.float32)
    del mm
    a[a <= FILL] = np.nan
    return a

def detect_mhw(exceed):
    """Hobday run rules on a (NDAY, C) boolean exceedance array -> (NDAY, C) MHW bool."""
    T, C = exceed.shape
    INF = 10**6
    # distance to previous / next exceedance day (for gap filling)
    dprev = np.full(C, INF, float); dp = np.empty((T, C))
    for t in range(T):
        dprev = np.where(exceed[t], 0, dprev + 1); dp[t] = dprev
    dnext = np.full(C, INF, float); dn = np.empty((T, C))
    for t in range(T - 1, -1, -1):
        dnext = np.where(exceed[t], 0, dnext + 1); dn[t] = dnext
    gap_len = dp + dn - 1
    bounded = (dp < INF) & (dn < INF)
    mhw = exceed | ((~exceed) & bounded & (gap_len <= MAX_GAP))   # merge short gaps
    # run length of contiguous MHW days
    run = np.zeros(C); runlen = np.empty((T, C))
    for t in range(T):
        run = np.where(mhw[t], run + 1, 0); runlen[t] = run
    # keep only runs that reach >= MIN_DUR (backfill from each run's end)
    valid = np.zeros((T, C), bool); carry = np.zeros(C, bool)
    for t in range(T - 1, -1, -1):
        end = mhw[t] & (True if t == T - 1 else ~mhw[t + 1])
        carry = np.where(end, runlen[t] >= MIN_DUR, carry)
        carry = np.where(~mhw[t], False, carry)
        valid[t] = carry
    return valid

# accumulators (full 300x300), assembled per latitude chunk
mhw_oct = np.zeros((len(YEARS) * 31, NY, NX), np.int8)      # 620 October days
int_oct = np.zeros((len(YEARS) * 31, NY, NX), np.float32)
mhw_sep18 = np.zeros((30, NY, NX), np.int8)                 # Sep 2018
int_sep18 = np.zeros((30, NY, NX), np.float32)
thr_map = np.full((NDAY, NY, NX), np.nan, np.float32)       # threshold per ordinal (for reference)

bounds = np.linspace(0, NY, NCHUNK + 1).astype(int)
for ci in range(NCHUNK):
    ls = slice(bounds[ci], bounds[ci + 1]); nlat = ls.stop - ls.start
    # cube: (year, day, lat_chunk, lon)
    cube = np.full((len(YEARS), NDAY, nlat, NX), np.nan, np.float32)
    for yi, y in enumerate(YEARS):
        for k, p in day_list(y):
            cube[yi, k] = read_window(p, ls)
    log(f"chunk {ci+1}/{NCHUNK} lat[{ls.start}:{ls.stop}] read")

    # --- Hobday climatology + 90th-pct threshold per ordinal (pool +-HALFWIN across years)
    thr = np.full((NDAY, nlat, NX), np.nan, np.float32)
    clim = np.full((NDAY, nlat, NX), np.nan, np.float32)
    for j in range(NDAY):
        lo, hi = max(0, j - HALFWIN), min(NDAY, j + HALFWIN + 1)
        window = cube[:, lo:hi].reshape(-1, nlat, NX)          # (years*windowdays, lat, lon)
        thr[j] = np.nanpercentile(window, PCT, axis=0)
        clim[j] = np.nanmean(window, axis=0)
    # 31-day running-mean smoothing along ordinal
    def smooth(a):
        pad = SMOOTH // 2
        ap = np.concatenate([a[:1].repeat(pad, 0), a, a[-1:].repeat(pad, 0)], axis=0)
        ker = np.ones(SMOOTH) / SMOOTH
        return np.apply_along_axis(lambda m: np.convolve(m, ker, "valid"), 0, ap)
    thr = smooth(thr); clim = smooth(clim)
    thr_map[:, ls, :] = thr

    # --- detect MHW per year over the 91-day block ---
    for yi, y in enumerate(YEARS):
        T = cube[yi]                                            # (NDAY, nlat, NX)
        exceed = (T > thr)
        flat = exceed.reshape(NDAY, -1) & np.isfinite(T).reshape(NDAY, -1)
        valid = detect_mhw(flat).reshape(NDAY, nlat, NX)
        inten = np.where(valid, T - thr, 0).astype(np.float32)
        # October days -> rows [yi*31 : yi*31+31]
        mhw_oct[yi*31:(yi+1)*31, ls, :] = valid[OCT0:OCT1].astype(np.int8)
        int_oct[yi*31:(yi+1)*31, ls, :] = inten[OCT0:OCT1]
        if y == 2018:
            mhw_sep18[:, ls, :] = valid[0:30].astype(np.int8)   # Sep ordinals 0..29
            int_sep18[:, ls, :] = inten[0:30]
    log(f"chunk {ci+1}/{NCHUNK} MHW detected")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(OUT, mhw_oct=mhw_oct, int_oct=int_oct,
                    mhw_sep18=mhw_sep18, int_sep18=int_sep18, thr_map=thr_map,
                    years=np.array(YEARS))
frac = mhw_oct.mean()
log(f"saved -> {OUT}")
log(f"October MHW-day fraction (all cells): {frac:.2%}")
n = mhw_oct.reshape(len(YEARS), 31, NY, NX).mean((1,2,3))
log("per-year Oct MHW fraction: " + ", ".join(f"{y}:{f:.0%}" for y,f in zip(YEARS,n)))
