#!/usr/bin/env python3
"""
step1_bgc_series.py — build daily temp + phy time series for a region, from the
OFES-BGC `.dta` binaries. Replaces the OPeNDAP/CMEMS acquisition of the old plan.

Loops daily dates, reads each variable's surface field, crops to a region INDEX box,
and stacks into (time, y, x) arrays saved as one NetCDF.

Output: data/interim/bgc_series.nc  (vars: temp, phy ; coords time, y, x)

Set the CONFIG paths/box/span to your lab data. Works in grid-index space (mapping
y/x indices -> lat/lon degrees needs the OFES grid file — see read_ofes_bgc.py).

Requires: numpy, pandas, xarray, and read_ofes_bgc.py in the same folder.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import xarray as xr
from read_ofes_bgc import load_dta, NZ_STD

# ---------------- CONFIG (set to your lab paths / region / span) ----------------
VAR_DIRS = {                       # one directory per variable
    "temp": "E:/OFES_daily/temp/",
    "phy":  "D:/OFES_daily/phy/",
}
FNAME  = "{var}.{m:02d}.{d:02d}.{y}.dta"   # e.g. temp.01.01.2000.dta
START, END = "2000-01-01", "2009-12-31"    # the span you can pull (OFES has plenty)
# Subarctic NW Pacific: Kamchatka / Oyashio -> mid-North-Pacific, ~40-60N, 150-180E.
# (Cleaner seasonal-bloom signal than the Kuroshio.) For a first plumbing pass, use a
# smaller sub-box, e.g. LAT_IDX=(650,700), LON_IDX=(550,600).
LAT_IDX = (600, 800)               # region as GRID INDICES (lat)
LON_IDX = (500, 800)               # region as GRID INDICES (lon)
LEVEL   = 0                        # 0 = surface (3-D fields; use an upper-layer mean later if wanted)
OUT     = "data/interim/bgc_series.nc"


def main() -> int:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    dates = pd.date_range(START, END, freq="D")
    ys, xs = slice(*LAT_IDX), slice(*LON_IDX)
    ny, nx = LAT_IDX[1] - LAT_IDX[0], LON_IDX[1] - LON_IDX[0]

    arrs = {v: np.full((len(dates), ny, nx), np.nan, np.float32) for v in VAR_DIRS}
    missing_days = {v: 0 for v in VAR_DIRS}

    for i, d in enumerate(dates):
        for v, dirp in VAR_DIRS.items():
            path = os.path.join(dirp, FNAME.format(var=v, m=d.month, d=d.day, y=d.year))
            if not os.path.exists(path):
                missing_days[v] += 1
                continue
            a = load_dta(path, nz=NZ_STD)          # (105, 880, 1900), land->NaN
            arrs[v][i] = a[LEVEL, ys, xs]
        if i % 200 == 0:
            print(f"  {d.date()}  ({100*i/len(dates):4.1f}%)")

    ds = xr.Dataset(
        {v: (["time", "y", "x"], arrs[v]) for v in arrs},
        coords={"time": dates,
                "y": np.arange(*LAT_IDX),
                "x": np.arange(*LON_IDX)},
    )
    ds.attrs["note"] = "OFES-BGC surface daily series (grid-index coords)"
    ds.to_netcdf(OUT)
    print(f"saved -> {OUT}  dims={dict(ds.sizes)}  missing_days={missing_days}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
