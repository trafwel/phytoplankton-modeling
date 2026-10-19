#!/usr/bin/env python3
"""
get_ofes.py — Phase 1.1: download OFES Arabian Sea surface temperature (3-day).

Source (verified live): APDRC OPeNDAP, OFES 0.1deg NCEP hindcast, 3-day, 3D temp.
  http://apdrc.soest.hawaii.edu/dods/public_ofes/OfES/ncep_0.1_global_3day/temp
Grid: temp[time=5600][lev=54][lat=1500][lon=3600] Float32
  lat -74.95..74.95 @0.1 ; lon 0.05..359.95 @0.1 (0-360) ; lev surface=index0 (2.5 m)
  time 3-day 03jan1980..29dec2025 ; _FillValue -1.0E34
  (server units 'days since 1-1-1' is non-standard -> we build the time axis ourselves)

Output (3-day, surface, var 'sst' on lat/lon/time -- matches step3_3 & step4):
  data/interim/ofes_arabiansea_sst_1998_2014.nc
  + per-year files in data/raw/ofes_temp/ (so re-runs resume)

Downloads ONE YEAR PER REQUEST with retry/resume, because APDRC is slow/flaky.

Requires: xarray, netCDF4, pydap, numpy, pandas
Run: python get_ofes.py
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import pandas as pd
import xarray as xr

# ---------------- CONFIG ----------------
URL = ("http://apdrc.soest.hawaii.edu/dods/public_ofes/"
       "OfES/ncep_0.1_global_3day/temp")
LAT_MIN, LAT_MAX = 5.0, 25.0        # Arabian Sea
LON_MIN, LON_MAX = 50.0, 75.0       # 0-360 convention (50-75E is positive)
YEAR_START, YEAR_END = 1998, 2014   # keep <= 2014 (2015+ is IPRC-restricted)
FILL = -1.0e34

RAW_DIR    = "data/raw/ofes_temp"
MERGED_OUT = "data/interim/ofes_arabiansea_sst_1998_2014.nc"

# --- smoke test ---
# TEST_MODE=True downloads only a tiny slice (a few 3-day steps) to confirm the
# endpoint/box/format work, WITHOUT the full multi-year pull. The result is only
# a plumbing check -- it is NOT enough for MHW/anomaly science.
TEST_MODE  = False
TEST_STEPS = 10                                   # ~1 month of 3-day steps
TEST_OUT   = "data/interim/ofes_TEST.nc"

# OFES 3-day time axis, built manually from the server's stated start.
TIME_ORIGIN, TIME_STEP, N_TIME = "1980-01-03", "3D", 5600
MAX_RETRIES, RETRY_WAIT_S = 4, 20


def build_time_index() -> pd.DatetimeIndex:
    return pd.date_range(start=TIME_ORIGIN, periods=N_TIME, freq=TIME_STEP)


def year_bounds(times: pd.DatetimeIndex, year: int):
    idx = np.where(times.year == year)[0]
    return (int(idx[0]), int(idx[-1])) if idx.size else (-1, -1)


def download_year(ds, times, year):
    t0, t1 = year_bounds(times, year)
    if t0 < 0:
        return None
    sub = (ds["temp"]
           .sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
           .isel(lev=0, time=slice(t0, t1 + 1)))
    sub = sub.load()                                  # the slow network pull
    sub = sub.where(sub > FILL / 2.0)                 # land -> NaN
    sub = sub.assign_coords(time=times[t0:t1 + 1])    # real 3-day dates
    sub.name = "sst"
    sub.attrs.update(long_name="OFES surface potential temperature (SST proxy)",
                     units="degC", region="Arabian Sea 5-25N 50-75E")
    return sub.to_dataset()


def main() -> int:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MERGED_OUT), exist_ok=True)
    times = build_time_index()

    print(f"opening OPeNDAP:\n  {URL}")
    ds = xr.open_dataset(URL, decode_times=False)
    print("  connected. temp dims:", dict(ds["temp"].sizes))

    # ---- smoke test: grab a tiny slice and stop ----
    if TEST_MODE:
        t0, _ = year_bounds(times, YEAR_START)
        print(f"TEST_MODE: pulling {TEST_STEPS} steps from {times[t0].date()} ...")
        da = (ds["temp"]
              .sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
              .isel(lev=0, time=slice(t0, t0 + TEST_STEPS)).load())
        da = da.where(da > FILL / 2.0).assign_coords(time=times[t0:t0 + TEST_STEPS])
        da.name = "sst"
        os.makedirs(os.path.dirname(TEST_OUT), exist_ok=True)
        da.to_dataset().to_netcdf(TEST_OUT)
        print(f"TEST file -> {TEST_OUT}  {dict(da.sizes)}  "
              f"(mean {float(da.mean(skipna=True)):.1f} degC)")
        return 0

    files = []
    for year in range(YEAR_START, YEAR_END + 1):
        fpath = os.path.join(RAW_DIR, f"ofes_sst_{year}.nc")
        if os.path.exists(fpath):
            print(f"  [{year}] already have it")
            files.append(fpath)
            continue
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"  [{year}] downloading (attempt {attempt}) ...")
                da = download_year(ds, times, year)
                if da is None:
                    break
                da.to_netcdf(fpath)
                print(f"  [{year}] saved {dict(da['sst'].sizes)}")
                files.append(fpath)
                break
            except Exception as e:  # noqa: BLE001
                print(f"  [{year}] FAILED ({type(e).__name__}: {e})")
                if attempt == MAX_RETRIES:
                    print(f"  [{year}] giving up; re-run to resume.")
                else:
                    time.sleep(RETRY_WAIT_S)
                    try:
                        ds = xr.open_dataset(URL, decode_times=False)  # reopen dropped session
                    except Exception:
                        pass

    if not files:
        print("no data downloaded.")
        return 1

    print("concatenating yearly files ...")
    merged = xr.open_mfdataset(sorted(files), combine="by_coords")
    merged.to_netcdf(MERGED_OUT)
    print(f"DONE -> {MERGED_OUT}")
    print(merged)
    return 0


if __name__ == "__main__":
    sys.exit(main())
