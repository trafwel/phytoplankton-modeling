#!/usr/bin/env python3
"""
step3_3_temporal_align.py
--------------------------------------------------------------------------
Phase 3, Step 3.1 (mask) + Step 3.3 (temporal alignment).

Takes the raw 3-day OFES surface temperature and produces a MONTHLY-mean SST on
the 1998-2014 window, with its time axis normalized to month-start so it will
join cleanly with the (monthly) PFT later.

The original 3-day OFES file is NOT modified here -- Phase 4 still needs it.

Inputs
  data/interim/ofes_arabiansea_sst_1998_2014.nc   (3-day, surface; from your OFES loader)
Output
  data/interim/sst_monthly_1998_2014.nc

Requires: xarray, pandas, numpy  (the `ocean` env)

NOTE: confirm OFES_VAR / coord names against `print(xr.open_dataset(OFES_PATH))`.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import xarray as xr

# ---- CONFIG (edit if your OFES file differs) ----
OFES_PATH = "data/interim/ofes_arabiansea_sst_1998_2014.nc"
OUT_PATH  = "data/interim/sst_monthly_1998_2014.nc"
OFES_VAR  = "sst"          # temperature variable name
FILL_BELOW = -5e33         # OFES land fill is -1e34; anything below this = land
WIN_START, WIN_END = "1998-01-01", "2014-12-31"


def month_start(times) -> pd.DatetimeIndex:
    """Collapse any monthly timestamps to the first of their month.
    This is the fix for the 'month-anchor' bug: 2004-01-31 and 2004-01-15
    both become 2004-01-01, so joins match by year-month."""
    return pd.DatetimeIndex(times).to_period("M").to_timestamp()


def main() -> int:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # ---- Step 3.1: load + mask land ----
    ds = xr.open_dataset(OFES_PATH)
    sst = ds[OFES_VAR]
    sst = sst.where(sst > FILL_BELOW)          # land/fill -> NaN
    print(f"loaded OFES {OFES_VAR}: {dict(sst.sizes)}")

    # ---- Step 3.3a: 3-day -> monthly mean (NaN-aware) ----
    # resample('MS') groups by calendar month and labels at month-START.
    sst_monthly = sst.resample(time="MS").mean()  # skipna=True by default
    print(f"monthly-mean SST: {dict(sst_monthly.sizes)}")

    # ---- Step 3.3b: normalize time to month-start (belt-and-suspenders) ----
    sst_monthly = sst_monthly.assign_coords(
        time=month_start(sst_monthly["time"].values)
    )

    # ---- Step 3.3c: trim to the 1998-2014 overlap window ----
    sst_monthly = sst_monthly.sel(time=slice(WIN_START, WIN_END))
    n = sst_monthly.sizes["time"]
    print(f"trimmed to {WIN_START[:7]}..{WIN_END[:7]}: {n} months "
          f"(expect 204){' OK' if n == 204 else '  <-- CHECK!'}")

    # ---- save ----
    sst_monthly.name = "sst"
    sst_monthly.attrs.update(long_name="OFES monthly-mean SST", units="degC")
    sst_monthly.to_dataset().to_netcdf(OUT_PATH)
    print(f"saved -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
