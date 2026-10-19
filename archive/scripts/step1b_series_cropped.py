#!/usr/bin/env python3
"""
step1b_series_cropped.py — build a temp+phy(+ntr) daily series from the PRE-CROPPED
OFES_test `.dta` files (shape (105, 150, 160)).

Use this for the copied test data in data/raw/ofes_test/. (The full-domain
step1_bgc_series.py is for the 880x1900 fields; these files are already the region,
so there is NO index-box crop here.)

The files are autumn-only and sparse, so we glob whatever exists and parse the date
from each filename rather than assuming a continuous daily range.

Input : data/raw/ofes_test/{temp,phy,ntr}/<var>.MM.DD.YYYY.dta
Output: data/interim/bgc_series.nc   (vars temp, phy[, ntr]; coords time, y, x)

Requires: numpy, pandas, xarray
"""
from __future__ import annotations
import os
import re
import glob
import numpy as np
import pandas as pd
import xarray as xr

DATA = "data/raw/ofes_test"
VARS = ["temp", "phy", "ntr"]        # ntr used only if its folder exists
NZ, NY, NX = 105, 150, 160           # pre-cropped shape (confirmed)
LEVEL = 0                            # surface
FILL_BELOW = -5e33
OUT = "data/interim/bgc_series.nc"

_DATE = re.compile(r"\.(\d{2})\.(\d{2})\.(\d{4})\.dta$")


def date_of(path: str) -> pd.Timestamp:
    mo, da, yr = map(int, _DATE.search(path).groups())
    return pd.Timestamp(yr, mo, da)


def load_surface(path: str) -> np.ndarray:
    a = np.fromfile(path, dtype=">f4").reshape((NZ, NY, NX))[LEVEL]
    return np.where(a <= FILL_BELOW, np.nan, a).astype("float32")


def main() -> int:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    present = [v for v in VARS if os.path.isdir(f"{DATA}/{v}")]

    # reference date list = the dates temp has
    dates = sorted({date_of(f) for f in glob.glob(f"{DATA}/temp/*.dta")})
    print(f"vars: {present}   dates: {len(dates)} "
          f"({dates[0].date()}..{dates[-1].date()})")

    arrs = {v: np.full((len(dates), NY, NX), np.nan, np.float32) for v in present}
    for i, d in enumerate(dates):
        for v in present:
            fn = f"{DATA}/{v}/{v}.{d.month:02d}.{d.day:02d}.{d.year}.dta"
            if os.path.exists(fn):
                arrs[v][i] = load_surface(fn)

    ds = xr.Dataset(
        {v: (["time", "y", "x"], arrs[v]) for v in present},
        coords={"time": dates, "y": np.arange(NY), "x": np.arange(NX)},
    )
    ds.attrs["note"] = ("OFES_test pre-cropped (105,150,160) surface series; "
                        "subtropical/tropical NW Pacific, autumn only")
    ds.to_netcdf(OUT)
    print(f"saved -> {OUT}  dims={dict(ds.sizes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
