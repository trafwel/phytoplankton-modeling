#!/usr/bin/env python3
"""
step1b_multiyear_anom.py — build the multi-year model table from the PRECOMPUTED
anomaly fields (temp_anom / phy_anom / ntr_anom), October 2000-2019.

Supersedes step1b + step2b for the multi-year run:
  * uses the lab's own daily anomalies (deseasonalized vs the 20-yr Oct climatology),
    which is better than the per-cell z-score hack.
  * pools all 20 Octobers -> real interannual variability.
  * standardizes each field (divide by its pooled std) so step5's sigma-based class
    edges make sense.

Reads from the DRIVE and writes the table to the DRIVE (keeps the big file off the
low-space laptop).

Input : <DATA_DIR>/{temp,phy,ntr}/<var>_anom.10.DD.YYYY.dta
Output: <DATA_DIR>/model_table.parquet   (cols: time, year, lat, lon, temp_z, ntr_z, phy_z)

Requires: numpy, pandas, pyarrow
"""
from __future__ import annotations
import os
import re
import glob
import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
# Read RAW .dta straight from the source drive (read-only is fine; reading never
# modifies it) -- no 40 GB copy needed. Only the small table below gets written.
DATA_DIR = "/Volumes/HD-SHU3/OFES_test/Data"
# Output written to the dedicated folder on the laptop (~200 MB).
OUT      = "OFES Data/model_table.parquet"
NZ, NY, NX = 28, 300, 300                        # CORRECTED crop shape (data.ipynb: [0:28,300:600,200:500])
FILL_BELOW = -5e33
# variable -> (folder, anomaly-file prefix, output column)
VARS = {
    "temp": ("temp", "temp_anom", "temp_z"),
    "ntr":  ("ntr",  "ntr_anom",  "ntr_z"),
    "phy":  ("phy",  "phy_anom",  "phy_z"),   # phy_z is the TARGET
}
_DATE = re.compile(r"\.(\d{2})\.(\d{2})\.(\d{4})\.dta$")


def load_surface(path: str) -> np.ndarray:
    """Read a .dta anomaly field; return its surface (300,300) with fill->NaN.
    Handles either a full (28,300,300) field or a 2-D (300,300) surface file."""
    a = np.fromfile(path, dtype=">f4")
    if a.size == NZ * NY * NX:
        a = a.reshape((NZ, NY, NX))[0]
    elif a.size == NY * NX:
        a = a.reshape((NY, NX))
    else:
        raise ValueError(f"{os.path.basename(path)}: unexpected size {a.size}")
    return np.where(a <= FILL_BELOW, np.nan, a).astype("float32")


def dates_for(folder: str, prefix: str):
    """All (timestamp, path) for the anomaly files of one variable, sorted."""
    out = {}
    for f in glob.glob(f"{DATA_DIR}/{folder}/{prefix}.*.dta"):
        m = _DATE.search(f)
        if not m:
            continue
        mo, da, yr = map(int, m.groups())
        out[pd.Timestamp(yr, mo, da)] = f
    return dict(sorted(out.items()))


def main() -> int:
    # use temp's dates as the reference axis (phy/ntr share them)
    ref = dates_for(*VARS["temp"][:2])
    dates = list(ref)
    print(f"dates: {len(dates)} ({dates[0].date()}..{dates[-1].date()})")

    cubes = {}
    for key, (folder, prefix, col) in VARS.items():
        files = dates_for(folder, prefix)
        cube = np.full((len(dates), NY, NX), np.nan, np.float32)
        for i, d in enumerate(dates):
            if d in files:
                cube[i] = load_surface(files[d])
        # pooled standardization (anomaly mean ~0; divide by pooled std)
        std = np.nanstd(cube)
        cubes[col] = cube / std if std else cube
        print(f"  {key}: filled {np.isfinite(cube).any(axis=(1,2)).sum()} days, "
              f"pooled std={std:.4g} -> {col}")

    # cube -> tidy table
    T, Y, X = len(dates), NY, NX
    tt = np.repeat([d.value for d in dates], Y * X)
    yy = np.tile(np.repeat(np.arange(Y), X), T)
    xx = np.tile(np.arange(X), T * Y)
    df = pd.DataFrame({
        "time": pd.to_datetime(tt),
        "year": pd.to_datetime(tt).year,
        "lat": yy, "lon": xx,
        **{col: cubes[col].reshape(-1) for col in cubes},
    })
    df = df.dropna(subset=["phy_z"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"saved -> {OUT}: {df.shape[0]} rows, cols={list(df.columns)}")
    for c in ("temp_z", "ntr_z", "phy_z"):
        print(f"  {c}: mean={df[c].mean():.3f} std={df[c].std():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
