#!/usr/bin/env python3
"""
step1b_multiyear_physfeat.py — Stream A table: physics (temp, w) -> phy, ntr.

Extends the multi-year build with vertical velocity `w` (upwelling), so we can test
whether physical mixing/upwelling — not just SST — drives nutrients & phytoplankton.

Features : temp_z (surface temp anomaly), w_z (subsurface vertical-velocity anomaly)
Targets  : phy_z, ntr_z   (run step5 twice, once per target)

- temp/phy/ntr use the lab's precomputed `*_anom` (surface). w has no precomputed
  anomaly, so we build it here (climatology per day-of-October across the 20 years).
- `w` is taken at a SUBSURFACE level (surface w ~= 0; upwelling matters at the
  nutricline ~100 m).

Input : /Volumes/HD-SHU3/OFES_test/Data/{temp,ntr,phy,w}
Output: OFES Data/model_table.parquet   (temp_z, w_z, phy_z, ntr_z, year, lat, lon)
Requires: numpy, pandas, pyarrow
"""
from __future__ import annotations
import os
import re
import glob
import numpy as np
import pandas as pd

DATA_DIR = "/Volumes/HD-SHU3/OFES_test/Data"
OUT      = "OFES Data/model_table.parquet"
# CORRECTED crop shape (from HD-SHU3/script_test/data.ipynb: data_in[0:28,300:600,200:500]).
# The crop is 28 upper depth levels x 300 lat x 300 lon = 10-40N, 120-150E.
# (Was wrongly 105,150,160 before — same byte count, wrong arrangement.)
NZ, NY, NX = 28, 300, 300
FILL_BELOW = -5e33
W_LEVEL = 20            # index 20 ~ 100 m — subsurface upwelling (surface w ~= 0). Valid range 0-27.
_DATE = re.compile(r"\.(\d{2})\.(\d{2})\.(\d{4})\.dta$")

# precomputed-anomaly variables -> output column
ANOM = {"temp": "temp_z", "ntr": "ntr_z", "phy": "phy_z"}


def read_level(path: str, level: int) -> np.ndarray:
    a = np.fromfile(path, dtype=">f4").reshape((NZ, NY, NX))[level]
    return np.where(a <= FILL_BELOW, np.nan, a).astype("float32")


def dated_files(folder: str, prefix: str) -> dict:
    out = {}
    for f in glob.glob(f"{DATA_DIR}/{folder}/{prefix}.*.dta"):
        m = _DATE.search(f)
        if m:
            mo, da, yr = map(int, m.groups())
            out[pd.Timestamp(yr, mo, da)] = f
    return dict(sorted(out.items()))


def pooled_z(cube: np.ndarray) -> np.ndarray:
    s = np.nanstd(cube)
    return cube / s if s else cube


def main() -> int:
    # reference dates = where phy anomalies exist
    dates = list(dated_files("phy", "phy_anom"))
    print(f"dates: {len(dates)} ({dates[0].date()}..{dates[-1].date()})")

    cols = {}

    # --- precomputed anomalies: temp, ntr, phy (surface) ---
    for folder, col in ANOM.items():
        files = dated_files(folder, f"{folder}_anom")
        cube = np.full((len(dates), NY, NX), np.nan, np.float32)
        for i, d in enumerate(dates):
            if d in files:
                cube[i] = read_level(files[d], 0)
        cols[col] = pooled_z(cube)
        print(f"  {folder}_anom -> {col}")

    # --- w: build anomaly ourselves (subsurface level, clim per day-of-October) ---
    wfiles = dated_files("w", "w")
    wcube = np.full((len(dates), NY, NX), np.nan, np.float32)
    for i, d in enumerate(dates):
        if d in wfiles:
            wcube[i] = read_level(wfiles[d], W_LEVEL)
    doy = np.array([d.strftime("%m-%d") for d in dates])          # e.g. '10-01'
    wanom = np.empty_like(wcube)
    for dd in np.unique(doy):                                     # climatology per calendar day
        m = doy == dd
        wanom[m] = wcube[m] - np.nanmean(wcube[m], axis=0)
    cols["w_z"] = pooled_z(wanom)
    print(f"  w (level {W_LEVEL}) -> w_z  (filled {np.isfinite(wcube).any(axis=(1,2)).sum()} days)")

    # --- tidy table ---
    T = len(dates)
    tt = np.repeat([d.value for d in dates], NY * NX)
    df = pd.DataFrame({
        "time": pd.to_datetime(tt),
        "year": pd.to_datetime(tt).year,
        "lat": np.tile(np.repeat(np.arange(NY), NX), T),
        "lon": np.tile(np.arange(NX), T * NY),
        **{c: cols[c].reshape(-1) for c in cols},
    }).dropna(subset=["phy_z"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"saved -> {OUT}: {df.shape[0]} rows, cols={list(df.columns)}")
    for c in ["temp_z", "w_z", "ntr_z", "phy_z"]:
        print(f"  {c}: mean={df[c].mean():.3f} std={df[c].std():.3f} nan={df[c].isna().mean():.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
