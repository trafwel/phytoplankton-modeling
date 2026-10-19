#!/usr/bin/env python3
"""
step3_6_build_table.py
--------------------------------------------------------------------------
Phase 3, Step 3.6 - assemble the ML-ready table.

Merges every monthly field onto the common (time, lat, lon) grid and reshapes the
cube into a TIDY TABLE: one row per (lat, lon, month), columns = features + targets.
This is the single file the model (Phase 5/7) reads.

Inputs (whatever exists)
  data/interim/sst_monthly_1998_2014.nc       (monthly SST)
  data/interim/anomalies_monthly.nc           (SST/DIATO/NANO anomalies + ratio)
  data/interim/pft_on_ofes_grid.nc            (absolute DIATO/NANO/CHL)
  data/interim/mhw_monthly_features.nc        (OPTIONAL - from Phase 4, if ready)
Output
  data/processed/model_table.parquet

Requires: xarray, pandas, numpy  (parquet needs pyarrow: `pip install pyarrow`)
"""
from __future__ import annotations
import os
import pandas as pd
import xarray as xr

INTERIM = "data/interim"
OUT_PATH = "data/processed/model_table.parquet"
# columns that must be present for a row to be usable (drop rows all-NaN in these)
REQUIRED = ["logdiato_anom", "lognano_anom"]


def main() -> int:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # ---- gather the datasets that exist ----
    pieces = []
    def add(path):
        if os.path.exists(path):
            pieces.append(xr.open_dataset(path))
            print(f"  + {os.path.basename(path)}")
        else:
            print(f"  - {os.path.basename(path)} (missing, skipped)")

    print("merging:")
    add(f"{INTERIM}/sst_monthly_1998_2014.nc")
    add(f"{INTERIM}/anomalies_monthly.nc")
    add(f"{INTERIM}/pft_on_ofes_grid.nc")
    add(f"{INTERIM}/mhw_monthly_features.nc")   # optional, from Phase 4

    # ---- merge on shared coords (time, lat, lon) ----
    merged = xr.merge(pieces, join="inner")     # inner = keep only shared cells/months
    print(f"merged dims: {dict(merged.sizes)}")

    # ---- cube -> tidy table ----
    # to_dataframe() gives a MultiIndex (time,lat,lon); reset_index makes them columns.
    df = merged.to_dataframe().reset_index()

    # ---- drop land / unusable rows ----
    have = [c for c in REQUIRED if c in df.columns]
    if have:
        df = df.dropna(subset=have, how="all")

    # keep lat/lon/time as COLUMNS (needed for blocked CV in Phase 7) - already are.
    print(f"table shape: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"columns: {list(df.columns)}")

    df.to_parquet(OUT_PATH, index=False)
    print(f"saved -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
