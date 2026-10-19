#!/usr/bin/env python3
"""
step2b_daily_table.py — cube -> tidy ML table for the single-season test.

Per-cell z-score anomalies (the only "anomaly" definable with one season):
  temp_z = z(temp)            feature
  ntr_z  = z(ntr)             feature (if present)
  phy_z  = z(log10(phy))      TARGET  (step5 bins it into collapse/normal/bloom)

Input : data/interim/bgc_series.nc   (from step1b)
Output: data/processed/model_table.parquet
Requires: xarray, numpy, pandas, pyarrow
"""
from __future__ import annotations
import os
import numpy as np
import xarray as xr

SERIES = "data/interim/bgc_series.nc"
OUT    = "data/processed/model_table.parquet"


def zscore_time(da: xr.DataArray) -> xr.DataArray:
    """Per-cell standardization over the time axis."""
    return (da - da.mean("time")) / da.std("time")


def main() -> int:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ds = xr.open_dataset(SERIES)

    data = {"temp_z": zscore_time(ds["temp"])}
    if "ntr" in ds:
        data["ntr_z"] = zscore_time(ds["ntr"])
    logphy = np.log10(ds["phy"].where(ds["phy"] > 0))
    data["phy_z"] = zscore_time(logphy)

    out = xr.Dataset(data)
    df = out.to_dataframe().reset_index().rename(columns={"y": "lat", "x": "lon"})
    df = df.dropna(subset=["phy_z"])
    df.to_parquet(OUT, index=False)
    print(f"saved -> {OUT}: {df.shape[0]} rows, cols={list(df.columns)}")
    for c in ["temp_z", "ntr_z", "phy_z"]:
        if c in df:
            print(f"  {c}: mean={df[c].mean():.3f} std={df[c].std():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
