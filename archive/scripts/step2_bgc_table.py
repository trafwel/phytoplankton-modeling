#!/usr/bin/env python3
"""
step2_bgc_table.py — OFES-BGC preprocessing -> ML table.

From the daily temp+phy series (step1), builds:
  FEATURES : sst_anom (monthly temp anomaly) + MHW features (from Hobday on daily temp)
  TARGET   : phy_anom (monthly log-phytoplankton anomaly, z-scored)
and writes one tidy table: one row per (cell, month).

Framing (locked): predict phytoplankton from TEMPERATURE / MHW only -> least circular,
a genuine "do heatwaves forecast phytoplankton collapse?" question. (ntr/sw/hmxl can
be added later for a mechanism-attribution pass.)

Input : data/interim/bgc_series.nc      (from step1)
Output: data/processed/model_table.parquet

Requires: numpy, pandas, xarray, marineHeatWaves, pyarrow
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import xarray as xr
import marineHeatWaves as mhw

SERIES   = "data/interim/bgc_series.nc"
OUT      = "data/processed/model_table.parquet"
CLIM_PERIOD = [2000, 2009]          # baseline for the Hobday threshold (fit your span)
STANDARDIZE = True


def month_start(times):
    return pd.DatetimeIndex(times).to_period("M").to_timestamp()


def monthly_anomaly(da: xr.DataArray, standardize=True):
    clim = da.groupby("time.month").mean("time")
    anom = da.groupby("time.month") - clim
    if standardize:
        std = da.groupby("time.month").std("time")
        anom = anom.groupby("time.month") / std
    return anom


def cell_mhw_monthly(daily_dates, temp_daily, month_index):
    """Hobday MHW on one cell's DAILY temp -> monthly (days, imax, icum)."""
    n = len(month_index)
    nan = np.full(n, np.nan)
    if np.isfinite(temp_daily).sum() < 365:
        return nan, nan, nan
    t = np.array([d.toordinal() for d in pd.DatetimeIndex(daily_dates).date])
    try:
        mhws, clim = mhw.detect(t, temp_daily, climatologyPeriod=CLIM_PERIOD)
    except Exception:
        return nan, nan, nan
    in_mhw = np.zeros(len(temp_daily), bool)
    inten = np.zeros(len(temp_daily))
    for i in range(mhws["n_events"]):
        a, b = mhws["index_start"][i], mhws["index_end"][i]
        in_mhw[a:b + 1] = True
        inten[a:b + 1] = temp_daily[a:b + 1] - clim["seas"][a:b + 1]
    daily = pd.DataFrame({"d": in_mhw.astype(int),
                          "i": np.where(in_mhw, inten, 0.0)},
                         index=pd.DatetimeIndex(daily_dates))
    g = daily.resample("MS")
    return (g["d"].sum().reindex(month_index).values,
            g["i"].max().reindex(month_index).values,
            g["i"].sum().reindex(month_index).values)


def main() -> int:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ds = xr.open_dataset(SERIES)
    temp, phy = ds["temp"], ds["phy"]
    daily_dates = pd.DatetimeIndex(temp["time"].values)
    month_index = pd.date_range(month_start(daily_dates)[0],
                                month_start(daily_dates)[-1], freq="MS")

    ys, xs = ds["y"].values, ds["x"].values
    shape = (len(month_index), len(ys), len(xs))
    days = np.full(shape, np.nan); imax = np.full(shape, np.nan); icum = np.full(shape, np.nan)

    # ---- MHW features per cell (the heavy loop) ----
    for iy in range(len(ys)):
        for ix in range(len(xs)):
            series = temp.isel(y=iy, x=ix).values
            d, mx, cm = cell_mhw_monthly(daily_dates, series, month_index)
            days[:, iy, ix] = d; imax[:, iy, ix] = mx; icum[:, iy, ix] = cm
        print(f"  MHW row {iy+1}/{len(ys)}")

    mhw_ds = xr.Dataset(
        {"mhw_days": (["time", "y", "x"], days),
         "mhw_intensity_max": (["time", "y", "x"], imax),
         "mhw_intensity_cum": (["time", "y", "x"], icum)},
        coords={"time": month_index, "y": ys, "x": xs})

    # ---- monthly anomalies: temp feature + phy target ----
    temp_m = temp.resample(time="MS").mean()
    temp_m = temp_m.assign_coords(time=month_start(temp_m["time"].values))
    sst_anom = monthly_anomaly(temp_m, STANDARDIZE)

    logphy = np.log10(phy.where(phy > 0))
    logphy_m = logphy.resample(time="MS").mean()
    logphy_m = logphy_m.assign_coords(time=month_start(logphy_m["time"].values))
    phy_anom = monthly_anomaly(logphy_m, STANDARDIZE)

    merged = xr.Dataset({
        "sst_anom": sst_anom, "phy_anom": phy_anom,
    })
    merged = xr.merge([merged, mhw_ds], join="inner")

    # ---- cube -> tidy table; name index coords lat/lon so step5 works unchanged ----
    df = merged.to_dataframe().reset_index()
    df = df.rename(columns={"y": "lat", "x": "lon"})
    df = df.dropna(subset=["phy_anom"])
    df.to_parquet(OUT, index=False)
    print(f"saved -> {OUT}: {df.shape[0]} rows, cols={list(df.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
