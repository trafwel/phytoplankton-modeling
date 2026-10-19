#!/usr/bin/env python3
"""
step4_mhw_features.py — Phase 4: marine-heatwave detection -> monthly features.

Runs the Hobday (2016) MHW algorithm on OFES surface temperature, per grid cell,
then aggregates the events into monthly features that join the model table:
    mhw_days            (days in MHW that month)
    mhw_intensity_max   (peak daily anomaly, degC)
    mhw_intensity_cum   (sum of daily anomaly over MHW days, degC*days)

Design notes (see docs/4_marine_heatwave_detection_walkthrough.md):
  * Detection is DETERMINISTIC (no ML). Its outputs are FEATURES for step5_model.
  * OFES is 3-day; Hobday assumes daily -> we interpolate each cell to daily.
  * The threshold needs a long baseline: ideally feed the FULL OFES record
    (1980-2014) so the 1983-2012 climatology is valid, even though we only KEEP
    monthly features for 1998-2014.
  * This is the heavy step (per-cell loop). Test on ONE cell first.

Input : data/interim/ofes_arabiansea_sst_3day_1980_2014.nc   (3-day surface SST)
Output: data/interim/mhw_monthly_features.nc                  (monthly, 1998-2014)

Requires: xarray, numpy, pandas, marineHeatWaves  (the `ocean` env)
"""
from __future__ import annotations
import os
from datetime import date
import numpy as np
import pandas as pd
import xarray as xr
import marineHeatWaves as mhw

# ---------------- CONFIG ----------------
OFES_PATH = "data/interim/ofes_arabiansea_sst_1998_2014.nc"  # 3-day, same file as step3_3
OUT_PATH  = "data/interim/mhw_monthly_features.nc"
OFES_VAR  = "sst"
FILL_BELOW = -5e33
# SHORTENED baseline (decision): use the available 1998-2014 record as the climatology.
# Caveat: Hobday ideally wants >=30 yr; a 17-yr baseline gives a noisier 90th-pct
# threshold. Revisit if a longer OFES record (back to ~1980) is pulled later.
CLIM_PERIOD = [1998, 2014]
KEEP_START, KEEP_END = "1998-01-01", "2014-12-31"   # months we retain for the model
LAT, LON = "lat", "lon"


# ---------------- Step A: one cell -> monthly MHW features ----------------
def cell_monthly_features(three_day_dates: pd.DatetimeIndex,
                          sst_3day: np.ndarray,
                          month_index: pd.DatetimeIndex):
    """Interpolate a single cell's 3-day series to daily, detect MHWs, and return
    monthly (mhw_days, intensity_max, intensity_cum) aligned to `month_index`.
    Returns arrays of NaN for a land/empty cell."""
    n_months = len(month_index)
    nan = np.full(n_months, np.nan)
    if np.isfinite(sst_3day).sum() < 100:       # basically land / too little data
        return nan, nan, nan

    # 3-day -> daily (linear); this is the Hobday-compatibility step
    s = pd.Series(sst_3day, index=three_day_dates).resample("1D").interpolate("linear")
    daily_dates = s.index
    temp = s.values
    t = np.array([d.toordinal() for d in daily_dates.date])

    # detect: returns events dict + climatology (seasonal cycle + 90th-pct threshold)
    try:
        mhws, clim = mhw.detect(t, temp, climatologyPeriod=CLIM_PERIOD)
    except Exception:
        return nan, nan, nan

    # build daily "in MHW" mask + daily intensity (temp above seasonal climatology)
    in_mhw = np.zeros(len(temp), dtype=bool)
    intensity = np.zeros(len(temp))
    for i in range(mhws["n_events"]):
        a, b = mhws["index_start"][i], mhws["index_end"][i]
        in_mhw[a:b + 1] = True
        intensity[a:b + 1] = temp[a:b + 1] - clim["seas"][a:b + 1]

    # daily -> monthly aggregation
    daily = pd.DataFrame(
        {"in_mhw": in_mhw.astype(int),
         "intensity": np.where(in_mhw, intensity, 0.0)},
        index=daily_dates,
    )
    g = daily.resample("MS")
    days = g["in_mhw"].sum()
    imax = g["intensity"].max()
    icum = g["intensity"].sum()

    # align onto the shared month_index (fill months with no data as NaN)
    days = days.reindex(month_index)
    imax = imax.reindex(month_index)
    icum = icum.reindex(month_index)
    return days.values, imax.values, icum.values


# ---------------- Step B: loop the grid ----------------
def main() -> int:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    ds = xr.open_dataset(OFES_PATH)
    sst = ds[OFES_VAR].where(ds[OFES_VAR] > FILL_BELOW)   # mask land
    lats = sst[LAT].values
    lons = sst[LON].values
    three_day_dates = pd.DatetimeIndex(sst["time"].values)
    print(f"OFES loaded: {dict(sst.sizes)}  ({three_day_dates[0].date()}..{three_day_dates[-1].date()})")

    # the monthly axis we will KEEP (1998-2014 = 204 months)
    month_index = pd.date_range(KEEP_START, KEEP_END, freq="MS")
    n_months = len(month_index)

    # output arrays (month, lat, lon)
    shape = (n_months, len(lats), len(lons))
    out_days = np.full(shape, np.nan)
    out_imax = np.full(shape, np.nan)
    out_icum = np.full(shape, np.nan)

    total = len(lats) * len(lons)
    done = 0
    for iy in range(len(lats)):
        for ix in range(len(lons)):
            series = sst.isel({LAT: iy, LON: ix}).values
            d, imax, icum = cell_monthly_features(three_day_dates, series, month_index)
            out_days[:, iy, ix] = d
            out_imax[:, iy, ix] = imax
            out_icum[:, iy, ix] = icum
            done += 1
        print(f"  row {iy+1}/{len(lats)}  ({100*done/total:4.1f}%)")

    # ---------------- Step C: save as a monthly dataset ----------------
    coords = {"time": month_index, LAT: lats, LON: lons}
    out = xr.Dataset(
        {
            "mhw_days":          (["time", LAT, LON], out_days),
            "mhw_intensity_max": (["time", LAT, LON], out_imax),
            "mhw_intensity_cum": (["time", LAT, LON], out_icum),
        },
        coords=coords,
    )
    out.attrs["note"] = ("Hobday MHW monthly features from OFES; "
                         f"climatology {CLIM_PERIOD[0]}-{CLIM_PERIOD[1]}, "
                         "3-day interpolated to daily")
    out.to_netcdf(OUT_PATH)
    print(f"saved -> {OUT_PATH}  ({n_months} months)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
