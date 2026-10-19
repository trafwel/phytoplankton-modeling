#!/usr/bin/env python3
"""
step3_4_anomalies.py
--------------------------------------------------------------------------
Phase 3, Step 3.4 - climatology & anomalies (the feature/target track).

Turns absolute monthly values into ANOMALIES (departures from the normal for that
calendar month), so "unusual" means unusual-for-the-season, not just summer/winter.

- Chlorophyll (DIATO/NANO) is log10-transformed first (it is log-normal).
- Monthly climatology = mean per calendar month over the record.
- Anomaly = value - that-month's-climatology.  (+ optional z-score standardization)
- DCHL/NCHL ratio = DIATO / NANO.

NOTE: this monthly anomaly climatology is DIFFERENT from Phase 4's daily 90th-pct
MHW threshold. Two different climatologies for two different jobs.

Inputs
  data/interim/sst_monthly_1998_2014.nc     (from step 3.3)
  data/interim/pft_on_ofes_grid.nc          (monthly PFT on OFES grid, from 3.2 regrid)
Output
  data/interim/anomalies_monthly.nc

Requires: xarray, numpy, pandas
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import xarray as xr

SST_PATH = "data/interim/sst_monthly_1998_2014.nc"
PFT_PATH = "data/interim/pft_on_ofes_grid.nc"
OUT_PATH = "data/interim/anomalies_monthly.nc"
WIN_START, WIN_END = "1998-01-01", "2014-12-31"
STANDARDIZE = True   # also emit z-score anomalies (value-clim)/std


def month_start(times) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(times).to_period("M").to_timestamp()


def monthly_anomaly(da: xr.DataArray, standardize: bool = False):
    """Return (anomaly, climatology). Uses xarray's groupby-by-month idiom."""
    clim = da.groupby("time.month").mean("time")          # 12 maps (seasonal cycle)
    anom = da.groupby("time.month") - clim                 # value - month-climatology
    if standardize:
        std = da.groupby("time.month").std("time")
        anom = anom.groupby("time.month") / std            # -> z-score per month
    return anom, clim


def main() -> int:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    sst = xr.open_dataset(SST_PATH)["sst"]
    pft = xr.open_dataset(PFT_PATH)

    # normalize PFT time to month-start so it shares SST's axis, then trim
    pft = pft.assign_coords(time=month_start(pft["time"].values))
    pft = pft.sel(time=slice(WIN_START, WIN_END))
    sst = sst.sel(time=slice(WIN_START, WIN_END))

    # ---- log-transform chlorophyll (guard non-positive -> NaN) ----
    log_diato = np.log10(pft["DIATO"].where(pft["DIATO"] > 0))
    log_nano  = np.log10(pft["NANO"].where(pft["NANO"] > 0))

    # ---- anomalies ----
    sst_anom, _   = monthly_anomaly(sst,       standardize=STANDARDIZE)
    diato_anom, _ = monthly_anomaly(log_diato, standardize=STANDARDIZE)
    nano_anom, _  = monthly_anomaly(log_nano,  standardize=STANDARDIZE)

    data = {
        "sst_anom":        sst_anom,
        "logdiato_anom":   diato_anom,
        "lognano_anom":    nano_anom,
        "dchl_nchl_ratio": pft["DIATO"] / pft["NANO"].where(pft["NANO"] > 0),
    }

    # ---- DINO -> diatom-dominance index (the diatom->dinoflagellate signal) ----
    if "DINO" in pft:
        log_dino = np.log10(pft["DINO"].where(pft["DINO"] > 0))
        dino_anom, _ = monthly_anomaly(log_dino, standardize=STANDARDIZE)
        denom = pft["DIATO"] + pft["DINO"]
        dominance = pft["DIATO"] / denom.where(denom > 0)     # diatom fraction of diatom+dino
        dom_anom, _ = monthly_anomaly(dominance, standardize=STANDARDIZE)
        data["logdino_anom"]          = dino_anom
        data["diatom_dominance"]      = dominance             # drops when dino replaces diatoms
        data["diatom_dominance_anom"] = dom_anom
        print("  DINO present -> added logdino_anom + diatom_dominance(+anom)")
    else:
        print("  DINO not in PFT file -> dominance index skipped")

    out = xr.Dataset(data)
    out.attrs["note"] = ("monthly anomalies vs per-calendar-month climatology; "
                         "chl anomalies on log10; z-scored" if STANDARDIZE else "")
    out.to_netcdf(OUT_PATH)
    print(f"saved -> {OUT_PATH}")

    # quick sanity: anomalies should center near 0
    for name in ["sst_anom", "logdiato_anom", "lognano_anom"]:
        m = float(out[name].mean(skipna=True))
        print(f"  mean {name} = {m:.3f} (expect ~0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
