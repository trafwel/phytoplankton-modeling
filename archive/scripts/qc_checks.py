#!/usr/bin/env python3
"""
qc_checks.py — verification/sanity checks for the OFES-BGC test run.

This is the step that turns "it ran" into "it ran correctly." Run it after step1
and again after step2; it prints stats and saves check figures to figures/.

Usage:
  python qc_checks.py series   # after step1_bgc_series.py  -> checks bgc_series.nc
  python qc_checks.py table    # after step2_bgc_table.py   -> checks model_table.parquet
  python qc_checks.py all      # both (default)

Requires: xarray, pandas, numpy, matplotlib
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = "data/interim/bgc_series.nc"
TABLE  = "data/processed/model_table.parquet"
FIGDIR = "figures"


def _first_ocean_cell(da2d):
    """Indices (iy, ix) of a finite (ocean) cell near the box center."""
    finite = np.argwhere(np.isfinite(da2d.values))
    if len(finite) == 0:
        return None
    return tuple(finite[len(finite) // 2])   # a middle-ish ocean cell


# ---------------- Stage 1: the daily series ----------------
def check_series():
    if not os.path.exists(SERIES):
        print(f"[series] {SERIES} not found — run step1 first.")
        return
    os.makedirs(FIGDIR, exist_ok=True)
    ds = xr.open_dataset(SERIES)
    print(f"[series] dims: {dict(ds.sizes)}")
    for v in ds.data_vars:
        a = ds[v]
        oc = float(np.isfinite(a.isel(time=0)).mean())
        print(f"  {v}: ocean-fraction(t0)={oc:.0%}  "
              f"min={float(a.min()):.3g} mean={float(a.mean()):.3g} max={float(a.max()):.3g}")

    # mean maps
    fig, axes = plt.subplots(1, len(ds.data_vars), figsize=(6*len(ds.data_vars), 5))
    axes = np.atleast_1d(axes)
    for ax, v in zip(axes, ds.data_vars):
        ds[v].mean("time").plot(ax=ax, robust=True)
        ax.set_title(f"mean {v}")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/qc_series_maps.png", dpi=120)
    print(f"  saved {FIGDIR}/qc_series_maps.png")

    # one cell's time series
    cell = _first_ocean_cell(ds["temp"].isel(time=0))
    if cell:
        iy, ix = cell
        fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        ds["temp"].isel(y=iy, x=ix).plot(ax=ax[0]); ax[0].set_title(f"temp @ cell (y={ds.y.values[iy]}, x={ds.x.values[ix]})")
        ds["phy"].isel(y=iy, x=ix).plot(ax=ax[1]); ax[1].set_title("phy (same cell)")
        fig.tight_layout(); fig.savefig(f"{FIGDIR}/qc_series_cell.png", dpi=120)
        print(f"  saved {FIGDIR}/qc_series_cell.png  (expect seasonal temp + phy spring peak)")


# ---------------- Stage 2: the model table ----------------
def check_table():
    if not os.path.exists(TABLE):
        print(f"[table] {TABLE} not found — run step2 first.")
        return
    os.makedirs(FIGDIR, exist_ok=True)
    df = pd.read_parquet(TABLE)
    print(f"[table] shape: {df.shape}")
    print(f"  columns: {list(df.columns)}")

    # all-NaN columns?
    allnan = [c for c in df.columns if df[c].isna().all()]
    print(f"  all-NaN columns: {allnan or 'none'}")

    # feature/target stats (should center near 0 for anomalies)
    for c in ["sst_anom", "phy_anom", "mhw_days", "mhw_intensity_max", "mhw_intensity_cum"]:
        if c in df:
            s = df[c]
            print(f"  {c}: mean={s.mean():.3f} std={s.std():.3f} "
                  f"nan={s.isna().mean():.0%}")

    # class balance of the target (3-class edges [-0.75, 0.75])
    if "phy_anom" in df:
        cls = np.digitize(df["phy_anom"].dropna().values, bins=[-0.75, 0.75])
        names = {0: "collapse", 1: "normal", 2: "bloom"}
        vc = pd.Series(cls).value_counts().sort_index()
        print("  class balance (phy_anom):")
        for k in [0, 1, 2]:
            n = int(vc.get(k, 0))
            print(f"    {names[k]:9s} {n:6d} ({100*n/len(cls):4.1f}%)")

        fig, ax = plt.subplots(figsize=(7, 4))
        df["phy_anom"].plot.hist(bins=60, ax=ax)
        for e in (-0.75, 0.75):
            ax.axvline(e, color="r", ls="--")
        ax.set_title("phy_anom distribution (red = collapse/bloom cutoffs)")
        fig.tight_layout(); fig.savefig(f"{FIGDIR}/qc_table_phyanom.png", dpi=120)
        print(f"  saved {FIGDIR}/qc_table_phyanom.png")


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("series", "all"):
        check_series()
    if what in ("table", "all"):
        check_table()


if __name__ == "__main__":
    main()
