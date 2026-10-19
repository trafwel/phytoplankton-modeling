#!/usr/bin/env python3
"""
visualize_test.py — overview figures for the OFES-BGC test data + result.

Panels:
  1-3  mean surface temp / phy / nutrient maps of the crop
  4    one ocean cell's temp & phy through October
  5-6  phy_z vs ntr_z and phy_z vs temp_z density (why nutrients dominate)

Input : data/interim/bgc_series.nc , data/processed/model_table.parquet
Output: figures/test_data_overview.png
Requires: xarray, pandas, numpy, matplotlib
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cmocean  # oceanography colormaps

SERIES = "data/interim/bgc_series.nc"
TABLE  = "data/processed/model_table.parquet"
OUT    = "figures/test_data_overview.png"


def main() -> int:
    os.makedirs("figures", exist_ok=True)
    ds = xr.open_dataset(SERIES)
    df = pd.read_parquet(TABLE)

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # --- maps (time-mean surface fields) ---
    for a, v, cmap, title in [
        (ax[0, 0], "temp", cmocean.cm.thermal, "Mean surface temperature (°C)"),
        (ax[0, 1], "phy",  cmocean.cm.algae,   "Mean surface phytoplankton"),
        (ax[0, 2], "ntr",  cmocean.cm.matter,  "Mean surface nutrient"),
    ]:
        if v in ds:
            im = a.imshow(ds[v].mean("time"), origin="lower", cmap=cmap, aspect="auto")
            plt.colorbar(im, ax=a, shrink=0.8)
        a.set_title(title); a.set_xlabel("x (lon idx)"); a.set_ylabel("y (lat idx)")

    # --- a single ocean cell's time series ---
    surf = ds["temp"].isel(time=0).values
    fin = np.argwhere(np.isfinite(surf))
    iy, ix = fin[len(fin) // 2]
    a = ax[1, 0]
    a.plot(ds["time"], ds["temp"].isel(y=iy, x=ix), color="firebrick", label="temp")
    a.set_ylabel("temp (°C)", color="firebrick"); a.tick_params(axis="x", rotation=45)
    a2 = a.twinx()
    a2.plot(ds["time"], ds["phy"].isel(y=iy, x=ix), color="seagreen", label="phy")
    a2.set_ylabel("phy", color="seagreen")
    a.set_title(f"Cell (y={int(ds.y[iy])}, x={int(ds.x[ix])}) through October")

    # --- density: phy_z vs ntr_z and vs temp_z ---
    for a, xcol, color in [(ax[1, 1], "ntr_z", cmocean.cm.dense),
                           (ax[1, 2], "temp_z", cmocean.cm.amp)]:
        if xcol in df:
            a.hexbin(df[xcol], df["phy_z"], gridsize=50, cmap=color, mincnt=1)
            r = np.corrcoef(df[xcol], df["phy_z"])[0, 1]
            a.set_title(f"phy_z vs {xcol}   (r = {r:.2f})")
            a.set_xlabel(xcol); a.set_ylabel("phy_z")
            a.axhline(0, color="k", lw=.5); a.axvline(0, color="k", lw=.5)

    fig.suptitle("OFES-BGC test data (October 2000 crop) & phy drivers", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"saved -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
