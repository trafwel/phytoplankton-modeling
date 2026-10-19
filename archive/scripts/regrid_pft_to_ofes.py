#!/usr/bin/env python3
"""
regrid_pft_to_ofes.py
--------------------------------------------------------------------------
Phase 3.2 - Regrid the satellite PFT (DIATO/NANO/CHL, ~4 km) onto the coarser
OFES 0.1-degree Arabian Sea grid, using conservative (area-weighted) regridding.

Result: PFT variables on the SAME grid as OFES temperature, so they can be
correlated cell-by-cell. Time is left untouched here (aligned later in 3.3).

Inputs
  OFES : data/interim/ofes_arabiansea_sst_1998_2014.nc   (target grid donor)
  PFT  : data/raw/pft/pft_arabiansea_1998_2014.nc        (source to move)
Output
  data/interim/pft_on_ofes_grid.nc
  data/interim/regrid_weights_pft_to_ofes.nc   (xesmf weights, reused on re-runs)
  figures/regrid_check_DIATO.png                (before/after sanity map)

Requires: xarray, xesmf, numpy, matplotlib  (all in the `ocean` env)

NOTE: adjust the CONFIG names below if your OFES file uses different coordinate
or variable names. Run `print(xr.open_dataset(OFES_PATH))` to check.
"""

from __future__ import annotations
import os
import numpy as np
import xarray as xr
import xesmf as xe
import matplotlib
matplotlib.use("Agg")  # no display needed; we save PNGs
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# CONFIG  -- edit here if names differ
# ----------------------------------------------------------------------
OFES_PATH = "data/interim/ofes_arabiansea_sst_1998_2014.nc"
PFT_PATH  = "data/raw/pft/pft_arabiansea_1998_2014.nc"
OUT_PATH  = "data/interim/pft_on_ofes_grid.nc"
WEIGHTS   = "data/interim/regrid_weights_pft_to_ofes.nc"
FIG_DIR   = "figures"

# coordinate names in each file
OFES_LAT, OFES_LON = "lat", "lon"            # <- change if your OFES file differs
PFT_LAT,  PFT_LON  = "latitude", "longitude"  # confirmed from CMEMS describe

# PFT variables to regrid (add "DIATO_uncertainty","NANO_uncertainty" if downloaded)
PFT_VARS = ["DIATO", "NANO", "DINO", "CHL"]   # DINO -> diatom-dominance index later

METHOD = "conservative"     # area-weighted; correct for fine -> coarse concentration
NA_THRES = 0.5              # a target cell becomes NaN if >50% of its source is NaN


# ----------------------------------------------------------------------
def bounds_from_centers(c: np.ndarray) -> np.ndarray:
    """Cell edges (len N+1) from monotonic cell centers (len N).
    Conservative regridding needs edges, not just centers."""
    c = np.asarray(c, dtype=float)
    mids = (c[:-1] + c[1:]) / 2.0
    first = c[0] - (mids[0] - c[0])
    last = c[-1] + (c[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def make_grid(lat: np.ndarray, lon: np.ndarray) -> xr.Dataset:
    """Build an xesmf-ready rectilinear grid (centers + bounds)."""
    return xr.Dataset(
        {
            "lat":   (["lat"], np.asarray(lat, float)),
            "lon":   (["lon"], np.asarray(lon, float)),
            "lat_b": (["lat_b"], bounds_from_centers(lat)),
            "lon_b": (["lon_b"], bounds_from_centers(lon)),
        }
    )


def main() -> int:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # --- load ---
    ofes = xr.open_dataset(OFES_PATH)
    pft = xr.open_dataset(PFT_PATH)

    # --- standardize PFT coord names to lat/lon (what xesmf expects) ---
    pft = pft.rename({PFT_LAT: "lat", PFT_LON: "lon"})
    ofes_lat = ofes[OFES_LAT].values
    ofes_lon = ofes[OFES_LON].values

    # --- longitude convention harmonize (both are 50..75 positive here; explicit) ---
    # Put OFES lon into [-180,180] to match PFT. For 50-75E this changes nothing,
    # but it makes the convention explicit and safe if the box ever moves.
    ofes_lon = np.where(ofes_lon > 180.0, ofes_lon - 360.0, ofes_lon)
    pft = pft.assign_coords(lon=np.where(pft["lon"].values > 180.0,
                                         pft["lon"].values - 360.0,
                                         pft["lon"].values))

    # --- source and target grids (with bounds) ---
    src_grid = make_grid(pft["lat"].values, pft["lon"].values)
    dst_grid = make_grid(ofes_lat, ofes_lon)

    # --- build (or reuse) the conservative regridder ---
    reuse = os.path.exists(WEIGHTS)
    print(f"Building regridder (method={METHOD}, reuse_weights={reuse}) ...")
    regridder = xe.Regridder(
        src_grid, dst_grid, METHOD,
        filename=WEIGHTS, reuse_weights=reuse,
    )
    print(regridder)

    # --- attach source lat/lon to each var and regrid ---
    out = {}
    for v in PFT_VARS:
        if v not in pft:
            print(f"  ! {v} not in PFT file, skipping")
            continue
        da = pft[v]  # dims (time, lat, lon)
        da_r = regridder(da, skipna=True, na_thres=NA_THRES, keep_attrs=True)
        out[v] = da_r
        print(f"  regridded {v}: {dict(da.sizes)} -> {dict(da_r.sizes)}")

    ds_out = xr.Dataset(out)
    ds_out.attrs["history"] = "PFT regridded onto OFES 0.1deg grid (conservative)"
    ds_out.to_netcdf(OUT_PATH)
    print(f"Saved -> {OUT_PATH}")

    # --- verification 1: domain-mean should be nearly preserved ---
    if "DIATO" in pft and "DIATO" in ds_out:
        before = float(pft["DIATO"].mean(skipna=True))
        after = float(ds_out["DIATO"].mean(skipna=True))
        print(f"DIATO domain-mean  before={before:.4f}  after={after:.4f} "
              f"(should be close)")

    # --- verification 2: before/after map of the first month ---
    if "DIATO" in pft and "DIATO" in ds_out:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        pft["DIATO"].isel(time=0).plot(ax=ax1, robust=True)
        ax1.set_title("DIATO native ~4 km (month 0)")
        ds_out["DIATO"].isel(time=0).plot(ax=ax2, robust=True)
        ax2.set_title("DIATO on OFES 0.1deg (month 0)")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "regrid_check_DIATO.png"), dpi=120)
        print(f"Saved check figure -> {FIG_DIR}/regrid_check_DIATO.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
