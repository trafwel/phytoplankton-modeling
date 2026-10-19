#!/usr/bin/env python3
"""
read_ofes_bgc.py — reader for OFES-BGC (Sasai NPZD) raw `.dta` binaries.

Same format your lab notebooks use: big-endian float32 (">f4"), NW-Pacific grid
880(lat) x 1900(lon), 105 depth levels (35 for shortwave), missing = -9.9999998e+33.

Variables: temp, phy (phytoplankton), ntr (nutrient), sw (shortwave), hmxl (2-D MLD),
htflx (2-D), pressur (2-D), and the *clim daily climatologies.

NOTE: works in GRID-INDEX space (like the notebooks). Mapping indices -> lat/lon
degrees needs the OFES grid coordinate file (ask Yasunaka-san / APDRC) — TODO.
"""
from __future__ import annotations
import os
import numpy as np

MISSING = -9.9999998e+33
NY, NX = 880, 1900            # lat, lon
NZ_STD = 105                  # temp / phy / ntr
NZ_SW = 35                    # shortwave

# 28 upper-layer depth centers (m), from the lab notebooks
DEPTHS_28 = np.array([
    2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5,
    52.5, 57.5, 62.5, 67.5, 72.5, 77.5, 82.5, 87.5, 92.5, 97.5,
    102.5, 107.523117, 112.615456, 117.822685, 123.189758,
    128.760666, 134.578110, 140.683319, 147.115692,
])


def load_dta(path, nz=NZ_STD, ny=NY, nx=NX):
    """Read a .dta field. nz=None -> a 2-D field (ny, nx). Returns float32 array
    with missing values as NaN."""
    n = np.fromfile(path, dtype=">f4")
    expected = (ny * nx) if nz is None else (nz * ny * nx)
    if n.size != expected:
        raise ValueError(f"{os.path.basename(path)}: {n.size} floats != expected {expected} "
                         f"(check nz/ny/nx)")
    a = n.reshape((ny, nx) if nz is None else (nz, ny, nx)).astype("float32")
    a[a <= MISSING / 2] = np.nan          # land / missing -> NaN
    return a


def surface(a):
    """Top level of a 3-D field (index 0)."""
    return a[0] if a.ndim == 3 else a


def crop(a, lat_slice, lon_slice):
    """Crop by GRID INDICES (Python slices), e.g. crop(a, slice(335,366), slice(422,453))."""
    return a[..., lat_slice, lon_slice]


if __name__ == "__main__":
    # self-test on the one file that ships with the repo
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "old reference docs/temp.01.01.2000.dta"
    t = load_dta(p, nz=NZ_STD)
    s = surface(t)
    print(f"loaded {p}: shape={t.shape}")
    print(f"surface: valid={np.isfinite(s).mean():.2%}, "
          f"min={np.nanmin(s):.2f}, mean={np.nanmean(s):.2f}, max={np.nanmax(s):.2f}")
