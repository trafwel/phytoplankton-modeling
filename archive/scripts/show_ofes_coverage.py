#!/usr/bin/env python3
"""
show_ofes_coverage.py — visualize why the OFES biology is limited to one region.

Physics (temp/u/v/w): FULL domain (105, 880, 1900).
Biology (phy/ntr):    one crop      (105, 150, 160)  ~= 1.4% of the domain.

Draws the full-domain SST with the biology-crop footprint boxed, next to the crop
itself. (Box location is APPROXIMATE — from the single value-match anchor
crop[75,80]==full[340,280]; the exact crop geometry is unresolved.)
"""
from __future__ import annotations
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cmocean

FULL = "old reference docs/temp.01.01.2000.dta"        # (105,880,1900)
CROP = sorted(glob.glob("data/raw/ofes_test/temp/*.dta"))[0]  # (105,150,160)
NYf, NXf = 880, 1900
NYc, NXc = 150, 160
OY, OX = 265, 200        # approx crop origin on full grid (anchor 340-75, 280-80)


def load(path, ny, nx):
    a = np.fromfile(path, dtype=">f4").reshape((105, ny, nx))[0]
    return np.where(a <= -5e33, np.nan, a)


def lat(y): return -19.95 + 0.1 * y      # grid -> degrees
def lon(x): return 100.05 + 0.1 * x


def main():
    full = load(FULL, NYf, NXf)
    crop = load(CROP, NYc, NXc)

    fcells = int(np.isfinite(full).sum())
    ccells = NYc * NXc
    print(f"physics domain (temp): {NYf}x{NXf} = {NYf*NXf} cells ({fcells} ocean)")
    print(f"biology crop (phy/ntr): {NYc}x{NXc} = {ccells} cells")
    print(f"biology covers ~{100*ccells/(NYf*NXf):.1f}% of the domain area")

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    im0 = ax[0].imshow(full, origin="lower", cmap=cmocean.cm.thermal, aspect="auto",
                       extent=[lon(0), lon(NXf), lat(0), lat(NYf)])
    ax[0].add_patch(Rectangle((lon(OX), lat(OY)), NXc * 0.1, NYc * 0.1,
                              fill=False, edgecolor="red", lw=2.5))
    ax[0].text(lon(OX), lat(OY) + NYc * 0.1 + 2, "biology crop (approx)",
               color="red", fontsize=11, weight="bold")
    ax[0].set(title="OFES PHYSICS = full domain (880x1900)\nBiology (phy/ntr) exists ONLY in the red box",
              xlabel="longitude (E)", ylabel="latitude (N)")
    plt.colorbar(im0, ax=ax[0], shrink=0.7, label="SST (deg C)")

    im1 = ax[1].imshow(crop, origin="lower", cmap=cmocean.cm.thermal, aspect="auto")
    ax[1].set(title=f"The ENTIRE biology dataset (150x160 cells)\n"
                    f"= {100*ccells/(NYf*NXf):.1f}% of the domain area",
              xlabel="x index", ylabel="y index")
    plt.colorbar(im1, ax=ax[1], shrink=0.7, label="SST (deg C)")

    fig.suptitle("Why the OFES biology is limited to one region", fontsize=14)
    fig.tight_layout()
    fig.savefig("figures/ofes_coverage.png", dpi=120)
    print("saved -> figures/ofes_coverage.png")


if __name__ == "__main__":
    main()
