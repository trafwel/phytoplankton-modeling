#!/usr/bin/env python3
"""fig_oyashio.py — zoom the coupling heatmaps into the NE corner of the crop: the
Kuroshio-Oyashio transition zone (~33-40N, 140-150E). The Oyashio proper lies north of
the crop's 40N edge; this shows its southern reach / the subarctic front we DO capture.

Uses cached per-cell correlation grids from fig_heatmap.py (figures/_corr_{temp,ntr}.npy).
Saves: figures/oyashio_zoom.png
"""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
try: import cmocean; CM = cmocean.cm.balance
except Exception: CM = plt.get_cmap("RdBu_r")

# grid: lat_idx 0..299 -> 10.05..39.95N ; lon_idx 0..299 -> 120.05..149.95E
def latidx(d): return int(round((d - 10.05) / 0.1))
def lonidx(d): return int(round((d - 120.05) / 0.1))
LA0, LA1 = latidx(33), 300          # 33-40N
LO0, LO1 = lonidx(140), 300         # 140-150E
EXT = [120.05 + 0.1*LO0, 149.95, 10.05 + 0.1*LA0, 39.95]

Rt = np.load("figures/_corr_temp.npy")[LA0:LA1, LO0:LO1]
Rn = np.load("figures/_corr_ntr.npy")[LA0:LA1, LO0:LO1]

fig, axes = plt.subplots(1, 2, figsize=(13, 6.6))
for ax, R, ttl in [(axes[0], Rt, "A.  Temperature ↔ phytoplankton"),
                   (axes[1], Rn, "B.  Nutrients ↔ phytoplankton")]:
    ax.set_facecolor("0.82")
    im = ax.imshow(R, origin="lower", extent=EXT, cmap=CM, vmin=-1, vmax=1, aspect="auto")
    ax.axhline(40, color="k", lw=1)
    ax.text(140.4, 39.6, "40°N — crop edge (Oyashio core lies north of here)",
            fontsize=8, va="top")
    ax.plot([140.5, 149.5], [35, 37.5], color="0.15", lw=1.2, ls=":")
    ax.text(145.5, 36.6, "Kuroshio–Oyashio\ntransition", fontsize=8.5, style="italic",
            rotation=8, ha="center")
    ax.set_title(ttl, loc="left", weight="bold")
    ax.set_xlabel("longitude (°E)"); ax.set_ylabel("latitude (°N)")
    plt.colorbar(im, ax=ax, shrink=.85, label="per-cell correlation")

fig.suptitle("Kuroshio–Oyashio transition (NE corner of the crop, 33–40°N, 140–150°E)  ·  "
             "October 2000–2019\nTemperature↔phytoplankton is strongly NEGATIVE here — deepest "
             "at the 40°N Oyashio edge (warm anomalies suppress blooms); nutrients stay strongly "
             "positive", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("figures/oyashio_zoom.png", dpi=140)
print("saved -> figures/oyashio_zoom.png")
print(f"NE-corner mean corr: temp={np.nanmean(Rt):.3f}  ntr={np.nanmean(Rn):.3f}  "
      f"| temp positive-fraction={np.nanmean(Rt>0):.0%}")
