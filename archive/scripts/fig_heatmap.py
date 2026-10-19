#!/usr/bin/env python3
"""fig_heatmap.py — geographic heatmaps of phytoplankton coupling over the corrected
crop (28,300,300) = 10-40N, 120-150E. For every grid cell we correlate its October
time series (2000-2019) of phytoplankton anomaly against temperature and against
nutrients, giving two 300x300 maps.

Run in stages (each cell_corr pass is heavy; staging keeps memory/time bounded):
    python src/fig_heatmap.py temp     # -> figures/_corr_temp.npy
    python src/fig_heatmap.py ntr      # -> figures/_corr_ntr.npy
    python src/fig_heatmap.py plot     # -> figures/coupling_heatmap.png
Or `python src/fig_heatmap.py all` to do everything in one process (needs a few minutes).
Reads: OFES Data/model_table_corrected.parquet
"""
import sys, os, numpy as np, pyarrow.parquet as pq
NY = NX = 300; P = NY * NX
PARQ = "OFES Data/model_table_corrected.parquet"
os.makedirs("figures", exist_ok=True)

def cell_corr(var):
    """Per-cell Pearson corr of `var` anomaly vs phytoplankton anomaly -> (300,300)."""
    t = pq.read_table(PARQ, columns=["lat", "lon", var, "phy_z"])
    cell = (t["lat"].to_numpy().astype(np.int32) * NX + t["lon"].to_numpy().astype(np.int32))
    a = t[var].to_numpy(); b = t["phy_z"].to_numpy(); del t
    w  = (np.isfinite(a) & np.isfinite(b)).astype(np.float32)
    a0 = np.where(np.isfinite(a), a, np.float32(0)).astype(np.float32)
    b0 = np.where(np.isfinite(b), b, np.float32(0)).astype(np.float32)
    bc = lambda ww: np.bincount(cell, weights=ww, minlength=P)
    n, sx, sy = bc(w), bc(a0*w), bc(b0*w)
    sxy, sx2, sy2 = bc(a0*b0*w), bc(a0*a0*w), bc(b0*b0*w)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (n*sxy - sx*sy) / np.sqrt((n*sx2 - sx*sx) * (n*sy2 - sy*sy))
    r[n < 20] = np.nan
    return r.reshape(NY, NX)

def make(var, tag):
    r = cell_corr(var); np.save(f"figures/_corr_{tag}.npy", r)
    print(f"{tag}: saved figures/_corr_{tag}.npy | north r={np.nanmean(r[150:]):.3f} "
          f"south r={np.nanmean(r[:150]):.3f}", flush=True)

def plot():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    try: import cmocean; CM = cmocean.cm.balance
    except Exception: CM = plt.get_cmap("RdBu_r")
    Rt = np.load("figures/_corr_temp.npy"); Rn = np.load("figures/_corr_ntr.npy")
    EXT = [120.05, 149.95, 10.05, 39.95]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4))
    for ax, R, ttl in [(axes[0], Rt, "A.  Temperature ↔ phytoplankton"),
                       (axes[1], Rn, "B.  Nutrients ↔ phytoplankton")]:
        ax.set_facecolor("0.82")
        im = ax.imshow(R, origin="lower", extent=EXT, cmap=CM, vmin=-1, vmax=1, aspect="auto")
        ax.axhline(28, color="k", ls="--", lw=1)
        ax.text(149, 28.4, "28°N split", ha="right", va="bottom", fontsize=8)
        ax.set_title(ttl, loc="left", weight="bold")
        ax.set_xlabel("longitude (°E)"); ax.set_ylabel("latitude (°N)")
        plt.colorbar(im, ax=ax, shrink=.85, label="per-cell correlation")
    fig.suptitle("Where phytoplankton is coupled to temperature vs. nutrients  ·  OFES-BGC crop "
                 "10–40°N, 120–150°E  ·  October 2000–2019\nNutrients (B) drive phytoplankton "
                 "almost everywhere (red); temperature (A) is weak and mostly negative — dashed "
                 "line = 28°N north/south split", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig("figures/coupling_heatmap.png", dpi=140)
    print("saved -> figures/coupling_heatmap.png", flush=True)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("temp", "all"): make("temp_z", "temp")
    if mode in ("ntr", "all"):  make("ntr_z", "ntr")
    if mode in ("plot", "all"): plot()
