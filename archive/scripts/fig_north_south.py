#!/usr/bin/env python3
"""fig_north_south.py — visualize the temperature-limited (north) vs nutrient-limited
(south) regime split in the corrected (28,300,300) crop, 10-40N, 120-150E.

Panel A: per-1-deg-latitude correlation of phytoplankton anomaly with temperature and
         with nutrients. Nutrients dominate everywhere; the temp<->phy link strengthens
         toward the temperate north.
Panel B: temp-only forecast skill (log-loss, from rerun_corrected.py) by band vs the
         random baseline -- the north beats random, the warm south does not.
Reads: OFES Data/model_table_corrected.parquet   Saves: figures/north_south_regime.png
"""
import numpy as np, pyarrow.parquet as pq
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

t = pq.read_table("OFES Data/model_table_corrected.parquet",
                  columns=["latN", "temp_z", "ntr_z", "phy_z"])
latN = t["latN"].to_numpy(); tz = t["temp_z"].to_numpy()
nz = t["ntr_z"].to_numpy();  pz = t["phy_z"].to_numpy()

def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 1000 else np.nan

lats = np.arange(10, 40) + 0.5
ct = np.array([corr(tz[(latN >= l-.5) & (latN < l+.5)], pz[(latN >= l-.5) & (latN < l+.5)]) for l in lats])
cn = np.array([corr(nz[(latN >= l-.5) & (latN < l+.5)], pz[(latN >= l-.5) & (latN < l+.5)]) for l in lats])

DIV = 28.0
fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.5, 5.4), gridspec_kw={"width_ratios": [1.6, 1]})

# ---- Panel A: correlation vs latitude ----
axA.axvspan(10, DIV, color="#f6c9a0", alpha=.35)
axA.axvspan(DIV, 40, color="#a9c9e8", alpha=.35)
axA.axhline(0, color="0.6", lw=.8)
axA.plot(lats, cn, "-o", ms=3.5, color="#1a7f37", label="nutrients ↔ phytoplankton")
axA.plot(lats, ct, "-o", ms=3.5, color="#c1121f", label="temperature ↔ phytoplankton")
axA.axvline(DIV, color="0.35", ls="--", lw=1)
axA.text(19, 1.02, "SOUTH  (warm, nutrient-limited)", ha="center", color="#8a4b1a", fontsize=10, weight="bold")
axA.text(34, 1.02, "NORTH  (cool, temp-limited)", ha="center", color="#1c4f80", fontsize=10, weight="bold")
axA.set_xlim(10, 40); axA.set_ylim(-.5, 1.05)
axA.set_xlabel("latitude (°N)"); axA.set_ylabel("correlation with phytoplankton anomaly")
axA.set_title("A.  What phytoplankton tracks, by latitude", loc="left", weight="bold")
axA.legend(loc="center left", frameon=False, fontsize=9)

# ---- Panel B: temp-only skill by band (from rerun_corrected.py) ----
labels = ["South\n<28°N", "Full\nregion", "North\n≥28°N"]
ll = [1.045, 1.028, 0.988]
colors = ["#e08a4b", "#7a7a7a", "#3d7ec1"]
bars = axB.bar(labels, ll, color=colors, width=.62)
axB.axhline(1.10, color="0.3", ls="--", lw=1.2)
axB.text(2.42, 1.104, "random\nbaseline", ha="right", va="bottom", fontsize=8.5, color="0.3")
for b, v in zip(bars, ll):
    axB.text(b.get_x()+b.get_width()/2, v-0.012, f"{v:.3f}", ha="center", va="top",
             color="white", fontsize=10, weight="bold")
axB.set_ylim(0.95, 1.12)
axB.set_ylabel("temp-only log-loss  (lower = more skill)")
axB.set_title("B.  Temp-only forecast skill by band", loc="left", weight="bold")
axB.annotate("most skill\nin the north", xy=(2, 0.988), xytext=(1.1, 0.965),
             fontsize=9, color="#1c4f80",
             arrowprops=dict(arrowstyle="->", color="#1c4f80"))

fig.suptitle("Temperature adds weak, latitude-dependent forecast skill — strongest in the temperate "
             "north; nutrients dominate everywhere\nOFES-BGC crop 10–40°N, 120–150°E · "
             "October 2000–2019 · 51M cells · year-blocked CV", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("figures/north_south_regime.png", dpi=140)
print("saved -> figures/north_south_regime.png")
print("north mean corr(temp,phy) = %.3f | south = %.3f" %
      (np.nanmean(ct[lats >= DIV]), np.nanmean(ct[lats < DIV])))
