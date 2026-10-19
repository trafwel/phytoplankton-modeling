#!/usr/bin/env python3
"""
jcope_vs_ofes.py — validate JCOPE (data-assimilating reanalysis) against OFES-BGC for the
OVERLAPPING Octobers 2000-2004, on equal footing. Checks whether the observation-constrained
product reproduces the two OFES headlines:
  (1) nutrient dominance   -> corr(phy, nutrients) strong +, corr(phy, temp) weak/negative
  (2) warm-day suppression -> phytoplankton is lower on warm (heatwave-like) days

Both sides: per-(cell, day-of-October) climatology over the 5 years -> anomalies -> region
z-score; then per-cell correlations and a warm-day (temp anomaly > 90th pct) composite,
split north (>=28N) vs south. Grids differ (OFES 0.1 deg, JCOPE 1/12 deg) so we compare
DISTRIBUTIONS/relationships, not cell-by-cell.

Reads: OFES Data/model_table_corrected.parquet ; JCOPE Data/jcope_pilot/{t,phy,din}/*.npy
"""
import glob, numpy as np, pandas as pd, pyarrow.parquet as pq

def zscore(a):
    s = np.nanstd(a); return a / s if s else a

def percell_corr(x, y):
    """corr over axis 0 (time) per cell -> (H,W)."""
    xm = x - np.nanmean(x, 0); ym = y - np.nanmean(y, 0)
    num = np.nansum(xm*ym, 0)
    den = np.sqrt(np.nansum(xm*xm, 0) * np.nansum(ym*ym, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num/den

def warmday_composite(tz, pz, north_mask):
    """mean phy_z on warm days (tz>90th pct) vs rest, north/south. tz,pz flat (N,)."""
    out = {}
    for name, m in [("north", north_mask), ("south", ~north_mask)]:
        t, p = tz[m], pz[m]
        ok = np.isfinite(t) & np.isfinite(p); t, p = t[ok], p[ok]
        thr = np.nanpercentile(t, 90)
        warm = t > thr
        out[name] = (p[warm].mean() - p[~warm].mean(), warm.mean(), len(p))
    return out

# ================= JCOPE side =================
J = "JCOPE Data/jcope_pilot"
def load_oct(var):
    cube = []
    for y in range(2000, 2005):
        fs = sorted(glob.glob(f"{J}/{var}/{var}.{y}10??.npy"))   # October = MM '10'
        cube.append(np.stack([np.load(f).squeeze() for f in fs]))  # (31,H,W)
    return np.stack(cube)                                          # (5,31,H,W)
tJ, pJ, dJ = load_oct("t"), load_oct("phy"), load_oct("din")
H, W = tJ.shape[-2:]
clim = lambda c: c - c.mean(0, keepdims=True)                      # per (day-of-Oct, cell) over 5 yrs
tJa = zscore(clim(tJ).reshape(-1, H, W)); pJa = zscore(clim(pJ).reshape(-1, H, W)); dJa = zscore(clim(dJ).reshape(-1, H, W))
# JCOPE lat idx: lat = 10.4583 + j*0.08333 -> 28N at j~=211
latJ = 10.4583 + np.arange(H)*0.083333
northJ = (latJ >= 28)
rc_pd = percell_corr(pJa, dJa); rc_pt = percell_corr(pJa, tJa)
def band(r, nm): return np.nanmean(r[nm]) if r.ndim==2 else r
nmask2d = np.broadcast_to(northJ[:, None], (H, W))
print("================ JCOPE (obs-constrained) — Oct 2000-2004 ================")
print(f"corr(phy, nutrients): north {np.nanmean(rc_pd[northJ]):+.3f}  south {np.nanmean(rc_pd[~northJ]):+.3f}")
print(f"corr(phy, temp)     : north {np.nanmean(rc_pt[northJ]):+.3f}  south {np.nanmean(rc_pt[~northJ]):+.3f}")
# north mask for the flattened (T,H,W) arrays: tile the 2-D mask across time
northJ_flat = np.tile(np.broadcast_to(northJ[:, None], (H, W)).reshape(-1), tJa.shape[0])
comp = warmday_composite(tJa.reshape(-1), pJa.reshape(-1), northJ_flat)
for b in ("north", "south"):
    d, f, n = comp[b]; print(f"warm-day phy_z drop ({b}): {d:+.3f}  (warm frac {f:.0%}, n={n:,})")

# ================= OFES side (same Octobers) =================
import gc
del tJ, pJ, dJ, tJa, pJa, dJa, rc_pd, rc_pt, northJ_flat; gc.collect()
df = pq.read_table("OFES Data/model_table_corrected.parquet",
                   columns=["t","lat","lon","temp_z","ntr_z","phy_z","latN"],
                   filters=[("t", "<", 155)]).to_pandas()          # 2000-2004 Octobers only
north = df.latN.values >= 28
# per-cell corr via groupby cell over the (<=155) days
df["cell"] = df.lat.values*10000 + df.lon.values
def ofes_corr(a, b):
    """vectorized per-cell corr via grouped sums (fast)."""
    d = df[["cell", a, b, "latN"]].copy()
    d["_xy"] = d[a]*d[b]; d["_xx"] = d[a]*d[a]; d["_yy"] = d[b]*d[b]
    g = d.groupby("cell")
    n = g.size().values.astype(float)
    sx = g[a].sum().values; sy = g[b].sum().values
    sxy = g["_xy"].sum().values; sxx = g["_xx"].sum().values; syy = g["_yy"].sum().values
    cov = sxy - sx*sy/n; vx = sxx - sx*sx/n; vy = syy - sy*sy/n
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov/np.sqrt(vx*vy)
    r[n <= 5] = np.nan
    return r, g["latN"].first().values
print("\n================ OFES-BGC (free-running model) — Oct 2000-2004 ================")
for label, (a, b) in [("corr(phy, nutrients)", ("phy_z","ntr_z")), ("corr(phy, temp)     ", ("phy_z","temp_z"))]:
    r, latn = ofes_corr(a, b); nn = latn >= 28
    print(f"{label}: north {np.nanmean(r[nn]):+.3f}  south {np.nanmean(r[~nn]):+.3f}")
compO = warmday_composite(df.temp_z.values, df.phy_z.values, north)
for b in ("north", "south"):
    d, f, n = compO[b]; print(f"warm-day phy_z drop ({b}): {d:+.3f}  (warm frac {f:.0%}, n={n:,})")
