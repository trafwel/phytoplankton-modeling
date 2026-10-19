#!/usr/bin/env python3
"""
hobday_test.py — does marine-heatwave STATE (Hobday events) relate to phytoplankton, and
is that relationship stronger in the temperate north (Oyashio/transition) than the warm
south? Uses the MHW cache from hobday_build.py + the phytoplankton table.

Two analyses:
  1. COMPOSITE (no model): mean phy_z on MHW days vs non-MHW days, north (>=28N) vs south.
     The direct test of "do blooms crash during heatwaves in the north?"
  2. PREDICTIVE: RF with MHW features (flag, intensity) -> phy collapse class, north band,
     compared to the raw temp_z-only baseline (north log-loss was 0.988).

Run after hobday_build.py, on the Mac:
    python src/hobday_test.py
"""
import numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, accuracy_score

Z = np.load("OFES Data/hobday_mhw.npz")
mhw_oct, int_oct = Z["mhw_oct"], Z["int_oct"]      # (620,300,300)

# biology table: t (0..619 date order), lat/lon indices, phy_z, latN
df = pq.read_table("OFES Data/model_table_corrected.parquet",
                   columns=["t", "lat", "lon", "phy_z", "latN", "temp_z"]).to_pandas()
t = df["t"].to_numpy(); la = df["lat"].to_numpy(); lo = df["lon"].to_numpy()
df["mhw"] = mhw_oct[t, la, lo].astype(np.int8)         # 1 if that cell/day is in a MHW
df["mhw_int"] = int_oct[t, la, lo].astype(np.float32)  # deg C above threshold
df = df.dropna(subset=["phy_z"])
north = df["latN"] >= 28

print(f"October MHW-day fraction: overall {df.mhw.mean():.2%} | "
      f"north {df.mhw[north].mean():.2%} | south {df.mhw[~north].mean():.2%}")

# ---------- 1. composite ----------
print("\n=== COMPOSITE: mean phy_z (standardized phytoplankton anomaly) ===")
print(f"{'band':6s} {'n_MHW':>10s} {'phy|MHW':>9s} {'phy|calm':>9s} {'difference':>11s}")
for name, sub in [("north", df[north]), ("south", df[~north]), ("all", df)]:
    a = sub.phy_z[sub.mhw == 1]; b = sub.phy_z[sub.mhw == 0]
    diff = a.mean() - b.mean()
    print(f"{name:6s} {len(a):10d} {a.mean():9.3f} {b.mean():9.3f} {diff:+11.3f}")
print("(negative difference = phytoplankton is LOWER during marine heatwaves = suppression)")

# ---------- 2. predictive (north band) ----------
EDGES = [-0.75, 0.75]; NAMES = {0: "collapse", 1: "normal", 2: "bloom"}
def rf():
    return RandomForestClassifier(n_estimators=150, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=-1, random_state=0)
def run(sub, feats, tag, max_rows=1_500_000):
    d = sub.dropna(subset=feats).copy()
    d["y"] = np.digitize(d["phy_z"].values, EDGES)
    if len(d) > max_rows: d = d.sample(max_rows, random_state=0)
    X, y = d[feats].values, d["y"].values
    grp = (df.loc[d.index, "t"].values // 31)          # year block (31 Oct days per year)
    classes = np.array(sorted(np.unique(y)))
    proba = np.zeros((len(y), len(classes))); pred = np.empty(len(y), int)
    for tr, te in GroupKFold(5).split(X, y, grp):
        m = rf().fit(X[tr], y[tr]); p = m.predict_proba(X[te])
        for j, c in enumerate(m.classes_): proba[te, np.searchsorted(classes, c)] = p[:, j]
        pred[te] = m.predict(X[te])
    p = np.clip(proba, 1e-6, None); p /= p.sum(1, keepdims=True)
    imp = dict(sorted(zip(feats, rf().fit(X, y).feature_importances_), key=lambda kv: -kv[1]))
    print(f"{tag:42s} log-loss {log_loss(y,p,labels=classes):.3f}  acc {accuracy_score(y,pred):.3f}"
          f"  n={len(d):,}  imp=" + ", ".join(f"{k} {v:.2f}" for k,v in imp.items()))

print("\n=== PREDICTIVE: phy collapse/normal/bloom, NORTH >=28N (random ~1.10) ===")
run(df[north], ["mhw", "mhw_int"],          "MHW features (flag+intensity)")
run(df[north], ["temp_z"],                  "raw temp_z baseline (for comparison)")
run(df[north], ["mhw", "mhw_int", "temp_z"],"MHW + temp_z")
print("\n=== same, SOUTH <28N (control) ===")
run(df[~north], ["mhw", "mhw_int"],         "MHW features (flag+intensity)")
