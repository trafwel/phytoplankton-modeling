#!/usr/bin/env python3
"""
rerun_corrected.py — decisive re-run on the CORRECTED crop (28,300,300) = 10-40N, 120-150E.

Builds the multi-year surface table from the precomputed *_anom fields, then runs the
key tests with year-blocked CV:
  1. FULL REGION: nutrients + temp -> phy   (was log-loss 0.38 on the mis-shaped data)
  2. FULL REGION: TEMP-ONLY   -> phy        (the heatwave test; was 1.12 = worse than random)
  3. NORTH >=28N: TEMP-ONLY   -> phy        (temperate band -- NEVER sampled before)
  4. SOUTH <28N:  TEMP-ONLY   -> phy        (the old warm subset, for comparison)

Run on the Mac (drive attached locally):
    cd "/Users/lukequasar/Documents/Ocean Project"
    source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
    conda activate ocean
    python src/rerun_corrected.py
Requires: numpy, pandas, scikit-learn.  READ-ONLY on the drive.
"""
import numpy as np, pandas as pd, glob, re, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, accuracy_score

def _find_data_dir():
    """Auto-locate OFES_test/Data across /Volumes (drive may remount as HD-SHU3-1 etc.)."""
    import os
    cands = sorted(glob.glob("/Volumes/*/OFES_test/Data"))
    for c in cands:
        if glob.glob(f"{c}/phy/phy_anom.*.dta"):   # must actually contain the anom files
            return c
    hint = ("\n  Volumes present: " + ", ".join(sorted(os.listdir("/Volumes"))) +
            "\n  Is the HD-SHU3 drive connected? (it may have remounted under a new name)")
    raise SystemExit(f"ERROR: could not find OFES_test/Data with phy_anom files.{hint}")

D = _find_data_dir()                        # read-only source
NZ, NY, NX = 28, 300, 300                   # CORRECTED shape (data.ipynb: [0:28,300:600,200:500])
FILL = -5e33
DATE = re.compile(r"\.(\d{2})\.(\d{2})\.(\d{4})\.dta$")
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

def dated(folder, prefix):
    out = {}
    for f in glob.glob(f"{D}/{folder}/{prefix}.*.dta"):
        m = DATE.search(f)
        if m:
            mo, da, yr = map(int, m.groups())
            out[pd.Timestamp(yr, mo, da)] = f
    return dict(sorted(out.items()))

def surf(path):
    a = np.fromfile(path, dtype=">f4")
    a = a.reshape((NZ, NY, NX))[0] if a.size == NZ*NY*NX else a.reshape((NY, NX))
    return np.where(a <= FILL, np.nan, a).astype("float32")

# ---- build table ----
log(f"using data dir: {D}")
dates = list(dated("phy", "phy_anom"))
log(f"dates: {len(dates)} ({dates[0].date()}..{dates[-1].date()})")
cols = {}
for folder, col in [("temp","temp_z"), ("ntr","ntr_z"), ("phy","phy_z")]:
    files = dated(folder, f"{folder}_anom")
    cube = np.full((len(dates), NY, NX), np.nan, np.float32)
    for i, d in enumerate(dates):
        if d in files: cube[i] = surf(files[d])
    s = np.nanstd(cube)
    cols[col] = (cube / s).astype("float32") if s else cube
    log(f"  {folder}_anom -> {col}  (std={s:.3g})")

T = len(dates)
years = np.array([d.year for d in dates], dtype="int16")
df = pd.DataFrame({
    "t":    np.repeat(np.arange(T, dtype="int32"), NY*NX),
    "year": np.repeat(years, NY*NX),
    "lat":  np.tile(np.repeat(np.arange(NY, dtype="int16"), NX), T),
    "lon":  np.tile(np.arange(NX, dtype="int16"), T*NY),
    **{c: cols[c].reshape(-1) for c in cols},
}).dropna(subset=["phy_z"]).reset_index(drop=True)
df["latN"] = (-19.95 + 0.1*(300 + df["lat"].astype("float32"))).astype("float32")
df.to_parquet("OFES Data/model_table_corrected.parquet", index=False)
log(f"table: {len(df):,} rows saved -> OFES Data/model_table_corrected.parquet "
    f"(north>=28N: {(df.latN>=28).mean():.0%})")

EDGES = [-0.75, 0.75]
NAMES = {0:"collapse", 1:"normal", 2:"bloom"}

def add_lags(d, feats, lags=(1,2)):
    d = d.sort_values(["lat","lon","t"]).copy()
    g = d.groupby(["lat","lon"], sort=False)
    fc = list(feats)
    for f in feats:
        for L in lags:
            c = f"{f}_lag{L}"; d[c] = g[f].shift(L); fc.append(c)
    return d.dropna(subset=fc), fc

def rf():
    return RandomForestClassifier(n_estimators=150, min_samples_leaf=5,
                                  class_weight="balanced_subsample", n_jobs=-1,
                                  random_state=0)

def run(d, feats, tag, max_rows=1_500_000):
    d = d.copy()
    d["y"] = np.digitize(d["phy_z"].values, EDGES)
    d, fc = add_lags(d, feats)
    if len(d) > max_rows: d = d.sample(max_rows, random_state=0)
    X, y, grp = d[fc].values, d["y"].values, d["year"].values
    classes = np.array(sorted(np.unique(y)))
    proba = np.zeros((len(y), len(classes))); pred = np.empty(len(y), int)
    for tr, te in GroupKFold(5).split(X, y, grp):
        m = rf().fit(X[tr], y[tr])
        p = m.predict_proba(X[te])
        for j, c in enumerate(m.classes_): proba[te, np.searchsorted(classes, c)] = p[:, j]
        pred[te] = m.predict(X[te])
    p = np.clip(proba, 1e-6, None); p /= p.sum(1, keepdims=True)
    ll, acc = log_loss(y, p, labels=classes), accuracy_score(y, pred)
    bal = {NAMES[c]: f"{(y==c).mean():.0%}" for c in classes}
    imp = dict(sorted(zip(fc, rf().fit(X, y).feature_importances_), key=lambda kv:-kv[1]))
    log(f"=== {tag} (n={len(d):,}) ===")
    log(f"    class balance {bal}")
    log(f"    log-loss {ll:.3f}  accuracy {acc:.3f}   (random 3-class ~1.10)")
    log(f"    importances " + ", ".join(f"{k} {v:.2f}" for k, v in imp.items()))

run(df,                ["ntr_z","temp_z"], "FULL REGION: nutrients+temp -> phy  (was 0.38)")
run(df,                ["temp_z"],         "FULL REGION: TEMP-ONLY -> phy  (heatwave test; was 1.12)")
run(df[df.latN >= 28], ["temp_z"],         "NORTH >=28N: TEMP-ONLY -> phy  (never sampled before)")
run(df[df.latN <  28], ["temp_z"],         "SOUTH <28N:  TEMP-ONLY -> phy  (old warm subset)")
log("DONE")
