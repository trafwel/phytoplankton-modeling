# Test-Model Build — All Files, Breakdowns & Resources

Goal: a running test model — a Random Forest predicting a phytoplankton
collapse/normal/bloom class from temperature (and optional nutrient) features, on
the autumn-2000 cropped data. Single season → **daily** resolution, **no Hobday MHW**
(too little data), simple **per-cell z-score** anomalies.

Data on disk: `data/raw/ofes_test/{temp,phy,ntr}/` — 62 autumn-2000 daily files each,
shape `(105, 150, 160)`.

## The files (4 total)

| # | File | Status | Role |
|---|---|---|---|
| 1 | `src/step1b_series_cropped.py` | ✅ written (fallback) | `.dta` → `bgc_series.nc` cube |
| 2 | `src/step2b_daily_table.py` | ✍️ **to write** | cube → tidy feature/target table |
| 3 | `src/step5_model.py` | ✅ exists — 3 config edits | train + evaluate the RF |
| 4 | `src/qc_checks.py` | ✅ exists — 1 small tweak | verify each stage |

Run order: **1 → qc(series) → 2 → qc(table) → 3.**

---

## File 1 — `step1b_series_cropped.py`  (built; understand it)

**Purpose:** turn the many single-day `.dta` files into one `(time, y, x)` cube.
**In:** `data/raw/ofes_test/{temp,phy,ntr}/*.dta` → **Out:** `data/interim/bgc_series.nc`.

**Breakdown:**
1. Read one file: `np.fromfile(path, dtype=">f4").reshape((105,150,160))[0]` (surface),
   then mask `<= -5e33` → `NaN`.
2. Glob files, parse the date from each filename (regex `\.(\d{2})\.(\d{2})\.(\d{4})\.dta$`).
3. Pre-allocate `(n_dates,150,160)` NaN cubes, fill per day.
4. Wrap in `xarray.Dataset` (dims `time,y,x`; coords = dates + index arrays), `to_netcdf`.

**Gotchas:** big-endian (`>f4`) or values are garbage; C-order reshape; land sentinel
must become NaN; parse dates (files are sparse/autumn-only) rather than assume a range.

**Resources:**
- [NumPy `fromfile` / binary I/O](https://numpy.org/doc/stable/reference/generated/numpy.fromfile.html) + [byte order / dtypes](https://numpy.org/doc/stable/reference/arrays.dtypes.html#specifying-and-constructing-data-types)
- [Pythia — Introduction to Xarray](https://foundations.projectpythia.org/core/xarray/xarray-intro/)
- [Abernathey — Earth & Environmental Data Science](https://earth-env-data-science.github.io/intro.html)

---

## File 2 — `step2b_daily_table.py`  ✍️ THE ONE TO WRITE

**Purpose:** turn the cube into a tidy ML table — one row per (cell, day), with
temperature features and a phytoplankton target, all as per-cell anomalies.
**In:** `data/interim/bgc_series.nc` → **Out:** `data/processed/model_table.parquet`.

**Breakdown (step by step):**
1. Open the cube (`xr.open_dataset`).
2. **Per-cell z-scores over time** — the anomaly definition for a single season.
   For a field `da`: `z = (da - da.mean("time")) / da.std("time")`. This removes each
   cell's own mean/spread, leaving its temporal fluctuations (the only "anomaly" you
   can define with one season).
   - `temp_z` from `temp`.
   - `phy_z` from `log10(phy.where(phy>0))` — log first (phytoplankton is log-normal).
   - `ntr_z` from `ntr` (optional feature).
3. Merge those into one Dataset on `(time,y,x)`.
4. **Cube → tidy table:** `.to_dataframe().reset_index()` (one row per time/y/x).
5. Rename `y→lat`, `x→lon` so `step5` works unchanged (it groups on `lat`/`lon`).
6. Drop land rows: `dropna(subset=["phy_z"])`.
7. Save parquet.

Note: you do **not** compute lags here — `step5`'s `build_features` adds the 1–3 day
lags from the feature columns. `step2b` just outputs `temp_z`, `ntr_z` (features) and
`phy_z` (target).

**Gotchas:** log-transform phy before z-scoring; guard `std==0` cells (they become
NaN — fine, they drop out); keep `lat/lon/time` as columns for blocked CV.

**Resources:**
- [Pythia — Computations & Masks with Xarray](https://foundations.projectpythia.org/core/xarray/computation-masking/) (mean/std over a dim, `.where`)
- [xarray — `to_dataframe`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.to_dataframe.html)
- [pandas — reshaping / tidy data](https://pandas.pydata.org/docs/user_guide/reshaping.html)
- z-score / standardization concept: [scikit-learn preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling)

---

## File 3 — `step5_model.py`  (exists — 3 small edits)

**Purpose:** bin the target into classes, train the RF with spatial-block CV, evaluate,
interpret (SHAP), save. **In:** `model_table.parquet` → **Out:** metrics + `models/`.

**Edits for this test:**
1. `TARGET_BASE = "phy_z"` (the per-cell z-score target from File 2).
2. `FEATURES = ["temp_z", "ntr_z"]` (drop the old `sst_anom`/`mhw_*` names).
3. `CLASS_MODE = 3` (already added) and `BLOCK_DEG` set for **index units** — `lat`/`lon`
   here are grid indices 0–149 / 0–159, so use e.g. `BLOCK_DEG = 40` (≈ 4° blocks).

**Breakdown (unchanged machinery):** `label_severity` bins `phy_z` into
{collapse, normal, bloom}; `build_features` adds temp lags; `out_of_fold` runs
spatial-block CV; `evaluate` prints log-loss / precision-recall / confusion matrix;
`interpret` does importances + SHAP; `fit_final_and_save` calibrates and saves.

**Resources:**
- [scikit-learn RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [scikit-learn — cross-validation / GroupKFold](https://scikit-learn.org/stable/modules/cross_validation.html#group-k-fold)
- [scikit-learn — probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [SHAP docs](https://shap.readthedocs.io/) · concept: *Intro to Statistical Learning* Ch.8 (statlearning.com)

---

## File 4 — `qc_checks.py`  (exists — 1 tweak)

**Purpose:** the verification step; look before trusting. **In:** the interim/processed
files → **Out:** printouts + `figures/`.

**Tweak:** the `table` check currently bins `phy_anom`; point it at `phy_z` (or just
read whatever target column File 2 wrote). The `series` check works as-is.

**Breakdown:** `series` → box maps + a cell's temp/phy trace; `table` → all-NaN-column
check, feature stats, target class balance + histogram.

**Resources:**
- [Matplotlib — pyplot tutorial](https://matplotlib.org/stable/tutorials/pyplot.html)
- [xarray — plotting](https://docs.xarray.dev/en/stable/user-guide/plotting.html)

---

## "Test passed" looks like
- `bgc_series.nc`: `(time≈62, 150, 160)`, warm coherent temp map, phy trace sensible.
- `model_table.parquet`: ~(ocean cells × 62) rows; `temp_z`, `ntr_z`, `phy_z` present,
  centered near 0; no all-NaN columns.
- `step5`: runs; class balance printed (is "collapse" present?); log-loss beats a
  trivial baseline; SHAP shows whether temp/its lags carry signal.

## Honest caveats (write these down)
- **Single season** → the anomalies and any "skill" are plumbing validation, not
  science. Real results need multiple autumns.
- **Circularity** → temp and phy both come from the coupled model, so the RF is partly
  re-learning the model's internal relationship. Frame as a proof-of-concept/emulator.

## Consolidated learning path
Read in this order, trying each idiom on your own data as you go:
Pythia Xarray-Intro → Pythia Computations-&-Masks (covers Files 1–2) →
pandas reshaping (File 2 table) → scikit-learn RF page + GroupKFold (File 3) →
skim SHAP (File 3 interpret).
