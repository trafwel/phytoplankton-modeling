# Subarctic NW Pacific (Oyashio/Kamchatka) — Small Test Run: File Checklist

Goal: run the OFES-BGC pipeline end to end on a small subarctic NW Pacific box to
confirm the plumbing (data → features → target → model) before scaling up.
(Region changed from Kuroshio — too many overlapping oceanic phenomena there. The
subarctic HNLC region has a cleaner, mixing/light-driven seasonal bloom.)

Test region (approx, verify vs grid file): **40–60°N, 150–180°E** →
`LAT_IDX = (600, 800)`, `LON_IDX = (500, 800)`.
First plumbing pass: smaller sub-box `LAT_IDX = (650, 700)`, `LON_IDX = (550, 600)`.

---

## Files — status & action

### ✅ Already written (reuse; set config only)
- [x] `src/read_ofes_bgc.py` — `.dta` reader. Tested. **No change.**
- [x] `src/copy_ofes_to_drive.py` — stage data onto the 5 TB drive.
      **Set** `SRC_DIR`, `DST_DIR`; run with `DRY_RUN=True` then `False`.
- [x] `src/step1_bgc_series.py` — build daily temp+phy series for the box.
      **Set** `VAR_DIRS` (drive paths), `LAT_IDX=(520,620)`, `LON_IDX=(420,600)`,
      `START/END` (a few years is plenty for a test, e.g. 2000–2005).
- [x] `src/step2_bgc_table.py` — MHW + anomalies → `model_table.parquet`.
      **Set** `CLIM_PERIOD` to match the span.
- [x] `src/step5_model.py` — RF on `phy_anom`. **No change** (maybe drop to 3 classes
      for the small test — see below).

### ✍️ To write / modify
- [x] **`src/step5_model.py` (tweak)** — added `CLASS_MODE = 3` → `{collapse, normal,
      bloom}` with edges `[-0.75, 0.75]` (set `CLASS_MODE = 5` for the full run).
- [x] **`src/step6_correlate.py` (retarget)** — `RESPONSE` now `phy_anom`.
- [x] **`src/qc_checks.py` (NEW)** — `python qc_checks.py series|table|all`: box maps,
      a cell's temp/phy series, all-NaN-column check, feature stats, target class
      balance + histogram. This is the step that makes it a real test.
- [ ] **`run_kuroshio_test.sh` (NEW, optional)** — one-command chain of
      step1 → qc → step2 → qc → step5 → step6 with the test config. (Say the word.)

### 🔜 Not needed for the test (later)
- [ ] OFES grid lat/lon file → map index box to real degrees (Yasunaka-san).
- [ ] `ntr`/`sw`/`hmxl` mechanism-attribution pass.
- [ ] Satellite validation overlay; subsurface target.

---

## Run order (once data is on the drive)
1. `copy_ofes_to_drive.py`  (stage temp + phy onto the 5 TB drive)
2. `step1_bgc_series.py`    → `data/interim/bgc_series.nc`
3. `qc_checks.py` (stage 1) → confirm the box has ocean, sane temp/phy
4. `step2_bgc_table.py`     → `data/processed/model_table.parquet`
5. `qc_checks.py` (stage 2) → confirm MHW/anomaly columns, class balance
6. `step5_model.py`         → metrics + SHAP + saved model
7. `step6_correlate.py`     → MHW→phy lag/composite

## "Test passed" looks like
- `bgc_series.nc`: box is mostly ocean, temp ~5–30 °C seasonal, phy positive with a
  spring peak.
- `model_table.parquet`: ~(box cells × months) rows; `sst_anom`, `mhw_*`, `phy_anom`
  all present and centered near 0.
- `step5`: runs without error; class balance printed; log-loss beats a trivial
  baseline; SHAP shows MHW features carrying signal.
- No crashes, no all-NaN columns, shapes consistent throughout.
