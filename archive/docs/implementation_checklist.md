# Implementation Checklist — Forecasting Diatom/Nano Phytoplankton Collapse & Bloom from Marine Heatwaves

**Locked decisions:** OFES (Indian Ocean, physical → `thetao` for MHW) + satellite PFT (diatom/nano chlorophyll target & validation) · Indian Ocean scope · multi-day OFES series for a Hobday MHW baseline · interpretable ML (Random Forest primary).

**Architecture in one line:** OFES Indian Ocean temperature → *Marine-Heatwave model*; satellite PFT diatom/nano chlorophyll → *Phytoplankton-anomaly model*; correlate the two and map anomalies onto heatwave conditions.

---

## Phase 0 — Foundations & finishing Week 1 (do this first)

### 0.1 Lock the scientific scope — ✅ LOCKED
- [x] **Region: Arabian Sea, 5°N–25°N, 50°E–75°E.** (Strong seasonal diatom blooms, monsoon upwelling, documented MHWs, major fisheries, excellent OC-CCI coverage.)
- [x] **Study period: 1998–2016** (OC-CCI starts 1997–98; OFES2 hindcast ends 2016 — the overlap is your usable window). Confirm final endpoints once OFES access is settled.
- [x] **Prediction task: probabilistic forecast with severity.** NOT a hard binary. Output two things per (cell, time): (1) a **probability** that a diatom/nano anomaly event (collapse or bloom) occurs, and (2) a **projected severity/magnitude** of that anomaly. Implementation: RF classifier `predict_proba` for the probability + a severity estimate (ordinal severity classes or a magnitude regression) — see Phases 5 & 7.
- [x] **Horizon: nowcast** (same-time association) for v1. Lead-time forecasting moved to stretch goals.

### 0.2 Finish outstanding Week 1 people-tasks — ⏸️ TEMPORARILY COMPLETE (revisit)
- [x] Email Yasunaka-san & Nishizuka-san to confirm: (1) OFES Indian Ocean physics availability & preferred access (APDRC vs internal JAMSTEC), (2) that the diatom/nano split does NOT exist in the OFES bio run you have, (3) whether an Indian-Ocean coupled bio run exists internally.
- [x] Email Professor Jacques (or a TA) with the model plan: two-model design, Random Forest primary, LOOCV vs spatial-block CV question (see 7.4), class-imbalance handling.
- [x] Ask Yuhi-san about the biological pump mechanism (for the write-up's motivation/mechanism section).
- [x] Ask about general chlorophyll surveying + CO₂ surveyance methods (validation framing).
- [x] (Optional) Ask Sakada-san about subsurface chlorophyll maximum as an indicator — relevant to the photic-zone question in 3.5.

### 0.3 Reconcile the spec — ✅ DONE
- [x] **Updated spec written:** `docs/thesis_spec_updated.md` — supersedes the original; OFES+PFT architecture, Arabian Sea scope, satellite diatom/nano target, probability+severity RF, spatial CV, validation anchors (2010 MHW, Noctiluca regime shift). Original planning todos folded into "open questions / stretch goals."

---

## Phase 1 — Data acquisition

### 1.1 OFES Indian Ocean physical data (for MHW detection) — 🟢 IN PROGRESS
- [x] **Access route confirmed & tested live:** APDRC OPeNDAP. Endpoint: `http://apdrc.soest.hawaii.edu/dods/public_ofes/OfES/ncep_0.1_global_3day/temp`. (The dataset browser lives under `.../OfES/`, not `.../OfesData/`.)
- [x] **Product/resolution chosen: OFES 0.1° NCEP hindcast, 3-day, 3D `temp`.** Reason the plan changed from 0.5°: the 0.5° product is **surface-only and ends 2004** (too short); the 0.1° 3D product runs **1980–2014 (public)**, giving a 35-yr baseline — far better for the Hobday climatology. We subset the Arabian Sea box via OPeNDAP so the 0.1° size is not a problem.
- [x] **Grid metadata captured:** `temp[time=5600][lev=54][lat=1500][lon=3600]`, Float32. lat −74.95..74.95 @0.1°; lon 0.05..359.95 @0.1° (0–360); lev surface = index 0 (2.5 m); time 3-day `03jan1980`→`29dec2025`; `_FillValue = -1.0E34` (land).
- [x] **Temperature variable decided:** surface level (`lev[0]`, 2.5 m) as SST proxy for Hobday MHW. Note the cadence is **3-day, not daily** — flag this caveat in Phase 4 (standard Hobday assumes daily; we adapt the window/percentile accordingly).
- [x] **Live sanity-check passed:** pulled the Arabian Sea surface box (~2010); returns realistic 26–28 °C over ocean, `-1.0E34` over land. Endpoint, box indices, and fill value all verified.
- [x] **Acquisition walkthrough written:** `docs/1.1_ofes_acquisition_walkthrough.md` — verified constants table + step-by-step.
- [x] **Loader written (by request):** `get_ofes.py` — subsets Arabian Sea surface temp **1998–2014, 3-day**, one year per OPeNDAP request with retry/resume, masks land, outputs var `sst` on `lat/lon/time` → `data/interim/ofes_arabiansea_sst_1998_2014.nc` (exactly what step3_3 + step4 read).
- [ ] **Run it** (`pip install xarray netCDF4 pydap numpy pandas`, then `python get_ofes.py`). APDRC is slow; resumes if interrupted.
- [ ] After download, confirm the merged file opens, plot one SST map to eyeball the coastline/box.
- [ ] (Optional aux drivers, same endpoint/grid) also available if wanted later: `hmxl` (mixed-layer depth), `hflx` (heat flux), `salt`.

### 1.2 Satellite PFT data (diatom/nano chlorophyll — target + validation) — 🟢 IN PROGRESS
- [x] **Product chosen & verified: `OCEANCOLOUR_GLO_BGC_L4_MY_009_104`** (Copernicus-GlobColour, global, 4 km, 1 Sep 1997–present, NetCDF-4, DOI 10.48670/moi-00281).
- [x] **Diatom/nano confirmed as separate variables:** `DIATO` (= your DCHL) and `NANO` (= your NCHL), both "expressed as chlorophyll" — no haptophyte proxy needed. Product also has `MICRO/PICO/HAPTO/DINO/GREEN/PROKAR/PROCHLO` + total `CHL`.
- [x] **Cadence pinned: monthly (P1M)** — the PFT split is a monthly product. **dataset_id confirmed against the catalog:** `cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M`.
- [x] **Lon convention noted:** −180/180 (Arabian Sea = 50–75, no 0–360 flip needed — differs from OFES).
- [x] **Acquisition walkthrough written:** `docs/1.2_pft_acquisition_walkthrough.md` — verified table + steps + CLI/Python subset templates (you implement, per no-code-first rule).
- [ ] Register free CMEMS account; `pip install copernicusmarine`; `copernicusmarine login`.
- [x] Confirmed via `describe`: dataset_id `cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M`; variables `DIATO`, `NANO`, `CHL` present (+ `DIATO_uncertainty`/`NANO_uncertainty` for Phase 8). Version 202603, −180/180 grid, from Sept 1997.
- [x] **Dry-run validated** (2026-07-07): resolves to DIATO/NANO/CHL, box 50.02–74.98°E × 5.02–24.98°N, 1998-01→2014-12 (204 months), 673 MB file.
- [x] **Added `DINO`** to the pull (dinoflagellates = the diatom's replacement in the Arabian Sea regime shift) → enables diatom-dominance index `DIATO/(DIATO+DINO)` + regime-shift validation. Vars now `DIATO NANO DINO CHL`.
- [ ] Run the real subset → `data/raw/pft/` (now have 5 TB disk). Space-saving options if ever needed: drop `CHL`, skip uncertainty layers. Command ready in the walkthrough.
- [ ] Inspect: dims ~(204 months, lat, lon), plot one `DIATO` map (expect Oman/Somalia upwelling max), build `dchl_nchl_ratio`.
- [ ] Keep any per-pixel uncertainty/flag layer for Phase 8 weighting.

### 1.3 (Optional) PISCES model PFT cross-check for the SINTEX comparison
- [ ] From **Copernicus `GLOBAL_MULTIYEAR_BGC_001_029`** (PISCES: same nano+diatom types as SINTEX-F2bio), check the Product User Manual for whether per-PFT diatom/nano chlorophyll are exposed as download variables (standard product may only give total chl + phytoplankton carbon).
- [ ] If exposed, download for the region — this is your model-consistent bridge to a later SINTEX-F comparison. If not, skip; satellite PFT carries the project.

### 1.4 (Optional) Auxiliary drivers
- [ ] If OFES provides them for the Indian Ocean box: mixed-layer depth, shortwave/heat flux, nutrients (nitrate/silicate). Silicate is especially diagnostic for diatoms. These become model features later.

---

## Phase 2 — Environment & tooling — ✅ COMPLETE · walkthrough: `docs/2_environment_walkthrough.md`

- [x] Core acquisition stack confirmed working (Python, xarray, copernicusmarine, numpy, pandas, netCDF4, dask) via the 1.2 dry-run.
- [x] Dedicated conda env `ocean` created; scientific stack + `xesmf`/`esmpy` installed via conda-forge.
- [x] MHW + interpretability helpers installed: `marineHeatWaves` (from GitHub — not on PyPI) + `shap`.
- [x] Create a clean Python env (conda/venv). Core stack: `numpy`, `xarray`, `dask`, `netCDF4`, `pandas`, `scipy`, `matplotlib`, `cartopy`, `scikit-learn`.
- [x] Add MHW + geospatial helpers: `marineHeatWaves` (GitHub), `xesmf` (regridding), `shap` (interpretability).
- [ ] Set up a project repo/folder structure: `data/raw`, `data/interim`, `data/processed`, `notebooks`, `src`, `figures`, `models`.
- [ ] Write a reusable **OFES loader** — but note: APDRC NetCDF uses `xarray`, NOT the `>f4` `.dta` `np.fromfile` recipe from your NW-Pacific files. Keep both loaders; label clearly.
- [ ] (Optional) Install **NASA Panoply** for quick NetCDF visual QC.

---

## Phase 3 — Preprocessing & harmonization (the real work of Week 2)

### 3.1 Subset & mask — 🟢 CODED (in `src/step3_3_temporal_align.py`) · walkthrough: `docs/3_preprocessing_full_walkthrough.md`
- [x] Masking coded: OFES land `-1.0E34` → NaN via `.where(sst > -5e33)` at load. (PFT `_FillValue` applied by xarray.)
- [ ] Confirm both clipped to the identical box (downloads already are); verify land shows NaN when you run it.

### 3.2 Spatial regridding to a common grid — 🟢 SPEC'D · walkthrough: `docs/3.2_regridding_walkthrough.md`
- [x] **Target grid decided: OFES 0.1°** (the coarser one — fine→coarse avoids inventing resolution). Resolves the "mapping resolution" open question.
- [x] **Method decided: conservative (area-weighted)** — correct for fine→coarse + a concentration field; needs cell bounds (xesmf can build them).
- [x] **Longitude convention flagged:** OFES 0–360 vs PFT −180/180; no-op for the 50–75°E box but handled explicitly.
- [x] **Script written (by request):** `src/regrid_pft_to_ofes.py` — conservative regrid, computes cell bounds, saves/reuses weights, NaN-aware, writes before/after check figure. (Runs once data is on disk; verify OFES coord names in the CONFIG block first.)
- [ ] Regrid PFT `DIATO`/`NANO` (4 km) → OFES grid via `xesmf` conservative; save the weights file for reuse.
- [ ] Handle NaN/land mask carefully (`skipna`/`na_thres`) so coastal upwelling cells aren't corrupted.
- [ ] Verify: before/after `DIATO` maps side by side, domain-mean preserved, coastline/land mask aligns. Save → `data/interim/pft_on_ofes_grid.nc`.

### 3.3 Temporal alignment — 🟢 SPEC'D · walkthrough: `docs/3.3_temporal_alignment_walkthrough.md`
- [x] **Common cadence = MONTHLY** (PFT-bound), 1998-01…2014-12 = **204 steps/cell**.
- [x] **Branch identified:** monthly-mean OFES SST for the anomaly *feature*, BUT keep the **3-day OFES intact** for Phase 4 MHW detection — averaging to monthly first would destroy the sub-monthly structure Hobday needs.
- [x] **Month-anchor gotcha flagged:** `resample("1M")` labels month-END, `"MS"` month-START, PFT has its own stamp — mismatched labels silently yield all-NaN joins. Fix: normalize both to a year-month key (PeriodIndex / first-of-month) before aligning.
- [x] **Coded:** `src/step3_3_temporal_align.py` — masks OFES, monthly-mean, month-start normalize, trim to 204 months, saves `sst_monthly_1998_2014.nc`. (Run once data on disk.)
- [ ] (After Phase 4) aggregate MHW metrics to monthly features and rejoin the 204-month index.
- [ ] PFT L4 is gap-filled already, so cloud-gap handling is light; still keep/inspect the uncertainty layer.

### 3.4 Climatology & anomalies — 🟢 SPEC'D · `docs/3_preprocessing_full_walkthrough.md`
- [x] **Two climatologies kept distinct:** Phase 4's daily 90th-pct MHW threshold vs THIS monthly anomaly climatology (mean per calendar month) for the feature/target track.
- [x] **Log-transform chlorophyll first** (`log10`) — DIATO/NANO are log-normal; anomalies on logged values are symmetric/standard.
- [x] **Coded:** `src/step3_4_anomalies.py` — log10 chl, monthly climatology + anomalies (z-scored), DCHL/NCHL ratio, saves `anomalies_monthly.nc`.

### 3.5 Photic-zone / depth decision (your open question)
- [ ] Satellite PFT is inherently a **surface/near-surface (first optical depth)** signal — so for consistency, use OFES **surface temperature** as the MHW driver. This side-steps needing full photic-zone coverage.
- [ ] Note the limitation in the write-up: subsurface chlorophyll maxima are not captured by surface ocean colour (this is the Sakada-san question). Flag as a known scope boundary, revisit only if time allows.

### 3.6 Build the analysis dataframe — 🟢 SPEC'D · `docs/3_preprocessing_full_walkthrough.md`
- [x] **Coded:** `src/step3_6_build_table.py` — merges monthly fields, cube→tidy table, drops land, keeps lat/lon/time as columns, saves `data/processed/model_table.parquet` (optionally folds in Phase 4 MHW features if present). Needs `pip install pyarrow`.

---

## Phase 4 — Marine-Heatwave detection — 🟢 SPEC'D · walkthrough: `docs/4_marine_heatwave_detection_walkthrough.md`

- [x] **Definition locked (Hobday et al. 2016):** seasonally-varying climatology + per-day-of-year 90th-percentile threshold + ≥5 consecutive days (gaps ≤2 merged). Deterministic — **no ML/training** (resolves the "pull a model" note); MHW metrics are engineered *features* for Phase 5/7.
- [x] **Adaptation 1 decided:** OFES is 3-day, Hobday assumes daily → **interpolate 3-day → daily, then run the standard algorithm** (keeps literature comparability; note as minor smoothing).
- [x] **Adaptation 2 decided:** use a **long climatology baseline** (e.g. 1983–2012 from the full OFES 1980–2014 record), even though the analysis window is 1998–2014; state it in methods (fixed baseline + warming trend → more MHWs later = expected).
- [x] **Coded:** `src/step4_mhw_features.py` — per-cell 3-day→daily interpolation, `marineHeatWaves.detect` (clim 1983–2012), daily→monthly aggregation into `mhw_days`/`mhw_intensity_max`/`mhw_intensity_cum`, saved as `mhw_monthly_features.nc` (1998–2014). Feature names match the (now-uncommented) `FEATURES` in `step5_model.py`.
- [ ] ⚠️ **Dependency:** needs a **long 3-day OFES record (ideally 1980–2014)** for a valid 1983–2012 baseline — the current 1.1 download is only 1998–2014. Extend the OFES download start year before running Phase 4, or shorten `CLIM_PERIOD`.
- [ ] Run it (heavy per-cell loop — test one cell first; consider `xmhw` to vectorize if too slow).
- [ ] Validate against known Arabian Sea MHW years (these are *model* MHWs from OFES) — Phase 8.

---

## Phase 5 — Phytoplankton-anomaly model (built from scratch — your core contribution) — 🟢 CODED (block-by-block) · `src/step5_model.py`

**Task (locked in 0.1): probabilistic event forecast + projected severity.** Two coupled outputs: P(anomaly) and its magnitude.

- [x] **Coded `src/step5_model.py`** (Phase 5 + 7 together): load table → 5-class severity labels (z-score bins) → lagged features → year-blocked out-of-fold RF → log-loss/precision-recall/confusion → feature importance + SHAP → calibrated model saved. Runs once `model_table.parquet` exists; uncomment MHW features in `FEATURES` after Phase 4.
- [x] Label design: single **multi-class RF over {strong_collapse, mild_collapse, normal, mild_bloom, strong_bloom}** via z-score cutoffs (±0.5, ±1.5); `predict_proba` = probability + severity in one model.
- [x] Imbalance handled (`class_weight="balanced_subsample"`); year-blocked CV; calibrated (isotonic).
- [ ] Original planning notes retained below for reference / refinement:
- [ ] Define the **event label** for the probability output. Candidates:
  - Collapse = diatom chl anomaly below the p10 threshold; Bloom = above p90; else "normal".
  - Or use the **DCHL/NCHL ratio** crossing a threshold (your spec's "DCHL < NCHL signals failing health" idea).
- [ ] Define the **severity target** for the magnitude output. Options: (a) **ordinal severity classes** (e.g. mild / moderate / severe, analogous to Hobday MHW categories) — keeps interpretability; or (b) a **continuous magnitude regression** on the standardized anomaly. Default: ordinal classes first, continuous as refinement.
- [ ] Decide how the two outputs combine: cleanest is a single **multi-class classifier** over {strong collapse, mild collapse, normal, mild bloom, strong bloom} — its `predict_proba` gives you BOTH the event probability AND the severity distribution in one interpretable model. Consider this the default; fall back to separate probability + severity models only if it underperforms.
- [ ] Handle **class imbalance** (collapses/blooms are rare): stratified sampling, class weights, or precision/recall-focused metrics (not raw accuracy).
- [ ] Assemble the feature matrix: MHW metrics (Phase 4) + temperature anomaly + lagged temperature/MHW features (blooms lag heatwaves — include t−1, t−2 lags) + aux drivers (silicate, MLD) if available.

---

## Phase 6 — Correlate the two models — 🟢 CODED · `src/step6_correlate.py`

- [x] **Coded:** operates on `model_table.parquet` (both sides already merged). Three views: (1) **lag-correlation** corr(heatwave at t−k, diatom anomaly at t) vs lag; (2) **composite** mean diatom anomaly at offsets around heatwave months; (3) **co-occurrence/lift** P(collapse | heatwave t−1) vs baseline. Saves a 2-panel figure + stats CSV.
- [x] This mapping is the thesis payload — MHW → diatom/nano response — and motivates the forecasting framing.
- [ ] Run once `model_table.parquet` (with Phase 4 features) exists; interpret the lag/composite (expect the diatom response to trail the heatwave by ~1–2 months).

---

## Phase 7 — ML model development

- [ ] **Primary model: Random Forest** (interpretable, handles non-linear boundaries, gives **probabilistic output via `predict_proba`** — exactly the probability + severity framing you want — no linearity/independence assumptions).
- [ ] Train the phytoplankton-anomaly model (multi-class severity, per Phase 5) on the feature matrix. Use `predict_proba` as the deliverable output: probability mass across severity classes = "probability of collapse/bloom + projected degree of severity."
- [ ] **Calibrate the probabilities** (`CalibratedClassifierCV` / isotonic) so the reported probabilities are trustworthy — essential when the headline output is a probability, not a label.
- [ ] **Split strategy — important:** your spec's LOOCV/80-10-10 needs care with spatiotemporal data. Rows from neighboring cells/times are correlated, so random splits leak. Use **spatial-block or temporal-block CV** (hold out whole regions or whole years) to get an honest skill estimate. Raise this with Prof. Jacques.
- [ ] Metrics: because the output is probabilistic + severity, use **Brier score / log-loss + a reliability (calibration) curve** for probability quality, **precision/recall/F1 per severity class** and macro-AUC (not accuracy, given imbalance), and a confusion matrix across severity classes to check ordinal errors (is it confusing "severe" with "mild" or with "normal"?).
- [ ] **Interpretability pass:** feature importances + SHAP values → which MHW metrics drive collapse? This is central to your "maximize interpretability" goal.
- [ ] Baseline comparison: persistence / climatology baseline (echoes the persistence-skill idea in your old SINTEX notebook) so you can show the model beats trivial predictors.
- [ ] (Skip the neural-net option — you already flagged the interpretability loss.)

---

## Phase 8 — Validation

- [ ] Cross-check diatom/nano signal against **ESA OC-CCI chlorophyll-a** (independent of the PFT product if you used GlobColour).
- [ ] If you pulled PISCES (1.3), compare model-vs-satellite PFT patterns.
- [ ] Verify MHW events against **documented Arabian Sea heatwave years (in-window: 1998, 2002, 2003, and especially 2010 — the longest on record, ~60–70 days)**; expect a rising MHW-day trend (~20 days/decade). Ref: *Marine Heatwaves in the Arabian Sea*, Ocean Science 18:639 (2022).
- [ ] **Phenomenon check (strong validation):** does the model reproduce the documented **diatom → dinoflagellate (Noctiluca) regime shift** / warming-driven diatom decline in the N. Arabian Sea? This ties a real observed collapse to the model's target.
- [ ] Report honest skill from the blocked CV, with uncertainty (use OC-CCI per-pixel uncertainty where relevant).
- [ ] **Verification task:** re-run the full pipeline on a held-out year end-to-end to confirm no leakage and reproducible numbers.

---

## Phase 9 — Visualization & UI/UX

- [ ] Map MHW intensity and phytoplankton-anomaly fields side by side (Cartopy).
- [ ] Time-series/Hovmöller plots of temperature anomaly vs diatom/nano anomaly for key cells.
- [ ] Feature-importance / SHAP summary figure.
- [ ] Forecast/association map: **predicted collapse/bloom probability** shaded by **projected severity** given MHW conditions (directly visualizes the probability + degree-of-severity output; your "phytoplankton concentration as primary visualizer" goal).
- [ ] (Panoply for quick QC; matplotlib/Cartopy for final figures.)

---

## Phase 10 — Write-up (summary due July 22 — hard deadline)

- [ ] Draft: motivation (carbon pump, O₂, food web) → data (OFES + PFT, Indian Ocean) → methods (Hobday MHW + RF anomaly model) → results (MHW→PFT association, model skill, feature importance) → limitations (surface-only, satellite gaps, no in-situ diatom/nano) → future work (SINTEX comparison, fisheries, global scale).
- [ ] Include the figures from Phase 9.
- [ ] Prepare the presentation (Week 5).

---

## Known issues tracker (from your spec — carried forward)
- [ ] Mapping resolution → resolved by choosing one common target grid in 3.2 (start 0.5°).
- [ ] Temperature-vs-MHW representation → resolved by using surface temperature + Hobday definition (3.5, Phase 4).
- [ ] Photic-zone coverage → resolved by scoping to the surface optical layer; note subsurface as a limitation (3.5).
- [ ] Data assimilation → explicitly out of scope for v1; revisit only if time allows.

## Stretch goals (only if ahead of schedule)
- [ ] Lead-time forecasting (predict PFT anomaly N days after MHW onset).
- [ ] Fisheries-yield overlay.
- [ ] SINTEX-F comparison via the PISCES bridge.
- [ ] Expand beyond the Arabian Sea box.
