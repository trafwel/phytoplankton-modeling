# Thesis Spec (Current — JCOPE probabilistic 7-day forecast)

**Supersedes all prior versions** (including the earlier OFES-BGC-era spec, preserved in git
history). This reflects the pivot to the observation-constrained **JCOPE reanalysis** and a
validated **7-day probabilistic collapse forecast**. Companion: `docs/COLLATED_RESULTS.md`
(full results), `docs/ofes_data_reference.md` (data dictionary), `docs/test_results_log.md`
(run log).

---

## Thesis
Use machine learning to produce a **calibrated probabilistic forecast, with severity, of
phytoplankton population collapses (and blooms) driven by marine heatwaves** in the NW Pacific
— a 7-day early-warning system — trained on the **observation-constrained JCOPE-EC ocean
reanalysis**.

## Motivation
Phytoplankton are the base of the marine food web, a primary oxygen source, and central to the
ocean carbon pump. Their productivity can fall under marine heatwaves (MHWs), which have grown
more frequent and intense with climate change. Forecasting the phytoplankton response to
heatwaves is therefore both scientifically and ecologically important — and a *probabilistic*
early-warning framing is directly useful for monitoring and fisheries.

**On diatoms vs. total phytoplankton.** Ideally the target would be **diatoms** (the dominant
carbon exporters and most heat-sensitive group). But a diatom-resolving target requires either
satellite PFT data (an algorithmic *inference* too uncertain to train on) or a diatom-resolving
model. JCOPE carries a single lumped phytoplankton compartment, so we target **total
phytoplankton**; diatom composition is future work.

## Why JCOPE (the decisive pivot)
The project began on **OFES-BGC**, a *free-running coupled model*. Two structural problems drove
the switch to JCOPE:
1. **Circularity** — OFES-BGC computes phytoplankton mechanistically from temperature/light/
   nutrients, so predicting phytoplankton from those inputs partly re-learns the model's own
   equations (inflating apparent skill; e.g. nutrient→phyto r≈0.9).
2. **Coverage** — the usable OFES-BGC biology was **October-only** (+ Sep 2018), precluding a
   real all-season marine-heatwave analysis.

**JCOPE-EC** (JAMSTEC) is a **data-assimilating reanalysis** (ingests satellite SST, altimetry,
in-situ), so it is anchored to observations — removing the circularity — and provides **daily,
all-season, 1993–2018, 1/12°, upper-200 m** fields including temperature, phytoplankton, and
nutrients. Validation showed JCOPE reproduces the OFES findings qualitatively but with a
realistic (non-circular) nutrient link and a ~2× stronger heatwave suppression signal. OFES-BGC
is retained only as a cross-check.

## Data & methodology
- **Primary source:** JCOPE-EC reanalysis, region **10–40°N, 120–150°E**, upper 200 m (10 levels),
  daily 1993–2018. Variables used: `t` (temperature), `phy` (phytoplankton, ≈ mmol N/m³), `din`
  (dissolved inorganic nitrogen). Extracted from GrADS tar archives → per-day `.npy`.
- **MHW detection:** Hobday et al. (2016) — seasonal 90th-percentile threshold (±5-day window,
  31-day smoothed), events ≥5 consecutive days, gaps ≤2 days merged. Computed on surface temp.
- **Anomalies:** per-day-of-year, per-cell climatology; the standardized phytoplankton anomaly
  defines the target class.
- **Task framing:** a **7-day forecast** — features at day *t* → phytoplankton class at *t+7*.

## Target & features
- **Target:** phytoplankton **collapse / normal / bloom** (edges ±0.75σ of the standardized
  anomaly) at *t+7*. Output is a **calibrated probability per class** (probability + severity).
- **Features (all at day *t*):** surface + subsurface (−50 m) temperature and nutrient anomalies;
  their **7/14/28-day lags** (bloom memory); **stratification** (T_surf − T_−50m, the nutrient-
  supply gate); and **season** (sin/cos of day-of-year).

## Model
- **Random Forest**, multi-class, `class_weight="balanced_subsample"` for rarity.
- **Isotonic calibration** so probabilities are trustworthy (validated by reliability diagram).
- **Interpretable** via feature importances (stratification, season, lags, subsurface nutrients
  dominate; raw surface temperature minor).
- **Saved model:** `models/jcope_forecast_7d.joblib`.

## Validation
- **Cross-validation:** **year-blocked GroupKFold** (predict an entire unseen year) — the strict,
  leakage-safe estimate. Random splits are avoided (spatial/temporal autocorrelation leaks).
- **Overfitting check:** leakage-safe **blocked 80/10/10** (hold out whole 2° tiles from every
  year); TEST ≈ CV and blocked ≈ naive → not overfitting, not leakage-inflated.
- **Skill vs baselines:** **Brier Skill Score** relative to climatology and persistence — the
  metric that makes it a forecast rather than a classifier.
- **Calibration:** reliability diagram + Brier score.

## Key results achieved
- **7-day forecast:** accuracy 0.722 / log-loss 0.634 (≈ nowcast — ocean state is week-scale
  autocorrelated). **Brier Skill Score +0.160 vs climatology, +0.190 vs persistence** → a genuine,
  skillful probabilistic forecast.
- **Heatwave conditioning:** collapse risk 7 days after a warm day **20.4%** vs **14.5%** after a
  calm day (~40% relative increase); the calibrated model captures it.
- **MHW → phytoplankton (all-season, 1993–2018):** heatwaves suppress phytoplankton ~0.4σ,
  strongest in **spring/autumn**, weakest in winter; basin-wide with **deep-tropical (~10–18°N)
  enhancement**.
- **Vertical structure:** suppression is **surface-trapped (0–50 m)** and **reverses to
  subsurface enhancement at ~75–100 m** (deep chlorophyll maximum) — redistribution, not
  destruction; satellite/surface views overstate the impact.
- **Interpretability:** skill driven by **nutrient-supply physics** (stratification, subsurface
  nutrients, memory, season), not direct thermal forcing.

## Limitations (state explicitly)
1. **Total phytoplankton, not diatoms** — loses composition/carbon-export specificity.
2. **Reanalysis, not direct observation** — JCOPE is observation-*constrained* but still a model
   product (though far better anchored than free-running OFES).
3. **Modest per-cell-day predictability** — ~0.72 accuracy; some error is irreducible mesoscale
   chaos. Marine heatwaves are a real but second-order driver on top of nutrient control.
4. **Region-specific** (NW Pacific 10–40°N); generalization untested.

## Remaining / optional (polish, not core science)
- Severity layer (5-class, or probability paired with expected magnitude).
- Additional features needing extraction: salinity, light (PAR), currents.
- Spatial map products (in progress: `jcope_maps.py`, `jcope_forecast_maps.py`).
- Decadal-trend analysis (is southern suppression intensifying?).
- Write-up: `COLLATED_RESULTS.md` is the results-chapter skeleton.

## Status
**Core thesis result complete:** a validated, calibrated, interpretable 7-day probabilistic
collapse forecast with positive skill over climatology and persistence, plus the supporting
mechanism (seasonal + vertical MHW→phytoplankton structure). Remaining work is visualization and
write-up.

---

## Timeline of actions (chronological)

**Phase 0 — Framing & early pivots**
1. Original concept: forecast diatom collapse/bloom from marine heatwaves as probability +
   severity, interpretable ML.
2. Data/region pivots: satellite PFT + OFES; Arabian Sea → Pacific.
3. **Rejected satellite PFT as a target** (it is an algorithmic inference, not observation) →
   pivot to a self-consistent model, total phytoplankton.

**Phase 1 — OFES-BGC**
4. Adopted OFES-BGC (Sasai NPZD coupled model); built `.dta` loader (`read_ofes_bgc.py`), data
   staged on drive; walkthroughs + implementation checklist written.
5. **Crop-shape correction:** the biology crop was mis-shaped `(105,150,160)`; the lab's
   `data.ipynb` defines it as `data_in[0:28,300:600,200:500]` → true shape **`(28,300,300)`**,
   region **10–40°N, 120–150°E**, upper ~150 m. Verified against the data (clean SST gradient,
   real coastlines).
6. **Corrected multi-year model (October 2000–2019):** nutrients dominate (~86%); temperature
   weak; north (≥28°N) slightly more temp-sensitive than south. Regime maps + Oyashio zoom
   (strong negative temp↔phyto in the transition).
7. **Data-inventory corrections:** biology is October-only **+ September 2018** (not strictly
   October); full-year daily physics temperature exists in `OFES2_NP` (2000–2019); the crop temp
   is **bit-identical to `OFES2_NP` slice** (same run — no mismatch).
8. **October-only Hobday MHW analysis:** ~0.2σ suppression, basin-wide; discrete events not
   predictive per cell — a real but small graded drag.

**Phase 2 — JCOPE discovery & validation**
9. Discovered **JCOPE-EC** (data-assimilating reanalysis) on drive `HD-SHU`; characterized the
   format/grid (866×620×25, 1/12°, daily ~1993–2018, full biogeochem suite).
10. **Pilot extraction** (2000–2004 surface, `t/phy/din`); units/ranges verified sane; reproduces
    the warm/nutrient-poor-south vs cool/nutrient-rich-north contrast.
11. **JCOPE vs OFES validation (Oct 2000–2004):** directional agreement; OFES nutrient link is
    partly circular (r 0.9 vs JCOPE 0.33); heatwave suppression ~2× stronger in obs-constrained data.

**Phase 3 — Full JCOPE analysis**
12. **Storage:** reformatted the empty NTFS drive `HD-PGF-A` to exFAT (writable, 4.6 TB) for the
    extraction output.
13. **Full extraction** of `t/phy/din` upper-200 m to `HD-PGF-A/JCOPE_full/`, run in chunks
    (recent third first, then extended to full **1993–2018**, ~147 GB).
14. **All-season Hobday on JCOPE:** seasonal suppression (spring/autumn strongest, winter weakest);
    MHW frequency maps; validated over the full record.
15. **Depth analysis:** surface-trapped suppression reversing to subsurface enhancement (~75–100 m)
    — vertical redistribution.

**Phase 4 — Predictive model**
16. **Baseline classifier** (temp+nutrients): log-loss 0.934 / acc 0.54.
17. **Improved classifier** (+lags, subsurface, stratification, season): **0.632 / 0.725**;
    importances confirm nutrient-supply physics.
18. **Overfitting/robustness checks:** year-blocked CV; leakage-safe blocked 80/10/10 (TEST ≈ CV;
    blocked ≈ naive) → not overfitting, not leakage-inflated.
19. **Workflow:** feature-table caching (stop re-reading 3,287 files per experiment); warning
    suppression; I/O halving.
20. **Calibrated probabilities** (`jcope_probabilities.py`): reliability + Brier.

**Phase 5 — Final result & products**
21. **7-day probabilistic forecast** (`jcope_forecast.py`): calibrated P(collapse) at t+7;
    **BSS +0.16 vs climatology, +0.19 vs persistence**; heatwave conditioning (14.5%→20.4%);
    reliability on the diagonal; saved model.
22. **Documentation:** `COLLATED_RESULTS.md` and `test_results_log.md` kept current; this spec
    rewritten.
23. **Spatial map products** (in progress): seasonal + decadal suppression maps (`jcope_maps.py`);
    P(collapse) forecast map, skill map, risk-amplification map (`jcope_forecast_maps.py`).

People to confirm with: **Yasunaka-san** (JCOPE/OFES access & provenance), **Prof. Jacques**
(CV/leakage methodology), **Yuhi-san / Sakada-san** (biological pump, subsurface).
