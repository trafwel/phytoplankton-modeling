# Marine Heatwaves and Phytoplankton — Collated Results

**Project:** forecasting / characterizing phytoplankton response to marine heatwaves in the
NW Pacific. **Region:** 10–40°N, 120–150°E (Kuroshio / subtropical gyre / Kuroshio–Oyashio
transition). This document collates the results to date across the two data streams —
**OFES-BGC** (free-running coupled model) and **JCOPE** (observation-constrained reanalysis).

---

## Executive summary

1. **Marine heatwaves suppress phytoplankton.** In the observation-constrained JCOPE
   reanalysis, phytoplankton is **~0.4σ lower on marine-heatwave days** than on calm days —
   a moderate, robust, basin-wide effect. This is the collapse-direction signal the project
   set out to test.
2. **The effect is strongly seasonal:** strongest in **spring and autumn** (−0.5 to −0.67σ),
   weakest in **winter** (−0.2σ) — the bloom/transition seasons are most heatwave-sensitive.
3. **The free-running model understates it.** OFES-BGC showed only ~0.2σ and, being
   October-only, saw just the autumn slice. JCOPE shows the true magnitude is ~2× larger and
   reveals the full seasonal cycle.
4. **Nutrient control is real but not as deterministic as the model implied.** OFES ties
   phytoplankton to nutrients at r≈0.9 — partly circular (the coupled model computes
   phytoplankton *from* nutrients). Observation-constrained JCOPE shows a more moderate
   r≈0.33, leaving genuine room for temperature.
5. **Spatial structure** is subtropical-suppression vs deep-tropical (~10–15°N) enhancement —
   not a clean north/south split.
6. **Vertical redistribution, not destruction.** The suppression is surface-trapped (0–50 m)
   and reverses to subsurface enhancement at ~75–100 m (the deep chlorophyll maximum). Heatwaves
   shift phytoplankton downward rather than emptying the column — so satellite (surface) views
   overstate the impact.
7. **A skillful 7-day early-warning forecast (the deliverable).** A calibrated model forecasts
   phytoplankton collapse probability 7 days ahead with **positive Brier Skill Score over both
   climatology (+0.16) and persistence (+0.19)** — a genuine, useful forecast. A marine heatwave
   raises next-week collapse risk from **14.5% → 20.4%** (~40% relative), which the model captures.

---

## Part 1 — OFES-BGC (free-running coupled model)

### 1.1 Data and the crop-shape correction
The coupled biology (OFES-BGC, Sasai NPZD) exists only as a cropped region of the OFES2
North Pacific hindcast. The crop was **mis-shaped for much of the work** as `(105,150,160)`;
the lab's own extraction code (`script_test/data.ipynb`) defines it as
`data_in[0:28, 300:600, 200:500]` → true shape **`(28, 300, 300)`** = **10–40°N, 120–150°E**,
upper ~150 m. The wrong reshape had the same byte count, which hid the error and produced
diagonal NaN artifacts. Correcting it revealed the region spans tropical-south to
temperate-north, not the small subtropical box originally assumed. The crop temperature is
**bit-identical to `OFES2_NP[0,300:600,200:500]`** — same run, so the full-year 2000–2019
physics temperature is exactly what the biology experienced.

### 1.2 Corrected model results (October, 2000–2019)
Random-forest, year-blocked CV, phytoplankton collapse/normal/bloom (random baseline
log-loss ≈ 1.10):

| test | log-loss | note |
|---|---|---|
| nutrients + temp → phy | 0.479 | nutrients ~86% importance |
| temperature only → phy | 1.028 | beats random (was 1.12 on mis-shaped data) |
| north ≥28°N, temp only | 0.988 | weak temperature signal appears |
| south <28°N, temp only | 1.045 | ≈ random |

Nutrients dominate; temperature alone carries a weak, latitude-dependent signal. A per-cell
correlation heatmap showed nutrients↔phy strong (r≈0.8–0.9) everywhere, temp↔phy weak and
mostly negative (~−0.2). In the **Kuroshio–Oyashio transition** (NE corner) temp↔phy is
strongly negative (mean −0.20), i.e. warm anomalies suppress blooms via stratification.

### 1.3 Marine heatwaves on OFES (October-only)
A Hobday MHW analysis was possible only within October (the biology is October-only, except
Sep 2018). Composite: phytoplankton **~0.2σ lower on MHW days**, basin-wide, roughly equal
north/south. Encoding heatwaves as discrete events was *not* predictive per cell-day
(log-loss ≈ random) — the effect is a real but small graded drag, not a forecastable trigger.

### 1.4 OFES caveats
Coupled model → **circularity** (phytoplankton computed from nutrients/temp/light);
October-only biology; model output with no in-dataset observational validation.

---

## Part 2 — Data inventory (verified from the drives)

- **OFES-BGC biology** (`phy`,`ntr`): October 2000–2019 **plus September 2018** only. The
  precomputed anomaly files (hence the model table) are October-only.
- **OFES2_NP physics** (`temp`,`w`): **full-year daily, 2000–2019**; currents (`u`,`v`)
  2000–2006 only. This continuous record is what enables a proper seasonal MHW threshold.
- **JCOPE-EC** (drive `HD-SHU`): a **data-assimilating reanalysis** — see Part 3.

---

## Part 3 — JCOPE-EC (observation-constrained reanalysis)

### 3.1 What it is
JCOPE-EC (JAMSTEC, Miyazawa) assimilates satellite SST, altimetry, and in-situ observations
— so it is anchored to reality, unlike free-running OFES. **Grid:** 866×620×25, **1/12°
(~9 km)**, 108–180°E / 10.5–62°N (covers our region *and the full Oyashio*). **Daily,
~1993–2018.** Variables: `t`,`s`,`phy`,`zoo`,`det`,`din`(nitrogen),`dip`,`dic`,`alk`,`ph`,
`pco2`. Stored as ~600 GB GrADS tar archives (~5.5 TB total).

### 3.2 Extraction and sanity
`src/jcope_extract_pilot.py` / `jcope_extract_full.py` extract `t`/`phy`/`din`, region-
cropped, to HD-PGF-A (reformatted exFAT). Units (inferred from ranges, all sane): `t` in °C,
`phy` ≈ mmol N/m³, `din` ≈ mmol N/m³ nitrate. The reanalysis independently reproduces the
warm/nutrient-poor south vs cool/nutrient-rich north contrast.

### 3.3 JCOPE vs OFES validation (Oct 2000–2004)

| metric (N / S) | JCOPE (obs) | OFES (model) |
|---|---|---|
| corr(phy, nutrients) | +0.32 / +0.35 | +0.86 / +0.90 |
| corr(phy, temp) | −0.18 / −0.53 | −0.15 / −0.18 |
| warm-day phy drop | −0.50 / −0.83 | −0.21 / −0.43 |

Directional agreement (validation passes). But OFES's nutrient link is partly **circular**
(0.9 vs 0.33), and the **heatwave suppression is ~2× stronger in observation-constrained
data**.

### 3.4 All-season Hobday marine-heatwave analysis (2010–2018)
Proper seasonal Hobday climatology + 90th-pct threshold, ≥5-day events, on the JCOPE surface
record. Suppression = phy_z(MHW) − phy_z(calm), σ units:

| season | north | south |
|---|---|---|
| ALL | −0.41 | −0.38 |
| DJF (winter) | −0.21 | −0.26 |
| MAM (spring) | −0.49 | −0.53 |
| JJA (summer) | −0.30 | −0.23 |
| SON (autumn) | **−0.67** | −0.49 |

MHW frequency ~7–10% of cell-days, elevated along the Kuroshio. **Spatially**, suppression
dominates the domain (deepest in the subtropical gyre / Kuroshio, 20–35°N); a
**tropical far-south band (~10–15°N) shows the opposite — phytoplankton boosted** on MHW days.
Figures: `figures/jcope_mhw_seasonal.png`, `figures/jcope_mhw_maps.png`.

### 3.5 Vertical structure (depth)  ✅  `src/jcope_depth.py`
Suppression at each of the 10 depth levels (MHW defined on surface temp). Figure:
`figures/jcope_mhw_depth.png`.

| depth (m) | north | south |
|---|---|---|
| 0 | −0.41 | −0.38 |
| −10 | −0.35 | −0.27 |
| −20 | −0.29 | −0.13 |
| −30 | −0.25 | −0.10 |
| −50 | −0.09 | −0.08 |
| −75 | **+0.10** | **+0.24** |
| −100 | +0.05 | +0.17 |
| −125…−200 | ≈ 0 | ≈ 0 |

**Key result — vertical redistribution, not destruction.** The suppression is **surface-
trapped** (0–50 m, strongest at surface), crosses zero near **~55 m**, and reverses to
**subsurface enhancement at ~75–100 m** (the deep chlorophyll maximum). A heatwave warms and
stratifies the surface → surface phytoplankton down, subsurface DCM up → the column shifts
downward rather than emptying. Subsurface enhancement is strongest in spring–summer (MAM/JJA).
**Implication:** satellite (surface-only) views overstate the true heatwave impact — depth-
resolved reanalysis is needed to see the compensation.

### 3.6 Matched classifier accuracy: JCOPE vs OFES
Same random forest, same collapse/normal/bloom target, same year-blocked CV, matched period
(Oct 2000–2004 surface, 90k cells). Random baseline log-loss ≈ 1.10 (`jcope_ofes_accuracy.py`):

| features | JCOPE (obs) | OFES (model) |
|---|---|---|
| temp only | log-loss 1.248, acc 0.503 | log-loss 1.275, acc 0.399 |
| temp + nutrients | log-loss 0.890, acc 0.613 | log-loss 0.601, acc **0.740** |

- **Temp-only (non-circular heatwave test):** comparable and both weak (both worse than the
  random log-loss) — temperature alone does not classify phytoplankton state per cell-day in
  either. JCOPE marginally better.
- **Temp+nutrients:** OFES looks far more accurate (0.74 vs 0.61) but that is **circularity** —
  OFES computes phytoplankton *from* nutrients. JCOPE's lower, obs-constrained number is the
  more trustworthy estimate of true predictability.
- **Takeaway:** JCOPE is not meaningfully less accurate; OFES's edge is self-consistency, not
  skill.

**All-season JCOPE classifier** (`jcope_classifier.py`, 2010–2018, 1.5M cell-days, year-blocked CV):

| features | overall | DJF | MAM | JJA | SON |
|---|---|---|---|---|---|
| temp only (log-loss / acc) | 1.193 / 0.44 | 1.375 / 0.40 | 1.286 / 0.42 | 0.955 / 0.50 | 1.160 / 0.45 |
| temp + nutrients | 0.934 / 0.54 | 1.011 / 0.49 | 0.995 / 0.51 | **0.825 / 0.61** | 0.908 / 0.57 |

- **Nutrients carry the skill;** temperature alone is worse than random except in summer.
  temp+nutrients beats baseline every season (honest, non-circular ~0.54 acc — vs OFES's
  inflated 0.74).
- **Summer (JJA) most predictable, winter (DJF) least** — strong summer stratification ties
  phytoplankton more deterministically to temp/nutrients.
- **Predictability ≠ effect size:** composite suppression is largest in spring/autumn, but the
  classifier is most accurate in summer. A large average heatwave response is not the same as
  forecastable per-cell state.
- **Ceiling ≈ 0.54 acc / 0.93 log-loss all-season** — real but modest. Improvement levers (not
  yet applied): temporal **lags** (bloom memory), **subsurface** nutrients + stratification/MLD
  (we have the depth data + salinity), and **spatial/temporal aggregation** (average out
  unpredictable mesoscale eddies). Some error is irreducible (per-cell-day chaos; heatwaves are
  a second-order driver).

---

### 3.7 Improved classifier — feature engineering  ✅  `src/jcope_classifier_v2.py`
Added temporal **lags** (temp & DIN at −7/−14/−28 d), **subsurface** DIN (−50 m),
**stratification** (T_surf − T_−50m), and **season** (sin/cos day-of-year). Per-cell
(1/12°, daily), all-season 2010–2018, year-blocked CV.

| model | overall | DJF | MAM | JJA | SON |
|---|---|---|---|---|---|
| baseline: temp + nutrients | 0.934 / 0.54 | — | — | — | — |
| **improved** (log-loss / acc) | **0.632 / 0.725** | 0.810 / 0.63 | 0.726 / 0.67 | **0.306 / 0.91** | 0.694 / 0.69 |

- **Adding physics ~halved log-loss (0.934→0.632) and raised accuracy 0.54→0.725.**
- **Importances:** stratification 0.19, season (sin+cos) 0.22, the six lags ~0.33 combined,
  subsurface DIN 0.09 — raw surface temp only 0.09. The skill comes from the **nutrient-supply
  physics** (stratification gate + subsurface reservoir + temporal memory + season), NOT direct
  thermal forcing. Fully consistent with the depth result (§3.5).
- **Summer near-deterministic** (0.306 / 0.91 — a low-variance stratified regime; log-loss
  confirms genuine skill), winter hardest.
- **Caveats:** summer accuracy flattered by class balance; further levers untapped: salinity,
  light, currents (need extraction); coarse-grained (~1°/weekly) scale test still pending.

**Overfitting check** (`src/jcope_holdout.py`, cached table; blocked = hold out whole 2° tiles
from every year = leakage-safe; naive = random cell-days):

| eval scheme | improved log-loss / acc | note |
|---|---|---|
| year-blocked 5-fold CV | 0.644 / 0.717 | strictest (predict an unseen year) |
| blocked 80/10/10 TEST | 0.585 / 0.751 | unseen tiles, all years seen |
| naive random TEST | 0.580 / 0.754 | for leakage contrast |
| blocked TRAIN | 0.365 / 0.922 | RF resubstitution optimism (expected) |

- **NOT overfitting:** held-out TEST (0.751) ≥ CV (0.717); val (0.740) agrees — three
  independent estimates cluster at 0.72–0.75. The +0.17 train−test gap is normal RF
  resubstitution optimism (TEST holds at CV level, doesn't collapse below it).
- **NOT leakage-inflated:** blocked TEST (0.751) ≈ naive TEST (0.754) — spatial blocking
  doesn't lower the score, so there is no autocorrelation leakage; skill is real physics that
  generalizes across space, not memorized locations.
- **Honest headline:** ~0.72 acc / ~0.64 log-loss, robust across CV, blocked, and naive splits.
  Year-blocked CV (unseen year) is the strictest and still gives 0.717.

### 3.8 FINAL RESULT — 7-day probabilistic collapse forecast  ✅  `src/jcope_forecast.py`
Features at day t → isotonic-calibrated P(collapse/normal/bloom) at **t+7** (a genuine
early-warning forecast). Out-of-fold, year-blocked, full 1993–2018.

**Forecast skill** (log-loss / acc): overall **0.634 / 0.722** — essentially matching the
nowcast (ocean state is autocorrelated over weeks, so a 7-day lead barely degrades). By season:
DJF 0.897/0.557, MAM 0.700/0.690, **JJA 0.268/0.928**, SON 0.680/0.705.

**Skill vs baselines — P(collapse in 7 days), Brier score (lower better):**

| forecast | Brier | Brier Skill Score |
|---|---|---|
| **model** | **0.1009** | — |
| climatology (seasonal base rate) | 0.1200 | **+0.160** |
| persistence (t+7 = current) | 0.1245 | **+0.190** |

→ The forecast **beats both baselines** (BSS > 0): 16% better than climatology, 19% better than
persistence. A genuine, skillful probabilistic forecast, not just a classifier.

**Heatwave conditioning (central hypothesis, quantified):** collapse rate 7 days after a warm
day (top-10% SST anomaly) = **20.4%** vs **14.5%** after a calm day — a **~40% relative increase**
in collapse risk. The calibrated model reproduces it (P=21.5% vs 15.5%).

**Calibration:** reliability curve on the diagonal (`figures/jcope_forecast_reliability.png`);
slightly under-confident at the high end (predicts 53% where 67% observed → conservative, the
safe direction for early warning). Model saved: `models/jcope_forecast_7d.joblib`.

**This is the thesis deliverable:** a skillful, calibrated, interpretable 7-day probabilistic
early-warning forecast of phytoplankton collapse from observation-constrained reanalysis, with
marine heatwaves as a quantified risk factor.

### 3.9 Verification diagnostics — accuracy hides the real skill  ✅  `src/jcope_diagnostics.py`
Figure: `figures/jcope_diagnostics.png`. Accuracy 0.722 barely beats the "always-normal"
baseline **0.693** and is misleading; the honest evaluation is discrimination + calibration:

- **Confusion matrix:** at argmax the model calls 79% of true collapses "normal" (recall only
  **0.179**) — but collapse↔bloom confusion ≈ 0.03 (never a catastrophic error).
- **Collapse discrimination:** **ROC-AUC 0.843, PR-AUC 0.477** (base rate 0.151 → ~3× lift).
  Sharpness plot: P(collapse) for real collapses (~0.35) is cleanly separated from non-collapses
  (~0). The argmax miss is a rare-class threshold artifact, not absent signal.
- **Operational early-warning trade-off** (flag when P(collapse) > thr):
  thr 0.2 → recall **0.80**, precision 0.34; 0.3 → 0.56 / 0.45; 0.4 → 0.30 / 0.57; 0.5 → 0.09 / 0.67.
  → can catch ~80% of collapses 7 days ahead at a tunable false-alarm rate.
- **Calibration:** Brier decomposition reliability **0.0014** (excellent), resolution 0.0278
  (informative), uncertainty 0.128; reliability on-diagonal in all four seasons.
- **Takeaway:** report the forecast via ROC-AUC / PR-AUC / reliability + the recall–precision
  trade-off, NOT accuracy. It is a well-calibrated, discriminating probabilistic warning tool.

## Synthesis (thesis narrative)

Observation-constrained data shows marine heatwaves exert a **real, moderate (~0.4σ),
strongly seasonal suppression of phytoplankton** in the NW Pacific — strongest in the spring
and autumn bloom seasons, weakest in winter, and reversing to enhancement in the deep
tropics. The free-running OFES model, hampered by circularity and October-only biology,
detected only a weak autumn slice of this and understated its magnitude. Switching to JCOPE
removes the circularity, recovers the full seasonal and spatial structure, and grounds the
result in assimilated observations.

## Code & data reference
- **OFES:** `read_ofes_bgc.py`, `step1b_multiyear_*.py`, `step5_model.py`, `rerun_corrected.py`,
  `fig_north_south.py`, `fig_heatmap.py`, `fig_oyashio.py`, `jcope_vs_ofes.py`.
- **JCOPE:** `jcope_extract_pilot.py`, `jcope_extract_full.py`, `jcope_hobday.py`, `jcope_depth.py`.
- **Data dictionary:** `docs/ofes_data_reference.md`. **Run log:** `docs/test_results_log.md`.

## Caveats
JCOPE all-season result is 2010–2018 so far (extending to full 1993–2018); surface (depth
queued); phytoplankton std small (0.052 mmol N/m³) so σ-scaling amplifies. OFES results carry
the circularity caveat. Both are model/reanalysis output; JCOPE is observation-constrained
but still not direct observation.
