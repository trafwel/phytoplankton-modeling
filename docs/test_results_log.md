# Test Results Log (OFES-BGC phytoplankton model)

Running record of the model runs so far. Data source: OFES-BGC coupled model
(`OFES_test/Data`), NW Pacific crop `(28,300,300)` = **10–40°N, 120–150°E**, **October only**.

> ### ⚠ Shape correction (2026-07-15) — read this before trusting the numbers below
> All runs below (Tests 1–3, Stream A) were built with the **wrong reshape**
> `(105,150,160)`. The lab's extraction code (`script_test/data.ipynb`) shows the true
> shape is **`(28,300,300)`**, region **10–40°N, 120–150°E** (`data_in[0:28,300:600,200:500]`).
>
> **What still holds:** every variable was reshaped *identically*, so a given table row
> paired `temp/ntr/phy` values from the **same flat byte offset = the same true grid cell**.
> Per-cell co-location was preserved, so the *relationships* (nutrients↔phy r≈0.84,
> temp↔phy r≈−0.10, RF importances) are valid **for the cells actually sampled.**
>
> **What was wrong:** taking `[0]` of `(105,150,160)` grabbed flat[0:24000] = the true
> surface (depth 0) but **only the southern 80 of 300 latitude rows** → we sampled only
> **≈10–28°N (the warm, nutrient-limited tropical band)**, never the temperate north
> (28–40°N, SST~20°C). And "`w` level 20" of the wrong shape spanned mixed true depths
> (that's why `w` behaved like noise in A1–A3).
>
> **Net:** "nutrients dominate, heatwaves don't" is validly established **only for the
> warm tropical subset** — precisely where we'd expect temperature not to matter. The
> temperate band where a heatwave→phyto signal is plausible was **never tested.** Re-run
> Tests 2/3 and Stream A on the corrected `(28,300,300)` region before drawing region-wide
> conclusions (loaders already updated). This likely **reopens the heatwave question.**

Target: phytoplankton collapse/normal/bloom (3 ordered classes). Features: temperature
(and nutrient) anomalies + 1–2 step lags. Model: Random Forest, `predict_proba`.

Baseline for context: random 3-class guessing → log-loss ≈ **1.10**, accuracy ≈ 33%.

---

## Test 1 — single month (October 2000)
- **Data:** 31 days, cropped region → 553,622 rows. Per-cell z-score anomalies
  (`temp_z`, `ntr_z`, `phy_z`). (`src/step2b_daily_table.py`)
- **Model:** RF 3-class, subsampled 120k rows, 3 spatial-block folds, 80 trees.
- **Skill:** log-loss **0.50**, accuracy **0.79**, macro-F1 0.78.
- **Class balance:** ~20% collapse / 60% normal / 20% bloom.
- **Feature importance:** `ntr_z` 0.51, `ntr_z_lag1` 0.22, `ntr_z_lag2` 0.10 → **~83% nutrients**;
  `temp_z` 0.06 + lags ~0.05 each → **~17% temperature**.
- **Correlations (visualize_test.py):** `phy_z` vs `ntr_z` **r = 0.84**; vs `temp_z` **r = −0.10**.
- **Read:** nutrients dominate; temperature nearly irrelevant. Proof-of-concept that the
  pipeline runs on real data.

## Test 2 — multi-year (20 Octobers, 2000–2019)
- **Data:** precomputed lab anomaly fields (`*_anom`), pooled-standardized →
  **14,036,180 rows** (13,990,902 after lags), subsampled to 1.5M.
  (`src/step1b_multiyear_anom.py` → `OFES Data/model_table.parquet`)
- **Model:** RF 3-class, 150 trees, **year-blocked CV** (hold out whole years — honest
  test of predicting an unseen year), isotonic-calibrated. (`src/step5_model.py`)
- **Class balance:** collapse **25.8%** / normal **47.7%** / bloom **26.5%**.
- **Skill:** log-loss **0.38**, accuracy **0.84**, macro-F1 **0.84**.
- **Confusion matrix:** errors almost entirely adjacent-class (collapse↔normal,
  normal↔bloom); collapse↔bloom flips ≈ 51 and 400 (near zero) → coherent ordinal structure.
- **Feature importance:** `ntr_z` 0.494, `ntr_z_lag1` 0.254, `ntr_z_lag2` 0.154 → **~90% nutrients**;
  `temp_z` 0.035 + lags ~0.03 → **~10% temperature**.
- **Read:** across 20 years with honest CV, **nutrients dominate even more; temperature
  marginal.** Strong, validated skill — but see circularity caveat.

## Test 3 — temperature only (isolate the heatwave signal)  ✅ DONE — decisive negative
- **Setup:** `FEATURES = ["temp_z", "temp_z_lag1", "temp_z_lag2"]` (nutrients dropped),
  1.5M rows, year-blocked CV. The non-circular "can heatwaves forecast phytoplankton?" test.
- **Skill:** log-loss **1.12** — *worse than the 1.10 random baseline*. Accuracy **0.38**
  (below the always-predict-"normal" rate of 0.48). Confusion matrix is scattered with no
  structure; importances split ~0.33/0.33/0.33 across the three temp features (= noise, no
  signal to concentrate on).
- **Read (decisive):** **temperature alone has essentially NO predictive power for
  phytoplankton here.** The strong skill of Test 2 was entirely the nutrients (and largely
  circular). Remove nutrients and nothing remains — the marine-heatwave → phytoplankton
  signal is absent in this region/season.

---

## Stream A — physics (temp + upwelling `w`) as predictors
Features: `temp_z`, `w_z` (subsurface vertical velocity, level ~20/100 m) + lags.
Table: `src/step1b_multiyear_physfeat.py`.

- **A1 — physics → phytoplankton (`phy_z`):** log-loss **1.099** (= random baseline),
  accuracy **0.40** (below majority 0.48), importances split evenly (~0.18 temp / ~0.16 w
  = noise). **Negative:** neither SST nor upwelling predicts phytoplankton directly. The
  physics→phy link is fully mediated by nutrients.
- **A2 — physics → nutrients (`ntr_z`):** log-loss 0.66, accuracy 0.82 — but a
  **class-imbalance mirage** (ntr split 10/83/7; model just predicts "normal", catching
  only 4%/1% of nutrient collapses/blooms; macro-F1 0.33; log-loss 0.66 > base-rate 0.56).
  **Negative:** temp + subsurface `w` do NOT predict nutrient anomalies.

- **A3 — nutrients + physics → phytoplankton (all predictors):** log-loss 0.366,
  accuracy 0.84 (≈ Test 2). Importances: `ntr` ~86.5%, `temp` ~7.2%, `w` ~6.3%.
  **`w` is no better than temperature** — both marginal; adding upwelling barely improved
  skill (0.38 → 0.366). Physics adds ~negligible signal beyond nutrients.

**Stream A conclusion:** with the co-located physics available (`temp`, `w`), neither
phytoplankton nor nutrients are predictable; even alongside nutrients, physics contributes
~13% combined (each ~6–7%, marginal). Nutrient variability — which *does* drive
phytoplankton — is likely set by **lateral advection (`u`/`v`)**, which is full-domain-only
and needs the crop geometry resolved to co-locate. **Stream A is blocked on `u`/`v` (a lab-
metadata question).** Move to Stream B (PISCES) for the region/season generalization.

## Corrected re-run (2026-07-15) — full `(28,300,300)` region, 10–40°N  ✅
`src/rerun_corrected.py`. Table rebuilt on the correct shape: **51,050,180 rows**
(37% at ≥28°N). Std of z-scoring: temp 0.706, ntr 0.313, phy 0.0254. Year-blocked CV,
1.5M-row subsample per test. Random 3-class baseline ≈ 1.10.

| Test (temp-only unless noted) | log-loss | accuracy | read |
|---|---|---|---|
| Full region: **nutrients+temp → phy** | **0.479** | 0.780 | ntr ~86% / temp ~15%; nutrients still dominate |
| Full region: **temp-only → phy** | **1.028** | 0.488 | now *beats* random (was 1.12) — weak signal appears |
| **North ≥28°N: temp-only** | **0.988** | 0.556 | temperate band — clearest temp signal |
| **South <28°N: temp-only** | **1.045** | 0.456 | warm subset — still ≈ random |

**What the correction changed:** the old "temperature has *no* (worse-than-random)
predictive power" was an artifact of only ever sampling the warm south. On the correct
full region, **temp-only now beats the random baseline (1.028 < 1.10)**, and the effect is
**concentrated in the temperate north** (0.988 at ≥28°N) while the **warm south stays at
random** (1.045). This is the expected regime split: a marine-heatwave/temperature signal
is detectable where the ocean is temperature-limited (north) and absent where it's
nutrient-limited (south).

**Figure:** `figures/north_south_regime.png` (`src/fig_north_south.py`) — per-latitude
corr(temp/ntr, phy) + temp-only skill by band. Note: the temp↔phy *linear correlation* is
flat across the divide (north −0.174 vs south −0.164); the regime difference shows up in
*forecast skill* (Panel B), not correlation. South (1.045) also dips below random — the
gradient is modest, not a clean on/off.

**Oyashio / Kuroshio–Oyashio transition (NE corner, 33–40°N, 140–150°E)** —
`figures/oyashio_zoom.png` (`src/fig_oyashio.py`). The Oyashio *core* is north of the 40°N
crop edge; we capture the transition. Here temp↔phy coupling is **strongly NEGATIVE**
(mean −0.197, up to −0.6/−0.8 at the 40°N edge; only 11% of cells positive), while
nutrients stay strongly positive (0.83). Physical reading: in this cold, nutrient-rich
water a **warm anomaly suppresses phytoplankton** (more stratification → less mixing →
less nutrient supply → fewer blooms). This is the **marine-heatwave → collapse** direction
the project originally hypothesized, and it is **localized to the northern transition zone**
— it is what gives the north its (modest) temp-only forecast skill, via a negative
nutrient-mediated pathway, NOT warmth stimulating growth. (Corrects the earlier
"warm helps in the temp-limited north" guess.)

**Honest magnitude:** the signal is *weak* — north temp-only improves log-loss by ~0.11
over random, and importances stay ~even across the three temp lags (no single lag
dominates). Nutrients still overwhelmingly control phytoplankton everywhere (full-region
log-loss 0.479 with ntr ~86%). So the story is **"heatwaves are a real but secondary,
region-dependent driver," not "heatwaves drive collapse."** But the earlier flat rejection
is overturned: the temperate north was never in the old sample, and it does carry signal.

## Marine heatwaves (Hobday) — event detection on full-year physics temp  ✅
`src/hobday_build.py` (+ `hobday_test.py`). Uses the fact that the crop temp is
bit-identical to `OFES2_NP[0,300:600,200:500]` (verified, corr 1.0000) — so the full-year
2000–2019 physics temperature IS what the biology felt. Hobday climatology + 90th-pct
threshold (±5-day window, 31-day smooth) from Sep–Nov temp; ≥5-day events, ≤2-day gaps.

- **MHW climatology sane:** 8.4% of Oct cell-days are MHW; realistic interannual spread
  (quiet 1–3% in 2004/09/11–15; peaks **2017 32%, 2016 19%, 2008 18%**).
- **COMPOSITE (real, but modest & basin-wide):** phytoplankton = **−0.19σ on MHW days vs
  +0.02σ on calm days → ~0.2σ suppression**, essentially equal north (−0.202) vs south
  (−0.209). Confirms the **heatwave → suppression (collapse-direction)** effect as a
  discrete-event phenomenon — but it is **NOT north-concentrated** (revises the Oyashio-map
  read). n ≈ 4.7M MHW cell-days, so highly significant.
- **PREDICTIVE (deflationary):** MHW event features (flag+intensity) → phy class in the
  north = **log-loss 1.109 ≈ random 1.10** — no per-cell forecast skill. Worse than the raw
  `temp_z` baseline (0.988): thresholding warmth into a binary event **discards** the
  graded information the model was using. (Remaining baseline/south runs pending re-paste.)
- **Read:** heatwaves exert a **real but small (~0.2σ), graded, basin-wide drag** on
  phytoplankton; encoding them as discrete Hobday events is not predictively useful.
  Nutrients remain the dominant control.

## JCOPE vs OFES validation (Oct 2000–2004)  ✅  `src/jcope_vs_ofes.py`
Cross-check of the free-running OFES-BGC against the **data-assimilating JCOPE** reanalysis
(surface, overlapping Octobers, same anomaly+z method, warm-day = top-10% temp anomaly).

| metric (N / S) | JCOPE (obs-constrained) | OFES (free-running) |
|---|---|---|
| corr(phy, nutrients) | +0.32 / +0.35 | **+0.86 / +0.90** |
| corr(phy, temp) | −0.18 / **−0.53** | −0.15 / −0.18 |
| warm-day phy_z drop | **−0.50 / −0.83** | −0.21 / −0.43 |

- **Directional agreement (passes):** signs + structure match — nutrients drive phy (+),
  temp relates (−), warm days suppress phy, suppression **stronger in the south**, both.
- **Correction 1 — OFES nutrient link is partly circular:** OFES r≈0.86–0.90 (model computes
  phy from nutrients) vs JCOPE r≈0.33. Real coupling is moderate, leaving room for temp.
- **Correction 2 — heatwave suppression ~2× STRONGER in obs data:** warm-day drop −0.5σ(N)/
  −0.83σ(S) in JCOPE vs −0.21/−0.43 in OFES. Free-running model **underestimates** the
  marine-heatwave → phytoplankton impact; obs-constrained effect is moderate-to-large.
- **Caveats:** JCOPE clim only 5 yrs (noisier, inflates r somewhat); "warm day" = top-10%
  temp anomaly, not full Hobday; surface, October only. All-season Hobday-on-JCOPE = next.

## All-season Hobday on JCOPE (2010–2018)  ✅  `src/jcope_hobday.py`
The payoff: proper Hobday MHW detection (seasonal 90th-pct threshold, ≥5-day events) on the
observation-constrained JCOPE surface record, all seasons. Suppression = phy_z(MHW) −
phy_z(calm), σ units. Figures: `jcope_mhw_seasonal.png`, `jcope_mhw_maps.png`.

- **Overall suppression ≈ −0.40σ** (north) / −0.38σ (south) — moderate, robust, basin-wide.
  ~2× the OFES estimate → free-running model understates the heatwave impact (as validation showed).
- **Strong SEASONALITY (new headline):** SON **−0.67/−0.49**, MAM −0.49/−0.53 (strongest);
  JJA −0.30/−0.23; DJF −0.21/−0.26 (weakest). Spring & autumn (bloom/transition seasons,
  growth governed by mixing/nutrients) are most heatwave-sensitive; winter (light/mixing-
  limited) least. **Recontextualizes OFES:** its October-only window caught autumn (a high-
  suppression season) but underestimated magnitude and missed the spring peak / winter min.
- **MHW frequency** ~7–10% of cell-days, fairly flat by month (seasonal threshold), elevated
  along the Kuroshio; north peak Aug 11.6%.
- **Spatial:** suppression dominates the domain, deepest in the subtropical gyre / Kuroshio
  (20–35°N); a **tropical far-south band (~10–15°N) shows the opposite — phytoplankton BOOSTED**
  on MHW days. Real structure = subtropical-suppression vs deep-tropical-enhancement, not a
  clean 28°N split.
- **Caveats:** 2010–2018 only so far (extending to full 1993–2018); surface; phy std small
  (0.052 mmol N/m³) so σ-scaling amplifies. (Map colorbar label was reversed in the first
  render — red = suppression there; fixed in script to RdBu_r for the full run.)

## JCOPE — vertical structure, full record, and the predictive model
(Full detail + tables in `docs/COLLATED_RESULTS.md` §3.5–3.7. Summary here.)

- **Depth (§3.5, `jcope_depth.py`):** MHW suppression is **surface-trapped** (0–50 m,
  strongest at surface −0.41 N/−0.38 S), crosses zero ~55 m, and **reverses to subsurface
  enhancement at ~75–100 m** (DCM). Heatwaves **redistribute phytoplankton downward, not
  destroy it** → satellite/surface views overstate the impact.
- **Full record 1993–2018 (`jcope_hobday.py`):** seasonal structure robust — MAM/SON strongest
  (−0.53/−0.46 N), DJF weakest. North suppression stable & strong; **south weaker over the long
  record (−0.22) than 2010–2018 (−0.38)** → possible recent intensification (test by decade).
  Spatial: suppression basin-wide (deepest north), deep-tropical (~10–18°N) **enhancement**.
- **Improved classifier (§3.7, `jcope_classifier_v2.py`):** added lags + subsurface DIN +
  stratification + season → **log-loss 0.934→0.632, acc 0.54→0.725**. Importances: stratification
  0.19, season 0.22, lags ~0.33, subsurface 0.09; raw surface temp only 0.09 → skill is
  nutrient-supply physics, not direct thermal.
- **Overfitting check (`jcope_holdout.py`):** blocked 80/10/10 (leakage-safe, 10% of every
  year as 2° tiles) TEST 0.751 ≈ CV 0.717; blocked≈naive (0.751 vs 0.754) → **not overfitting,
  not leakage-inflated.** Honest headline ~0.72 acc / ~0.64 log-loss, robust across all splits.
  CV mode uses **year-blocked GroupKFold** (predict an unseen year), not 80/10/10 — chosen
  because random splits leak via spatial/temporal autocorrelation.
- **Probabilities (`jcope_probabilities.py`):** isotonic-calibrated out-of-fold
  P(collapse/normal/bloom) per cell-day; reliability diagram + Brier score; saved model.
- **Verification diagnostics (`jcope_diagnostics.py`, §3.9):** accuracy 0.722 ≈ majority
  baseline 0.693 (misleading). Real skill: collapse **ROC-AUC 0.843, PR-AUC 0.477** (base 0.151);
  argmax collapse recall only 0.179 (rare-class threshold artifact) but at **P>0.2 recall 0.80 /
  precision 0.34**, P>0.3 0.56/0.45. Calibration excellent (Brier reliability 0.0014); collapse↔
  bloom confusion ≈0. Report via ROC/PR/reliability + recall–precision trade-off, not accuracy.
- **FINAL RESULT — 7-day forecast (`jcope_forecast.py`, §3.8):** features at t → calibrated
  P(collapse) at t+7. **acc 0.722 / log-loss 0.634** (≈ nowcast — ocean state autocorrelated).
  **Brier Skill Score +0.160 vs climatology, +0.190 vs persistence** → beats both baselines =
  a genuine skillful forecast. **Heatwave conditioning:** collapse rate 7 d later 20.4% (warm)
  vs 14.5% (calm), ~40% relative rise; model captures it (21.5% vs 15.5%). Calibrated
  (reliability on diagonal, conservative at high end). Model: `models/jcope_forecast_7d.joblib`.

## Key findings so far
1. **The pipeline works** end-to-end on real multi-year data with honest year-blocked CV.
2. **Nutrients control phytoplankton here (~90%); temperature/heatwaves are marginal (~10%)** —
   robust across 20 years. Consistent with a warm, nutrient-limited subtropical ocean.
3. **Temperature/heatwaves carry a weak, region-dependent signal** — *revised* after the
   shape correction. In the temperate **north (≥28°N)** temp-only beats random (log-loss
   0.988 vs 1.10); in the warm **south** it does not (1.045). The old "worse-than-random,
   rejected everywhere" (Test 3) was an artifact of sampling only the warm south. Nutrients
   still dominate, but heatwaves are a genuine **secondary, temperature-limited-regime**
   driver — not nothing.

## Caveats (must state in the write-up)
- **Autumn-only** biology (no full calendar year exists in the data).
- **Cropped subtropical region**; a **grid-geometry artifact** makes maps/spatial detail
  unreliable (per-cell modeling and correlations are fine).
- **Circularity:** in NPZD, phytoplankton growth is computed largely *from* nutrients, so
  the high `ntr→phy` skill partly reflects the model re-learning its own equation. This
  is why Test 3 (temperature-only) matters — it's the non-circular question.
- **Single region/model** — not yet generalized; higher-latitude (temperature-limited)
  biology would be needed to properly test the heatwave hypothesis.

## Reproduce
`step1b_multiyear_anom.py` (→ `OFES Data/model_table.parquet`) → `step5_model.py`
(edit `FEATURES`, `MAX_ROWS`, `CLASS_MODE`). Single-month path: `step1b_series_cropped.py`
→ `step2b_daily_table.py` → `step5_model.py`. Figures: `visualize_test.py`.
