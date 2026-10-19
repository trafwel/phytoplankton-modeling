# Forecasting Phytoplankton Collapse in the Northwest Pacific

A probabilistic 7-day forecast of phytoplankton population state (**collapse / normal / bloom**) across the northwest Pacific, trained on the observation-constrained **JCOPE-EC** ocean reanalysis from JAMSTEC.

Rather than a single hard prediction, the model outputs a *calibrated probability* for each outcome, so a warning threshold can be set according to whether a missed collapse or a false alarm is more costly.

**Scope:** 1993–2018 · daily · 1/12° (~9 km) · 10–40°N, 120–150°E · upper 200 m
**Evaluation:** ~1.2 million cell-day forecasts

---

## Results

### Forecast skill

| Measure | Result | Interpretation |
|---|---|---|
| Brier Skill Score vs. climatology | **+0.16** | Real skill; 0.1–0.3 is the useful operational band |
| Brier Skill Score vs. persistence | **+0.19** | Beats "assume nothing changes" |
| Collapse ROC-AUC | **0.843** | Ranks a true collapse above a non-collapse 84% of the time |
| Collapse PR-AUC | **0.477** (base rate 0.151) | ~3× better than chance on a rare event |
| Calibration | On the diagonal, every season | A stated 30% genuinely means ~30% |
| Overfitting check | Test 0.751 ≈ CV 0.717 | Year-blocking held; no leakage |

Accuracy is 72.2%, but a model that always predicts "normal" scores 69.3% — so **accuracy is close to a non-result here**. The value is entirely in the probabilities.

### Warning threshold trade-off

| P(collapse) > | Collapses caught | Warnings correct | Cell-days flagged |
|---|---|---|---|
| 0.2 | 79.6% | 34.0% | 35% |
| 0.3 | 56.0% | 44.9% | 19% |
| 0.4 | 29.8% | 57.4% | 8% |
| 0.5 | 9.4% | 67.2% | 2% |

Recall and precision cross near P ≈ 0.33.

### Skill by lead time

| Lead (days) | Accuracy | vs. climatology | vs. persistence |
|---|---|---|---|
| 1 | 0.751 | +0.269 | −1.443 |
| 3 | 0.730 | +0.218 | −0.166 |
| 5 | 0.717 | +0.180 | +0.068 |
| 7 | 0.709 | +0.157 | +0.169 |
| 10 | 0.702 | +0.133 | +0.249 |

Skill over climatology decays but stays positive through 10 days. Skill over persistence *rises* with lead and crosses zero near **day 5** — below that, "nothing changes" is unbeatable.

---

## Findings

**1. Marine heatwaves measurably raise collapse risk.** Collapse probability seven days out goes from **14.5% to 20.4%** following a heatwave day (Hobday et al. 2016 definition) — a ~40% relative increase. The calibrated model reproduces the effect rather than merely correlating with it.

**2. Suppression is redistribution, not destruction.** Surface anomalies during heatwaves average −0.41σ, but this *reverses to +0.1 to +0.24σ at 75–100 m*, at the deep chlorophyll maximum. Heatwaves stratify the surface and push phytoplankton downward rather than eliminating it — which implies **satellite ocean-color studies may overstate heatwave impact**, since they observe only the surface layer.

**3. Predictability lives in history and depth, not the surface snapshot.** Feature importance: temporal lags ≈ 0.33, season 0.22, stratification 0.19, subsurface nutrients 0.09 — versus only 0.09 for instantaneous surface temperature. A naive "warm today → collapse" framing is wrong.

**4. Free-running models produce circular skill.** OFES-BGC yields a nutrient–phytoplankton correlation of r ≈ 0.86–0.90 because it *computes* phytoplankton from those fields; an ML model trained on it re-learns the model's own equations. Observation-constrained JCOPE-EC gives a weaker r ≈ 0.33 that is far more trustworthy. The pivot cost apparent performance and bought validity.

**5. Seasonal structure.** Suppression is strongest in spring and autumn (≈ −0.53σ), weakest in winter (−0.21σ). Forecast skill runs opposite: best in summer (log-loss 0.268, accuracy 0.928), worst in winter (0.897, 0.557).

---

## Limitations

- **Validated against a reanalysis, not observations.** JCOPE's physics are observation-constrained, but its *biological* fields are model-derived and only weakly tied to measurements. The defensible claim is "this forecasts JCOPE's phytoplankton well," not "this forecasts the real ocean." Satellite chlorophyll validation is the necessary next step — complicated by finding #2, since satellites cannot see the subsurface compensation.
- **Precision is low at useful sensitivity.** Catching 80% of collapses means roughly two-thirds of warnings are false alarms.
- **Winter performance is poor** (0.557 accuracy); the headline numbers are carried by summer.
- **Collapse is defined relatively** (below −0.75σ of the local seasonal norm), not as an absolute biomass floor.
- **Total phytoplankton only** — no functional-group separation (e.g. diatoms).
- **One region, historical period.** Not a future-climate projection.

Overall: a working demonstration with modest-but-real skill, not a deployable early-warning system.

---

## Repository layout

```
src/        current JCOPE pipeline — extraction, Hobday MHW detection,
            classifiers, forecast, lead-time sweep, diagnostics
figures/    final figures referenced in the paper
paper/      the write-up (markdown, docx, pdf) and result tables
docs/       methodology walkthroughs, collated results, project status
archive/    superseded OFES-era scripts, docs, and figures, kept for
            provenance — see Finding #4 for why that path was abandoned
```

### Key scripts

| Script | Purpose |
|---|---|
| `src/jcope_extract_full.py` | Extract JCOPE fields to per-day arrays |
| `src/jcope_hobday.py` | Marine heatwave detection (Hobday et al. 2016) |
| `src/jcope_forecast.py` | Final 7-day calibrated forecast; builds the feature cache |
| `src/jcope_leadtime.py` | Skill vs. lead time (1–10 days) |
| `src/jcope_diagnostics.py` | Verification: confusion, PR curve, reliability, sharpness |
| `src/jcope_diag_figs.py` | Reliability-by-season and threshold-sweep figures |
| `src/jcope_holdout.py` | Overfitting / leakage validation |
| `src/jcope_depth.py` | Vertical structure of the heatwave response |

---

## Method

- **Model:** random forest (120 trees, balanced class weighting) with **isotonic calibration**. Chosen over a neural network for interpretability at comparable tabular performance.
- **Target:** phytoplankton class 7 days ahead. Collapse < −0.75σ, bloom > +0.75σ, normal between — all relative to a per-day-of-year climatology (±5-day window) at each grid cell, then standardized.
- **Features:** surface and subsurface (−50 m) temperature and dissolved inorganic nitrogen, stratification (surface minus subsurface temperature), lags at −7/−14/−28 days, and season as sin/cos of day-of-year.
- **Validation:** **year-blocked cross-validation** (GroupKFold over calendar years). A random split would place autocorrelated neighboring cell-days on both sides and inflate skill. A blocked hold-out withholding whole 2° tiles was run as a further check.

---

## Data

The JCOPE-EC reanalysis is not redistributed here. Large data and model artifacts (~17 GB: `models/`, `data/`, `JCOPE Data/`, `OFES Data/`) are excluded via `.gitignore`.

Scripts locate the extracted data by scanning for a `JCOPE_full/` directory containing `t/`, `phy/`, and `din/` subfolders. Most analysis scripts cache their feature tables as `.npz`, so reruns skip the expensive read.

---

## Reference

Hobday, A. J., et al. (2016). A hierarchical approach to defining marine heatwaves. *Progress in Oceanography*, 141, 227–238.

JCOPE ocean reanalysis — Japan Agency for Marine-Earth Science and Technology (JAMSTEC).

---

Luke Prasarttongosoth · University of Washington
