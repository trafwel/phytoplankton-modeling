# Modeling Phytoplankton Populations with Machine Learning

**Luke Prasarttongosoth**
University of Washington
August 2026

---

## Introduction

Phytoplankton are microscopic creatures that act as the bedrock of the ocean's delicate ecosystem, serving a variety of vital roles — for example, carbon fixation, oxygen production and the cycling of key nutrients as well as food web support. As their purpose is manifold and essential to oceanic life (and by proxy our own), they are worthy of being monitored and protected — a task that is growing ever more difficult with increases in phenomena such as climate-change driven marine heatwaves, oceanic acidification, and global warming, which threaten to damage the stability of the phytoplankton population.

Thus, a predictive tool to estimate phytoplankton populations would prove useful for disaster forecasting and phytoplankton research as a whole. However, such an idea is difficult to achieve in practice. Current oceanographic models are limited in their efficacy — mathematical models cannot encompass the ever-shifting complexity of the ocean and its physiochemical state, and manually capturing data from each and every point across the ocean is technologically and financially infeasible. To that end, this paper proposes to apply Machine Learning techniques to address this issue, leveraging computational power to address the mathematical and data driven deficiencies in current phytoplankton models.

Specifically, this paper asks whether the state of the phytoplankton population seven days ahead can be predicted from the present physical and chemical state of the ocean, and with what confidence. Rather than a single deterministic answer, the model developed here outputs a calibrated probability for each outcome, so predictions can be acted on according to the cost of a missed event versus a false alarm.

---

## Procedure

**Data.** The data used was the Japan Coastal Ocean Predictability Experiment (JCOPE-EC) ocean reanalysis, provided by JAMSTEC. It spans **1993–2018** at **daily** resolution and **1/12° (~9 km)** spatial resolution across the northwest Pacific (**10–40°N, 120–150°E**), covering the upper 200 m. Temperature, dissolved inorganic nitrogen (DIN), and phytoplankton concentration were drawn at the surface and at approximately 50 m depth.

A reanalysis was chosen deliberately over a free-running coupled model. A free-running model computes phytoplankton mechanistically from its own temperature and nutrient fields, so a machine learning model trained to predict phytoplankton from those fields is effectively re-learning the model's governing equations rather than discovering a relationship in the ocean. This circularity inflates apparent skill: in a matched comparison over October 2000–2004, a free-running model (OFES-BGC) produced a nutrient–phytoplankton correlation of r ≈ 0.86–0.90, whereas the observation-constrained JCOPE-EC produced a weaker but far more trustworthy r ≈ 0.33.

**Preprocessing.** Because phytoplankton concentration is dominated by a strong seasonal cycle, all variables were converted to anomalies — departures from a per-day-of-year climatology computed with an ±5-day window at each grid cell — then standardized. The model therefore predicts departures from what is normal for a given place and time of year rather than relearning the seasonal cycle. The phytoplankton anomaly was discretized into three classes: **collapse** (below −0.75σ), **normal** (−0.75σ to +0.75σ), and **bloom** (above +0.75σ). Under this definition collapse constitutes roughly 15% of cell-days and normal roughly 69%, making collapse the rare class of interest.

**Model selection.** In modeling the phytoplankton, the choice of model in particular was a point of contention. The importance of phytoplankton meant that it would be ill advised to use a model in which the results were not interpretable and understandable, as understanding the underlying factors as to why predicted changes are occurring is as important as the result itself. This, combined with the tabular format of the oceanographic datasets provided, left a **Random Forest Model** — a model in which large counts of decision trees have the majority taken — as the best possible choice for the situation. Random forests perform comparably to neural networks on tabular data of this kind while retaining direct interpretability through feature importance.

The model was trained to classify the state of the phytoplankton in a 9 km quadrant into bloom, collapse, or normal states, **seven days ahead** of the input observations.

**Features.** An initial model using only instantaneous surface temperature and nutrients performed weakly (log-loss 0.934, accuracy 0.543), indicating that surface conditions alone are insufficient — physically reasonable, since phytoplankton respond to the recent history of the water column and to conditions below the surface. The feature set was expanded to include surface temperature and DIN anomalies with lags at −7, −14, and −28 days; subsurface DIN near −50 m; stratification, represented as the surface-minus-subsurface temperature difference; and season, encoded as the sine and cosine of day-of-year. This improved log-loss to **0.632** and accuracy to **0.725**, confirming that temporal history and vertical structure carry most of the predictive signal. Feature importances bear this out: lags collectively contribute ≈0.33, season 0.22, stratification 0.19, subsurface DIN 0.09, and instantaneous surface temperature only 0.09.

**Training and validation.** The final model was a random forest of 120 trees with balanced class weighting, wrapped in isotonic calibration so its output probabilities are meaningful in absolute terms — that is, so cell-days assigned a 30% collapse probability collapse approximately 30% of the time. Validation used **year-blocked cross-validation**, withholding entire calendar years from training. This is essential for oceanographic data: a random split would place strongly autocorrelated neighboring cell-days on both sides of the split, leaking information and inflating apparent skill. Test accuracy (0.751) closely matched cross-validated accuracy (0.717), and a leakage-safe blocked hold-out produced nearly identical scores, indicating the model is neither overfit nor benefiting from leakage. Evaluation was performed on approximately 1.2 million cell-day forecasts.

---

## Results

**Skill relative to baselines.** Raw accuracy is misleading here. The model achieves 72.2% accuracy, but a trivial model always predicting "normal" achieves 69.3% simply because normal is the most common class. The meaningful assessment is probabilistic, measured by the Brier score relative to two reference forecasts: climatology (the historical collapse rate for that season) and persistence (assuming conditions in seven days match conditions today). The model achieves a **Brier Skill Score of +0.16 against climatology and +0.19 against persistence** — probabilistic error 16% and 19% lower than those baselines. Overall log-loss is 0.634. Positive skill against both confirms a genuine forecast rather than a restatement of the historical average or of present conditions.

**Discrimination.** Collapse **ROC-AUC is 0.843**: given a random collapse and a random non-collapse cell-day, the model assigns the higher collapse probability to the true collapse 84% of the time. **PR-AUC is 0.477** against a base rate of 0.151, approximately three times the no-skill floor. Precision–recall area is the more appropriate summary for a rare event, as it is directly penalized by false alarms in a way ROC-AUC is not.

**Operating characteristics.** Because the model outputs probabilities rather than hard classifications, the trade-off between catching collapses and issuing false alarms is tunable by the warning threshold:

| Threshold P(collapse) > | Collapses caught | Warnings correct | Cell-days flagged |
|---|---|---|---|
| 0.2 | 79.6% | 34.0% | 35% |
| 0.3 | 56.0% | 44.9% | 19% |
| 0.4 | 29.8% | 57.4% | 8% |
| 0.5 | 9.4% | 67.2% | 2% |

Recall and precision cross near P ≈ 0.33. The appropriate operating point depends on the relative cost of a missed collapse versus a false alarm; an early-warning application would favor the lower thresholds.

**Calibration.** Reliability curves computed separately for each season lie close to the 1:1 diagonal, indicating predicted probabilities correspond to observed collapse frequencies across all seasons rather than only in aggregate — the property that makes the threshold table actionable. Summer curves span a narrower probability range, because the model rarely issues high collapse probabilities during the strongly stratified summer months, when collapse is both rare and confidently ruled out.

<br>

**[ FIGURE 1 — `jcope_diagnostics.png` ]**

&nbsp;

&nbsp;

*Figure 1. Verification of the seven-day collapse forecast. (A) Row-normalized confusion matrix. (B) Collapse precision–recall curve, PR-AUC 0.477 against a base rate of 0.151. (C) Reliability by season. (D) Sharpness — predicted collapse probability separated by outcome.*

<br>

**Lead time.** Forecast skill was evaluated at leads of 1, 3, 5, 7, and 10 days using identical features and validation:

| Lead (days) | Accuracy | BSS vs. climatology | BSS vs. persistence |
|---|---|---|---|
| 1 | 0.751 | +0.269 | −1.443 |
| 3 | 0.730 | +0.218 | −0.166 |
| 5 | 0.717 | +0.180 | +0.068 |
| 7 | 0.709 | +0.157 | +0.169 |
| 10 | 0.702 | +0.133 | +0.249 |

Two opposing trends emerge. Skill relative to climatology decays with lead time but remains positive through ten days. Skill relative to persistence increases with lead time and crosses zero near day five: at short leads the ocean changes so little that assuming no change is nearly unbeatable, whereas at longer leads persistence degrades rapidly and the model's driver-based forecast becomes substantially better. The seven-day horizon used throughout sits in the regime where the model outperforms both references simultaneously.

<br>

**[ FIGURE 2 — `jcope_leadtime.png` ]**

&nbsp;

&nbsp;

*Figure 2. Forecast skill against lead time. (A) Skill against climatology decays but stays positive through ten days, while skill against persistence rises and crosses zero near day five. (B) Warning performance at P > 0.3 across lead times.*

<br>

**Marine heatwave response.** Marine heatwaves were identified following Hobday et al. (2016) — sea surface temperature exceeding the local seasonally varying 90th percentile for at least five consecutive days. Conditioning collapse rate on heatwave status shows a clear effect: observed collapse probability seven days after a non-heatwave day is **14.5%**, rising to **20.4%** after a heatwave day, a relative increase of roughly 40%. The calibrated model reproduces this structure, predicting approximately 15.5% and 21.5% respectively. Suppression varies seasonally, strongest in spring and autumn (≈ −0.53σ) and weakest in winter (−0.21σ); forecast skill follows a complementary pattern, highest in summer (log-loss 0.268, accuracy 0.928) and lowest in winter (0.897, 0.557).

**Vertical structure.** Analysis across ten depth levels revealed that heatwave-associated suppression is surface-trapped rather than column-wide. Surface anomalies during heatwaves average ≈ −0.41σ, but this weakens with depth and reverses to enhancement of +0.1 to +0.24σ at 75–100 m, near the deep chlorophyll maximum. This is consistent with heatwaves acting primarily by intensifying near-surface stratification, redistributing phytoplankton downward rather than eliminating it. The implication is that surface-only observations — including satellite ocean color — may systematically overstate the total biological impact of marine heatwaves.

<br>

**[ FIGURE 3 — `jcope_mhw_depth.png` ]**

&nbsp;

&nbsp;

*Figure 3. Phytoplankton anomaly during marine heatwaves as a function of depth. Suppression is confined to the surface layer and reverses to enhancement near 75–100 m.*

<br>

---

## Conclusion

The initial goal of creating a model to correctly forecast phytoplankton population states was met to an extent. As seen above, the model itself does perform adequately — producing calibrated seven-day probabilities that beat both climatological and persistence baselines, discriminate the rare collapse class well above chance, and reproduce the observed marine-heatwave signal. However, there are certain limitations that prevent it from being a ubiquitously usable tool.

Foremost is the limitation of the datasets. While JCOPE does provide impressive coverage, it is in itself an ocean reanalysis, not direct observation. Its physical fields are constrained by assimilated measurements, but its biological fields are model-derived and only weakly observationally constrained. Thus, there is the possibility of discrepancies between the phytoplankton variability represented in the reanalysis and that of the real ocean, and the forecast presented here should be understood as validated against a model rather than against nature. Independent verification against satellite ocean-color chlorophyll is the necessary next step — with the caveat, established by the depth analysis above, that satellites see only the surface layer where suppression is strongest.

Several further limitations bear noting. The definition of collapse is relative rather than absolute, denoting an unusually low concentration for a given place and season rather than a specific biomass floor. The model treats total phytoplankton rather than distinguishing functional groups such as diatoms, whose ecological roles differ substantially. The domain is a single region over a historical period, so the results should not be read as a projection of future climate states. And the trade-off between collapses caught and warnings proven correct limits how aggressively the model can be operated.

Furthermore, the topic itself can be refined further. The most direct extensions are the incorporation of additional physical drivers — light availability, salinity, and current-driven advection — which influence phytoplankton dynamics but were not included here; extension of the forecast horizon beyond ten days toward monthly and seasonal ranges, where short-term prediction gives way to genuinely anticipatory management; and expansion of the spatial domain beyond the northwest Pacific to test whether the relationships learned here generalize. Establishing that generality, alongside observational validation, would mark the transition from the demonstration presented here to a tool suitable for operational use.

---

## References

Hobday, A. J., et al. (2016). A hierarchical approach to defining marine heatwaves. *Progress in Oceanography*, 141, 227–238.

Miyazawa, Y., et al. JCOPE ocean reanalysis. Japan Agency for Marine-Earth Science and Technology (JAMSTEC).
