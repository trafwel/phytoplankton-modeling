# Thesis Document (completed)

## Thesis
Utilizing computer science and machine learning to forecast a **probabilistic
metric — accompanied by a projected severity — of diatom-phytoplankton population
collapses (and blooms) brought about by marine heatwaves.**

## Motivation
Phytoplankton play a crucial role in the ocean's carbon cycle, and are responsible
for a great deal of the natural processes we rely on to maintain a livable
environment, being regarded as one of the world's primary oxygen sources and an
integral part of the carbon cycle. Thus, it becomes especially important to monitor
the health of the phytoplankton population, which can suffer under marine heatwaves —
events that have themselves grown more frequent and more intense with climate change.

Specifically, we monitor diatom-phytoplankton as our target, in relation to
nano-phytoplankton, and the reasoning here is twofold. First, diatoms are the ocean's
carbon workhorses: being large and encased in dense silica shells, they sink rapidly
once they die, carrying carbon down into the deep ocean far more efficiently than the
smaller cells do. They account for a disproportionate share of the ocean's carbon
export, and so their fate is, in effect, the fate of the biological pump itself.
Second, and precisely because of that size, diatoms are the most demanding — and thus
the most fragile — of the phytoplankton: they require cool, nutrient-rich, well-mixed
water to flourish, which is exactly the condition a marine heatwave dismantles. When
the surface warms and stratifies, nutrients stop reaching the surface, and diatoms
are the first to fail, giving way to smaller, heat-tolerant taxa such as
dinoflagellates.

This is also why a diatom-specific view matters, and why general phytoplankton is not
enough. Total chlorophyll can hold steady — or even rise — while the community quietly
shifts from diatoms to dinoflagellates; the biomass still looks healthy, but the
carbon export has already collapsed underneath it. A bulk measurement would miss this
entirely. By tracking diatoms directly (and, with the dinoflagellate signal, the
diatom-versus-dinoflagellate balance), we capture the compositional change that
actually governs the carbon cycle, and we capture it *early* — the diatom collapse is
a leading indicator of a degradation that a total measure only registers, if ever,
much later. In short, diatoms are at once the most consequential compartment for the
world's carbon and the most sensitive early-warning signal for heatwave stress, which
is what makes them the natural target for a forecasting model.

## Methodology
Target switched to the **OFES** dataset, since SINTEX-F is predictive and isn't
something we can train our own model on. Area of study is the **Pacific** (accessible
by OFES; coverage is present but should be double-checked for the exact sub-region).
Notably, the lab's coupled **OFES-BGC** run is Pacific-domain, so it can serve as a
model-based **validation standard** (which satisfies its Pacific-only limitation).

From OFES we can derive the key physical features: **temperature**, and — from the
coupled BGC run — **phytoplankton** and **nutrient (ntr)**. However, OFES-BGC's total
phytoplankton does **not** cleanly split into the DCHL/NCHL distinction the thesis is
built on (it is a single-compartment NPZD model). We therefore use **Copernicus
satellite PFT** data as the biological target, which provides the functional-type
detail: **NANO** and **CHL**, plus **DIATO** (the diatom chlorophyll) and **DINO**
(dinoflagellates, which let us be more certain a decline is a true diatom collapse
rather than a general dip). Because satellite data is our primary biological source,
there are surveillance limitations (surface-only signal; the functional-type split is
an algorithmic partition carrying uncertainty) that must be addressed as stated
limitations in the research.

By training on these factors we build a model that forecasts the presence and impact
of marine-heatwave conditions on a diatom-phytoplankton population's health — which
carries much larger implications for the carbon cycle and ocean health as a whole.

> **Flag to confirm:** the draft said "DIATO … we only use this variable as a
> verification standard." That conflicts with the design, where **DIATO is the
> prediction target**. Written above as target; revert if the change was intentional.

## Model
For model selection the following are most apt, since we (a) want to maximize
interpretability, (b) don't have strictly independent data, and (c) don't want a
linear decision boundary and need a probabilistic output.

- **Random Forest / Decision Tree framework** — matches the ensemble style and gives
  classification or regression output as needed. **Temporarily settled on Random
  Forest.**
- A **multi-class Random Forest** over `{collapse, stable, bloom}` (extendable to five
  ordered severity classes: strong/mild collapse, normal, mild/strong bloom) fits the
  "probability + severity" goal, since `predict_proba` yields both the class
  probability and the severity distribution. This warrants further investigation.

We create **two models**: one to determine the presence of marine heatwaves in
general, and one to detect anomalies in the diatom/nano levels; we then correlate them
and map the anomaly response onto the heatwave signal. The first (MHW detection) is a
**standard deterministic algorithm** (Hobday et al. 2016) — no training needed, pulled
in as a library rather than a learned model. The second (the diatom-anomaly model) is
built from scratch and is the core contribution.

> **Flag to reconcile:** the draft still notes "LOOCV and 80/10/10." With gridded
> spatiotemporal data, LOOCV/random splits **leak** (neighbouring cells and months are
> correlated), inflating skill. The design uses **spatial-block cross-validation**
> (hold out whole regions) for an honest estimate. Keep this unless the advisor
> prefers otherwise. Also planned: probability **calibration** (isotonic), **SHAP**
> for interpretability, and a **persistence/climatology baseline** to beat.

## Scope
Limit to a single sea region for now (a Pacific sub-region — pick a specific box, e.g.
the NE Pacific to capture the well-documented 2013–16 "Blob" heatwave, or the Kuroshio
region where the lab works). Note global expansion as a possible later step.

## Validation of results
- **ESA Ocean Colour Initiative (OC-CCI)** chlorophyll-a to reaffirm diatom/nano
  presence (there may be visible differences to reconcile vs the GlobColour PFT).
- **OFES-BGC** total phytoplankton as an independent model-based cross-check.
- Known Pacific heatwave events (esp. the **2013–16 NE Pacific Blob**) as benchmarks
  the OFES-detected MHWs must reproduce.
- NASA hyperspectral imaging for phytoplankton/MHW indicators (limited by SST; requires
  researching where the photic zone lies).
- **Data assimilation** — doing DA ourselves is out of scope for v1; but we can *use*
  assimilated products (e.g. NOAA OISST, an optimal-interpolation SST product) as an
  observation-constrained comparison for the heatwave side. Self-run DA → future work.

## Possible additions after finished
- **Data visualization** of the consequences of poor DCHL/NCHL correlation; to make
  this a true forecast product, phytoplankton concentration should be the primary
  visualized variable (later task).
- **Fishery implications** — map fish-population / yield data onto the model. Heavily
  complicates the pipeline and risks interpretability/accuracy loss; leave as a later
  extension.
- **Global-scale increase** — harder, and coverage/relevance vary by region; would be
  nice but is a stretch.
- **Phytoplankton's role in climate change** — hard to quantify, likely out of scope.
- **Adding more features** — e.g. mixed-layer depth or silicate from OFES; kept lean
  for interpretability for now.

## Current issues
- Mapping resolution → **resolved** (regrid OFES ↔ PFT onto one grid).
- Temperature discrepancies → **shelved**.
- Photic-zone coverage → **temporarily noted as not fully possible** with surface
  ocean colour; scoped to the surface layer. (Note: OFES-BGC is 3-D, so subsurface is
  available on the model side if pursued later.)

## Timeline / status
- [x] Data-acquisition pipeline confirmed.
- [x] Data-acquisition code written.
- [x] Conflict issues resolved.
- [~] Data loader — code complete, **not yet run** (pending the data on disk).
- [x] Copernicus account made; data-pull verified (dry-run).
- [x] OFES → PFT mapping code (spatiotemporal); needs a real-run double-check.
- [x] **DINO factored into processing** — download, regrid, and anomalies now build
  `logdino_anom` + the `diatom_dominance` index `DIATO/(DIATO+DINO)`.
- [~] Data processing — coded end to end; **not yet run**.
- [ ] Run the full pipeline (test on one cell/month first), then model + correlation.
- [ ] Validation, visualization, write-up (summary due **July 22**).

## Todo (people / questions)
- Ask **Yuhi-san** about the biological pump mechanism.
- Ask about general chlorophyll surveying methods.
- Ask about CO₂ surveyance methods.
- Ask **Sakada-san** about subsurface chlorophyll as an indicator of phytoplankton.
- Email **Professor Jacques** (or a TA) about the ML classification / CV strategy.
- Confirm with the advisor: OFES-BGC (general, self-consistent) vs OFES-physics +
  satellite PFT (diatom-resolving, observational) as the biology source.

## Tools
- NASA Panoply for data visualization / NetCDF QC.

## Notes
- Fortunately, marine heatwaves tend to coincide with **lower cloud cover**, so ocean-
  colour gaps are less of a concern during the events we care about most.

## Relevant models and papers
- [Phytoplankton Adaptation to Climate Change](https://www.sciencedirect.com/science/article/pii/S0304380023001680)
- [Marine Heatwave Definition (Hobday et al. 2016)](https://www.sciencedirect.com/science/article/pii/S0079661116000057)
- [Marine Heatwave Impact on Phytoplankton in the South China Sea](https://www.sciencedirect.com/science/article/pii/S0048969724052495)
- [Digital Twin General Summary](https://www.edito.eu/)
- [Coastal Risk Data Visualization (reference viz tool)](https://coastal-risk.lab.dive.edito.eu/)
- [Kobe University disaster hazard mapping](https://www.kobe-u.ac.jp/en/news/article/20250303-66194/)
- [Carbon Cycle Basics (NASA)](https://cce.nasa.gov/cce/ocean_exports_call.htm)
- [Climate Model Basics (MIT)](https://climate.mit.edu/explainers/climate-models)
- [Satellite indicators of Phytoplankton Presence (New Zealand)](https://environment.govt.nz/assets/Publications/Files/satellite-indicators-of-phytoplankton-and-ocean-surface-temperature.pdf)
- [Determining Phytoplankton Population based on water parameters](https://www.sciencedirect.com/science/article/pii/S1470160X2400178X)
- [Phytoplankton Bloom inhibition in the Mediterranean Sea due to heatwaves](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024GL109141)
- [PFT phytoplankton satellite data (Copernicus OSR7)](https://sp.copernicus.org/articles/1-osr7/5/2023/)
- [Rousseaux et al. (2023), *Impact of Pacific Ocean heatwaves on phytoplankton community composition*, Communications Biology 6:263](https://www.nature.com/articles/s42003-023-04645-0) — **key reference.** Pacific MHWs shift PFTs diatom→dinoflagellate; higher-trophic mortality is triggered by the phytoplankton composition change (diatoms as the base-of-web early indicator). The 2014 "Blob" *raised* total chlorophyll while diatoms were replaced by dinoflagellates (total ≠ composition); the 2016 El Niño caused ~40% chl decline with near-total diatom collapse. Uses satellite ocean-colour assimilated into a BGC model — mirrors our OFES-BGC + satellite PFT design.
