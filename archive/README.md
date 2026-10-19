# Archive

Superseded work from the project's first phase, when the model was trained on
the **OFES-BGC** free-running coupled model rather than the JCOPE-EC reanalysis.

This material is kept for provenance, not for reuse. Two problems ended this line
of work:

1. **Circularity.** OFES-BGC computes phytoplankton mechanistically from its own
   temperature and nutrient fields, so a model trained to predict phytoplankton
   from those fields was largely re-learning the model's governing equations.
   The apparent nutrient–phytoplankton correlation (r ≈ 0.86–0.90) was an artifact;
   observation-constrained JCOPE-EC gives a weaker but trustworthy r ≈ 0.33.

2. **Coverage.** The OFES-BGC biology available here was effectively October-only
   (plus September 2018), which made all-season marine heatwave analysis impossible
   and masked the seasonal structure reported in the main results.

Contents:

- `scripts/` — OFES acquisition, regridding, the `step*` preprocessing chain,
  and the first-generation model and correlation scripts
- `docs/` — acquisition and regridding walkthroughs, the early implementation
  checklist, and superseded thesis drafts
- `figures/` — exploratory figures from the OFES phase (coupling heatmap,
  north/south regime split, Oyashio zoom, coverage and QC plots)

Scripts here reference data layouts and paths that no longer exist in the
current pipeline. Nothing in `src/` depends on them.
