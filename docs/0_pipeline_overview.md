# Project Overview — How Everything Fits Together

The one-line goal: **forecast diatom/nano phytoplankton collapses and blooms in the
Arabian Sea from marine heatwaves, as a probability + severity, using interpretable
ML.** This doc is the map: what each stage consumes, produces, and feeds into.

---

## The architecture in one picture

Two data sources → two analytical tracks that converge into one model table → one
interpretable model that outputs collapse/bloom probability + severity.

```
  OFES temperature (Arabian Sea)            Satellite PFT chlorophyll (GlobColour)
  0.1deg, 3-day, 1980-2014                  4 km, monthly, 1997-present
        │                                          │
        │ (Phase 1.1 acquire)                      │ (Phase 1.2 acquire)
        ▼                                          ▼
  ┌───────────────┐                          ┌───────────────────┐
  │ 3-day SST     │                          │ DIATO, NANO, CHL  │
  └──────┬────────┘                          └─────────┬─────────┘
         │                                             │ 3.2 regrid onto OFES grid
         │                                             ▼
         │                                   ┌───────────────────┐
         │                                   │ PFT on OFES grid  │
         │                                   └─────────┬─────────┘
         ├── keep 3-day ──► Phase 4 MHW detect          │ 3.4 log + anomalies
         │        │                                     ▼
         │        ▼                             ┌──────────────────────┐
         │  monthly MHW features                │ DIATO/NANO anomalies │  ← the TARGET
         │        │                             │ + DCHL/NCHL ratio    │
         │ 3.3 monthly-mean                      └──────────┬───────────┘
         ▼        │                                         │
  monthly SST + anomaly  ───────────────┐                   │
                                        ▼                   ▼
                              ┌──────────────────────────────────┐
                              │  3.6 analysis table (parquet)     │  one row per
                              │  features  |  target              │  (cell, month)
                              └───────────────┬───────────────────┘
                                              │  Phase 5/7
                                              ▼
                              ┌──────────────────────────────────┐
                              │ Random Forest (multi-class)       │
                              │ predict_proba → P(collapse/bloom) │
                              │ + severity, calibrated            │
                              └───────────────┬───────────────────┘
                                              │  Phase 8 validate · Phase 9 visualize
                                              ▼
                                     forecast maps + write-up
```

---

## The locked decisions (the spine of the project)

- **Region:** Arabian Sea, 5–25°N, 50–75°E. **Window:** 1998–2014 (OFES × PFT overlap).
- **MHW driver:** OFES surface temperature (matches the satellite's surface signal —
  the photic-zone question, resolved by scoping to the surface).
- **Target:** satellite `DIATO` (=DCHL) and `NANO` (=NCHL) — the diatom/nano split OFES
  lacks. Prediction = **probability of collapse/bloom + severity** (not a hard binary),
  nowcast.
- **Cadence:** monthly (PFT-bound) → 204 months per cell.
- **Model:** Random Forest, interpretable, calibrated probabilities; blocked CV.

---

## Stage-by-stage: consumes → produces → feeds

| Stage | Consumes | Produces | Feeds |
|---|---|---|---|
| **1.1 OFES acquire** | APDRC OPeNDAP | 3-day surface SST NetCDF | 3.1, Phase 4 |
| **1.2 PFT acquire** | CMEMS toolbox | DIATO/NANO/CHL monthly NetCDF | 3.2 |
| **3.1 mask** | raw OFES/PFT | land→NaN, trimmed | 3.2, 3.3 |
| **3.2 regrid** | PFT + OFES grid | `pft_on_ofes_grid.nc` | 3.4, 3.6 |
| **3.3 temporal align** | 3-day OFES | `sst_monthly_...nc` (204 mo) | 3.4, 3.6 |
| **Phase 4 MHW** | 3-day OFES (kept!) | monthly MHW features | 3.6 |
| **3.4 anomalies** | monthly SST + PFT | anomalies + DCHL/NCHL ratio | 3.6 |
| **3.6 build table** | all monthly fields | `model_table.parquet` | Phase 5/7 |
| **Phase 5/7 model** | model table | trained RF + predictions | 8, 9 |
| **Phase 8 validate** | predictions, OC-CCI | skill metrics | write-up |
| **Phase 9 visualize** | predictions | maps/figures | write-up |

**The key non-obvious link:** OFES temperature forks. The **3-day** version feeds
Phase 4 heatwave detection (which needs sub-monthly resolution); a **monthly-mean**
version becomes an anomaly feature. Phase 4's outputs come back as monthly features.
Both rejoin at 3.6.

**The two-climatology distinction:** Phase 4 builds a *daily 90th-percentile* threshold
(to detect heatwaves); Step 3.4 builds a *monthly* climatology (to make anomalies).
Different tools, different jobs — don't conflate them.

---

## Where the files live

**Walkthroughs (`docs/`)**: `1.1_ofes...`, `1.2_pft...`, `2_environment...`,
`3_preprocessing_full...` (+ `3.2_regridding...`, `3.3_temporal...`),
`4_marine_heatwave_detection...`, and this overview. Master checklist:
`implementation_checklist.md`.

**Code (`src/`)**: `regrid_pft_to_ofes.py` (3.2), `step3_3_temporal_align.py` (3.1+3.3),
`step3_4_anomalies.py` (3.4), `step3_6_build_table.py` (3.6). Your own `get_ofes.py` (1.1).

**Data (`data/`)**: `raw/` (immutable downloads) → `interim/` (regridded, aligned,
anomalies — disposable) → `processed/model_table.parquet` (model input).

---

## Status snapshot (as of this overview)

- ✅ **Spec'd & decided:** whole architecture, region/window, all Phase 3 + Phase 4.
- ✅ **Coded:** Phase 3 pipeline (3.2, 3.3, 3.4, 3.6); environment set up.
- ⏸️ **Blocked on disk space:** the actual OFES + PFT downloads — so nothing has been
  *run* end-to-end yet. All scripts are staged and waiting for data.
- ⬜ **Not yet built:** Phase 4 code, Phase 5/7 (the model — your core contribution),
  Phase 8 validation, Phase 9 visualization, Phase 10 write-up (due July 22).

## What's next
Phase 5/7 — the phytoplankton-anomaly model: define collapse/bloom + severity labels
from the anomalies, train the calibrated Random Forest with blocked CV, and interpret
it (SHAP). That's the deliverable the whole pipeline exists to feed.
```
