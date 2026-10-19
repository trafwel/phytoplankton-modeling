# Project Status & File Index

Snapshot of where the thesis project stands and what every file is for.
Companion to `docs/0_pipeline_overview.md` (the data-flow map) and
`implementation_checklist.md` (the detailed task list).

---

## The project in three sentences

Forecast diatom/nano phytoplankton **collapses and blooms in the Arabian Sea** from
**marine heatwaves**, as a **probability + severity**, using an interpretable Random
Forest. Temperature comes from **OFES** (heatwave driver); the diatom/nano target
comes from **satellite GlobColour PFT**. Everything is monthly, 1998–2014, 204 months.

## Locked decisions

- Region **Arabian Sea 5–25°N, 50–75°E**; window **1998–2014**.
- Driver = OFES **surface** temperature; target = satellite **DIATO**/**NANO** chl.
- Output = **probability of collapse/bloom + severity** (5 ordered classes), nowcast.
- Model = **Random Forest**, calibrated, **year-blocked CV**, SHAP for interpretability.

---

## File index

### Docs (`docs/`) — the walkthroughs (the "why/how")
| File | Covers |
|---|---|
| `0_pipeline_overview.md` | Data-flow map of the whole project |
| `PROJECT_STATUS.md` | This file — status + index |
| `1.1_ofes_acquisition_walkthrough.md` | Download OFES temp (APDRC OPeNDAP) |
| `1.2_pft_acquisition_walkthrough.md` | Download PFT (CMEMS toolbox) |
| `2_environment_walkthrough.md` | Conda env + packages |
| `3_preprocessing_full_walkthrough.md` | All of Phase 3, step by step |
| `3.2_regridding_walkthrough.md` | Regridding detail |
| `3.3_temporal_alignment_walkthrough.md` | Monthly alignment detail |
| `4_marine_heatwave_detection_walkthrough.md` | Hobday MHW method |

### Code (`src/`) — the pipeline (the "what runs"), in run order
| File | Phase | Does |
|---|---|---|
| `../get_ofes.py` | 1.1 | Your OFES downloader |
| `regrid_pft_to_ofes.py` | 3.2 | PFT 4 km → OFES 0.1° (conservative) |
| `step3_3_temporal_align.py` | 3.1+3.3 | Mask + OFES 3-day → monthly (204 mo) |
| `step3_4_anomalies.py` | 3.4 | Log-chl, monthly climatology + anomalies, DCHL/NCHL ratio |
| `step4_mhw_features.py` | 4 | Hobday MHW detect → monthly `mhw_*` features |
| `step3_6_build_table.py` | 3.6 | Merge all → `model_table.parquet` |
| `step5_model.py` | 5+7 | Labels → RF → blocked CV → metrics → SHAP → calibrated model |
| `step6_correlate.py` | 6 | Lag-correlation / composite / co-occurrence (MHW→diatom) |

**Run order:** get_ofes → regrid → step3_3 → step3_4 → step4 → step3_6 → step5 → step6.
(step4 needs the 3-day OFES; step3_6 pulls in step4's output automatically.)

### Data (`data/`)
| Path | State |
|---|---|
| `data/raw/pft/pft_arabiansea_1998_2014.nc.xrjsq1lw` | ⚠️ **partial/interrupted** PFT download — not a usable file yet |
| `data/interim/`, `data/processed/` | not created yet (produced when scripts run) |

### Reference / misc
- `old reference docs/` — prior lab notebooks (2018 typhoon study) + the original
  `temp.01.01.2000.dta` NW-Pacific OFES sample. Background, not part of the pipeline.
- `surface_check.png` — the NW-Pacific SST plot that first revealed the domain.
- `implementation_checklist.md` — the master task list with per-step status.

---

## Status

**✅ Done (designed + coded):** the entire pipeline — acquisition, preprocessing
(3.1–3.6), MHW detection (4), correlation (6), and the calibrated model (5/7).
Environment set up; dataset IDs and grids all verified against live sources.

**⏸️ Not yet run — blocked on disk space:** no script has executed end to end because
the OFES + PFT downloads aren't complete (the PFT download is a partial fragment).
Nothing has produced real numbers yet.

**⬜ Not built:** Phase 8 (validation vs OC-CCI), Phase 9 (visualization),
Phase 10 (write-up — **summary due July 22**).

## Open items to resolve before running
1. **Disk space** — needed for the ~0.7 GB PFT + OFES downloads; clean up the partial file.
2. **OFES record length** — Phase 4's climatology (1983–2012) needs OFES back to ~1980,
   but the 1.1 download is set to 1998–2014. Extend the start year, or shorten `CLIM_PERIOD`.
3. **Config names** — verify OFES variable/coord names in each script's CONFIG block
   against your actual `get_ofes.py` output on the first run.

## Immediate next actions (when space allows)
Download PFT + (extended) OFES → run the pipeline in order on a **single test cell/month**
first → confirm each interim file → then full run → interpret Phase 6 + model skill →
build Phase 8/9 → write up.
