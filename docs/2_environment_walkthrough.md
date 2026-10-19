# Walkthrough — Phase 2: Environment & Project Setup

Prep you can do now, before any data is on disk. Goal: a clean, reproducible
Python environment and a folder layout that keeps raw data safe and derived data
organized. Written for someone newer to the Python data stack.

---

## 2.1 What you already have

From running the CMEMS download you've confirmed working: **Python**, **xarray**,
**copernicusmarine**, and their dependencies (numpy, pandas, netCDF4, dask). That
covers acquisition and basic inspection. Phase 2 just adds the analysis tools and
sets up structure.

## 2.2 Use one isolated environment (why it matters)

Create a single dedicated environment for this project (conda or venv) and install
everything into it. The reason: it pins your library versions so results are
reproducible and a stray upgrade elsewhere can't silently change your numbers.
Activate it every time you work on the thesis. Write the exact versions down
(a `requirements.txt` or `environment.yml`) so the whole pipeline can be rebuilt
later — examiners and future-you will want that.

## 2.3 Packages to add now (grouped by the phase that needs them)

You don't need to understand these yet — just have them installed so later steps
don't stall. Group them mentally by purpose:

- **Marine-heatwave detection (Phase 4):** `marineHeatWaves` — the Hobday reference
  implementation. **Not on PyPI**, so install from GitHub:
  `pip install git+https://github.com/ecjoliver/marineHeatWaves.git` (needs `git`;
  `xcode-select --install` if missing). Or the no-git route: download the single
  `marineHeatWaves.py` into `src/` and `import` it (needs only numpy/scipy). It
  computes the climatology, 90th-percentile threshold, and event metrics for you.
  (`xmhw` is an xarray-native alternative but also not on PyPI.)
- **Regridding the two grids onto one (Phase 3.2):** `xesmf` (area-weighted
  regridding). Note: `xesmf` depends on `esmpy`, which installs most painlessly via
  **conda-forge** — if you're on pip-only it can be fussy, so this is the one to
  install early and confirm it imports.
- **Plotting maps (Phases 3, 9):** `matplotlib` + `cartopy` (coastlines, map
  projections). You already saw `.plot()` works; cartopy adds the geography.
- **The model + interpretability (Phases 5, 7):** `scikit-learn` (Random Forest,
  calibration, metrics) and `shap` (feature-importance explanations).

Install, then **confirm each imports without error** — an unimportable `xesmf`
discovered in Phase 3 is a bad time to find out.

## 2.4 Project folder structure

Set up a layout that separates immutable inputs from things you generate. A layout
that works well:

```
Ocean Project/
├── data/
│   ├── raw/         # exactly as downloaded — NEVER edit these
│   ├── interim/     # regridded / aligned / cleaned intermediate files
│   └── processed/   # final analysis-ready tables (the model's input)
├── docs/            # these walkthroughs + your notes  (already exists)
├── notebooks/       # exploratory work, one notebook per question
├── src/             # reusable functions once code stabilizes
├── figures/         # saved plots for the write-up
└── models/          # trained model files
```

The single most important rule: **`data/raw/` is sacred and read-only.** Everything
else is regenerated from it. If a processing step goes wrong, you delete `interim/`
and rerun — you never re-download. This discipline is also what makes the whole
thing reproducible.

## 2.5 Disk-space strategy (relevant right now)

Since space is tight, plan the footprint before downloading:

- **Raw:** OFES Arabian Sea surface temp + PFT (~0.7 GB) — the two big inputs.
- **Interim:** regridded copies roughly the same size again.
- Ways to stay small: subset tightly (already doing that), consider dropping
  variables you can re-derive (e.g. total `CHL`), delete `interim/` files once
  `processed/` is built, and — if needed — store intermediates as compressed NetCDF
  or Zarr. Keep only `raw/` and `processed/` long-term; `interim/` is disposable.

## 2.6 Verification (Phase 2 exit check)

Before calling the environment done: in a fresh session, activate the env and
import each key library (`xarray`, `copernicusmarine`, `sklearn`, `xesmf`, `cartopy`,
your MHW package) — all should import cleanly. Confirm the `data/` subfolders exist.
That's the green light for preprocessing.

---

## Notes carried forward
- `xesmf`/`esmpy` is the likeliest install snag — sort it now, not in Phase 3.
- Freeze versions (`requirements.txt`/`environment.yml`) for the thesis appendix.
- `data/raw/` read-only; `interim/` disposable to manage disk.
```
