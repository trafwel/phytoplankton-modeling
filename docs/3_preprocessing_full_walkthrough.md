# Walkthrough — Phase 3: Preprocessing (the whole pipeline, step by step)

Goal of Phase 3: take the two **raw** datasets (OFES temperature, PFT chlorophyll)
and turn them into **one analysis-ready table** the model can learn from — one row
per (grid-cell, month), with temperature/heatwave features on one side and
diatom/nano chlorophyll targets on the other.

This doc is the map of the whole phase. Two sub-steps (3.2, 3.3) have their own
detailed docs; the rest are written out here. Read top to bottom — the steps run in
this order and each feeds the next.

---

## First, the big picture: two parallel tracks

The confusing thing about Phase 3 is that OFES temperature travels **two paths at
once**. Hold this picture in your head and everything else falls into place:

```
                         ┌─ (3-day, kept raw) ──► Phase 4 MHW detection ─► monthly MHW features ─┐
   OFES temperature ─────┤                                                                       │
                         └─ monthly-mean ──► monthly SST + SST anomaly (a feature) ──────────────┤
                                                                                                 ├─► 3.6 analysis table
   PFT chlorophyll ──► regrid to OFES grid ──► monthly DIATO/NANO + anomalies + ratio (target) ──┘
```

- The **monthly track** builds features/targets on the common 204-month axis.
- The **3-day track** feeds marine-heatwave detection (Phase 4), which needs the
  fine time resolution; its outputs come back as monthly features.
- Everything converges in **3.6**, the final table.

Now the steps.

---

## Step 3.1 — Subset & mask (clean the raw inputs)

**Why:** raw files carry land, fill values, and more area/variables than you need.
Clean them before doing anything else so later steps aren't fighting garbage.

**What you do:**
- Clip both datasets to the identical Arabian Sea box (5–25°N, 50–75°E). (Your
  downloads are already subset to this, so this is mostly a confirmation.)
- Turn missing/fill values into `NaN` so math ignores them. OFES land = `-1.0E34`
  (mask with `.where(sst > -5e33)`); the PFT NetCDF carries its own `_FillValue`
  that xarray usually applies automatically — verify land shows as `NaN`.
- Keep only what you need: for OFES, the surface level; for PFT, `DIATO`, `NANO`
  (and `CHL` for validation).

**Output:** clean OFES (3-day, surface) and clean PFT, both `NaN` over land.
**Check:** plot one map of each; land should be blank, ocean sensible values.

---

## Step 3.2 — Regrid PFT onto the OFES grid  ·  (detail: `docs/3.2_regridding_walkthrough.md`)

**Why:** the two grids don't line up (OFES 0.1°, PFT 4 km), so a cell in one isn't
the same place as a cell in the other. You can't correlate them until they share a
grid.

**What you do (summary):** conservatively (area-weighted) regrid the fine PFT down
onto the coarse OFES grid; harmonize the longitude convention; handle NaN carefully
so coastal cells aren't smeared. Code already written: `src/regrid_pft_to_ofes.py`.

**Output:** `data/interim/pft_on_ofes_grid.nc` — PFT now on the OFES grid.
**Check:** before/after DIATO maps look the same but blockier; domain-mean preserved.

---

## Step 3.3 — Temporal alignment to monthly  ·  (detail: `docs/3.3_temporal_alignment_walkthrough.md`)

**Why:** PFT is monthly, OFES is 3-day. The joint analysis is monthly, so OFES must
be brought to monthly — **but only for the feature track**; the 3-day OFES stays
alive for Phase 4.

**What you do (summary):** average OFES 3-day → monthly-mean SST; normalize both time
axes to a year-month key (avoid the month-anchor all-NaN trap); trim both to
1998-01…2014-12 and align → **204 months** each.

**Output:** `data/interim/sst_monthly_1998_2014.nc` (+ the untouched 3-day file kept
for Phase 4).
**Check:** both arrays report exactly 204 time steps, labels match one-to-one.

---

## Step 3.4 — Climatology & anomalies (make "unusual" mean something)

**Why:** the Arabian Sea has a huge seasonal cycle (monsoon upwelling). Raw values
mostly tell you the season, not whether something is *unusual*. Anomalies —
value minus the normal for that time of year — are what let you detect a real
collapse or bloom rather than "it's summer."

**Important distinction (don't conflate two climatologies):**
- The **MHW threshold climatology** (per-day-of-year 90th percentile) is built
  *inside Phase 4* on the 3-day→daily SST. That's for detecting heatwaves.
- The **monthly anomaly climatology** here is a *different, simpler* thing: the
  average value for each calendar month (all Januaries, all Februaries, …) over the
  baseline. You subtract it to get monthly anomalies for the feature/target track.

**What you do:**
- **Chlorophyll first gets a log transform.** DIATO/NANO span orders of magnitude and
  are log-normally distributed; working with `log10(chl)` makes anomalies symmetric
  and well-behaved (standard practice in ocean colour). Do the climatology/anomaly
  on the logged values.
- Build the **monthly climatology**: group by calendar month, take the mean (NaN-aware)
  over 1998–2014, per cell. Result: 12 maps (the seasonal cycle).
- Compute **anomalies**: for each month, value − that month's climatology. Do this for
  SST, log-DIATO, log-NANO.
- Optionally **standardize** (divide by each month's standard deviation) to get
  z-score anomalies — convenient later when you set collapse/bloom thresholds
  (e.g. below −1.28σ ≈ 10th percentile).
- Compute the **DCHL/NCHL ratio** = DIATO/NANO (guard divide-by-zero) — your spec's
  core health signal.

**Output:** SST anomaly, log-DIATO anomaly, log-NANO anomaly, DCHL/NCHL ratio, all
monthly on the 204-month grid.
**Check:** anomalies center near zero and are roughly symmetric; the seasonal cycle
is gone from the anomaly series (a cell's summer and winter anomalies look similar in
spread).

---

## Step 3.5 — Photic-zone / depth decision (already made)

**Decision:** satellite PFT is a surface (first optical depth) signal, so pair it with
OFES **surface** temperature. This sidesteps needing full photic-zone coverage.
**Limitation to state:** subsurface chlorophyll maxima aren't captured — a known scope
boundary (the Sakada-san question), revisited only if time allows. Nothing to compute
here; it's why every step above uses surface fields.

---

## Step 3.6 — Assemble the analysis table (the ML-ready output)

**Why:** the model doesn't consume grids; it consumes a **table** — rows of examples,
columns of features + target. This step reshapes everything into that form.

**What you do:**
- Merge onto the common (time=204-month, lat, lon) grid: monthly SST + SST anomaly,
  the monthly MHW features (from Phase 4), log-DIATO/log-NANO + their anomalies, the
  DCHL/NCHL ratio, and any aux drivers.
- **Reshape grid → tidy table**: xarray `.to_dataframe().reset_index()` turns the
  (time, lat, lon) cube into one row per (lat, lon, month). This "long" format is
  what scikit-learn expects.
- **Drop land / all-NaN rows** (cells that were land or fully masked).
- **Keep `lat`, `lon`, `time` as columns** — you'll need them for the spatial/temporal
  blocked cross-validation later (Phase 7), not just as an index.
- Save to `data/processed/` as **parquet** (compact, fast, preserves types).

**Output:** `data/processed/model_table.parquet` — the single input to Phase 5/7.
**Check:** row count ≈ (ocean cells) × 204; no all-NaN feature columns; target columns
(DIATO/NANO anomalies or ratio) present and sane.

---

## Data-flow summary (what exists after each step)

| After | File(s) |
|---|---|
| 3.1 | clean OFES (3-day, surface) + clean PFT, land = NaN |
| 3.2 | `data/interim/pft_on_ofes_grid.nc` |
| 3.3 | `data/interim/sst_monthly_1998_2014.nc` (+ 3-day OFES kept) |
| Phase 4 | `data/interim/mhw_monthly_features.nc` |
| 3.4 | monthly anomalies (SST, log-DIATO, log-NANO), DCHL/NCHL ratio |
| 3.6 | `data/processed/model_table.parquet`  ← feeds the model |

## Order of operations
3.1 → 3.2 → 3.3 → (Phase 4 in parallel on the 3-day file) → 3.4 → 3.6.
3.5 is a standing decision, not a compute step.

## Notes carried forward
- Two distinct climatologies: Phase 4's daily 90th-pct threshold vs 3.4's monthly
  anomaly climatology. Keep them mentally separate.
- Log-transform chlorophyll before anomalies.
- Keep lat/lon/time as columns in the final table for blocked CV.
- The 3-day OFES file survives Phase 3 untouched for Phase 4.
```
