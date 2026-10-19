# Walkthrough — Phase 4: Marine-Heatwave Detection

The heart of the pipeline: turn OFES temperature into marine-heatwave (MHW) events
and metrics, which become the features that drive the phytoplankton model. Grounded
in the Hobday et al. (2016) definition you cite. Written for someone newer to the
tools.

---

## What a marine heatwave IS (the definition)

A **marine heatwave** is a prolonged period of anomalously warm sea temperature at
a location. The standard (Hobday et al. 2016) definition has three ingredients:

1. A **seasonally-varying climatology** — the "normal" temperature for each
   day-of-year at that spot (so summer isn't automatically flagged just for being
   warm).
2. A **threshold** — the **90th percentile** of temperature for each day-of-year,
   computed over a fixed baseline period. "Unusually warm for this date" = above
   this line.
3. A **duration rule** — temperature must stay above the threshold for **≥5
   consecutive days** to count as an event (brief spikes don't qualify). Two events
   separated by a gap of ≤2 days are merged into one.

So a MHW is not "warm water" — it's "water warmer than the 90th percentile *for that
time of year*, sustained for at least 5 days." That day-of-year framing is what makes
it meaningful in a monsoon-driven place like the Arabian Sea.

## The metrics each event carries

Every detected event comes with descriptors you'll turn into model features:
- **duration** (days)
- **intensity_mean / intensity_max** (°C above climatology)
- **intensity_cumulative** (°C·days — the "total heat" of the event)
- **rate of onset / decline**
- **category**: moderate / strong / severe / extreme (1×–4× the distance from
  climatology to threshold).

## Key point: this is a deterministic algorithm, not a model to train

Detection is a fixed recipe — there's **no machine learning here**. This resolves
your spec's "pull a public model or make one" note: for MHWs you run the standard
algorithm, full stop. The ML in this project is *only* the phytoplankton-anomaly
model (Phase 5/7); MHW metrics are engineered **inputs** to it.

## The tool: `marineHeatWaves`

The reference implementation works on **one time series at a time** (a single grid
cell's temperature vs time). Core call: `detect(t, sst)` where `t` is an array of
ordinal dates and `sst` the temperatures; it returns (a) an events dictionary with
all the metrics above and (b) the climatology/threshold it built. Key parameters:
`pctile=90`, `climatologyPeriod=[startYear, endYear]`, `minDuration=5`, `maxGap=2`.

## Adaptation 1 — 3-day data vs a daily definition (the important one)

Hobday assumes **daily** data; OFES is **3-day**. The "5 consecutive days" and the
windowing all assume daily spacing. Two ways to handle it:

- **(Recommended) Interpolate OFES SST from 3-day to daily, then run the standard
  algorithm unchanged.** Linear interpolation across a 3-day gap is mild and keeps
  you fully comparable to the published MHW literature and definition. Cleanest,
  most defensible.
- **(Alternative) Adapt the parameters** to the 3-day grid (e.g. minDuration in
  3-day steps). Avoids interpolation but makes your events subtly non-standard and
  harder to compare to other studies.

Go with interpolation to daily. Note the choice (and that it's a minor smoothing) in
your methods section.

## Adaptation 2 — the climatology baseline period

The 90th-percentile threshold needs a stable, long baseline. You have OFES
**1980–2014**. Use a long climatology window (the full record, or a fixed 30-year
span like 1983–2012) even though your *analysis* window is 1998–2014 — a longer
baseline gives a more stable percentile. The baseline choice shifts what counts as
"extreme," so state it explicitly. (Warming trend caveat: a fixed baseline means
later years show more MHWs partly due to trend — that's expected and is itself
scientifically interesting for a climate-change framing.)

## Applying it across the grid

`detect` is **point-wise**, so you run it independently at every grid cell in the
Arabian Sea box, looping (or vectorizing) over lat/lon. Each cell yields its own MHW
time series (flag + intensity per day). This is the heavy compute step — start on a
**single cell**, confirm it works and looks sane, then scale to the whole grid.
(`xmhw` is an xarray-native option that vectorizes this if the per-cell loop is too
slow — but get the single-cell version right first.)

## Output, and rejoining 3.3

Detection produces per-cell, per-**day** MHW flag + intensity. Aggregate these to
**monthly** features (rejoining the 204-month axis from 3.3):
- MHW days per month
- monthly max intensity, monthly mean intensity
- monthly cumulative intensity
- highest category that month
These monthly MHW features sit next to the monthly PFT target in the model table.

## "Model MHWs" caveat

Because OFES is a model (not observations), these are *simulated* MHWs — internally
consistent, but validate against known Arabian Sea heatwave years in the literature
(Phase 8) so you can trust that OFES reproduces real events.

---

## Fill-in skeleton (template — you write the bodies)

```python
import numpy as np
import xarray as xr
import marineHeatWaves as mhw
from datetime import date

ofes = xr.open_dataset("data/interim/ofes_arabiansea_sst_1998_2014.nc")  # 3-day, keep this!
# NOTE: for the climatology, load the FULL OFES record (1980-2014), not just 1998+

# --- one cell first ---
# TODO: pick a cell (e.g. an offshore point); pull its 1-D sst series + dates
# TODO: interpolate 3-day -> daily (pandas/xarray .resample("1D").interpolate())
# TODO: t = np.array([d.toordinal() for d in daily_dates])
# TODO: mhws, clim = mhw.detect(t, sst_daily, climatologyPeriod=[1983,2012])
# TODO: inspect mhws["duration"], mhws["intensity_max"], mhws["category"] ...

# --- then the whole grid ---
# TODO: loop over lat/lon (or use xmhw); build a per-cell daily MHW flag+intensity cube
# TODO: aggregate to monthly features (days-in-MHW, max/mean/cum intensity, category)
# TODO: save -> data/interim/mhw_monthly_features.nc  (joins the 204-month axis)
```

## Verification
- Single-cell check: plot the cell's daily SST with the climatology and 90th-pct
  threshold overlaid; shade detected events — they should sit on the visible warm
  spells, not on ordinary summers.
- Sanity-check counts: a handful of events per year in a typical cell, not zero and
  not hundreds.
- Confirm a known Arabian Sea MHW year shows up (cross-check literature in Phase 8).

## Notes carried forward
- MHW detection is deterministic (no training) — its outputs are **features** for
  the Phase 5/7 model.
- Interpolate 3-day → daily so the standard Hobday definition applies unchanged.
- Use a long climatology baseline (e.g. 1983–2012); state it in methods.
- Monthly-aggregated MHW features rejoin the 204-month table from 3.3.
```
