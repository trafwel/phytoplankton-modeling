# OFES Data Reference (as verified from the drive)

A structural description of the OFES data used in this project — format, grid,
variables, regions, and purpose. All facts here were verified directly by reading
the files. (JCOPE reanalysis — a separate, observationally-constrained product — is in §9.)

---

## 1. What OFES is

**OFES** = Ocean General Circulation Model For the Earth Simulator (JAMSTEC) — an
eddy-resolving (0.1°) ocean circulation hindcast. **OFES2** is the newer global
version. Separately, **OFES-BGC** is OFES coupled to a **Sasai NPZD** (Nutrient–
Phytoplankton–Zooplankton–Detritus) biology model.

On your drive there are **two distinct products**:
- a **physical** model output (full North Pacific domain), and
- a **coupled biology** output (one small cropped region).

---

## 2. File format (identical for all `.dta`)

- **Raw binary, no header** — just a stream of numbers.
- **Big-endian 32-bit float** (`numpy` dtype `">f4"`). Reading as little-endian gives
  garbage — this is the #1 gotcha.
- **Reshape to (depth, lat, lon)**, C-order (lon varies fastest):
  - full-domain fields → `(105, 880, 1900)`
  - cropped biology fields → `(28, 300, 300)` **(CORRECTED — see §5.1)**. The crop
    keeps only the **upper 28 depth levels** and a **300×300 horizontal window**.
    (An earlier guess of `(105, 150, 160)` had the same byte count, 2,520,000 floats,
    but the wrong arrangement — it produced diagonal NaN streaks. `(28, 300, 300)` is
    the shape the lab's own extraction code writes and reads.)
- **Missing / land value** = `-9.9999998e+33` (mask anything `<= -5e33` to NaN).
- One file = **one variable, one day**.

## 3. Grid & coordinates

- Horizontal resolution **0.1°**.
- Latitude: `lat = -19.95 + 0.1 * y_index` → **−19.95°N to 67.95°N** (y = 0…879).
- Longitude: `lon = 100.05 + 0.1 * x_index` → **100.05°E to 289.95°E** (x = 0…1899).
  (i.e. the North Pacific from the Asian coast east across the dateline to ~70°W.)
- So the full domain is 880 (lat) × 1900 (lon) = 1,672,000 surface cells.

## 4. Depth

- **105 vertical levels**; surface = index 0 (~2.5 m), deepening to ~5900 m.
- Upper 28 layer-centres (m), used in the lab notebooks:
  2.5, 7.5, 12.5 … 102.5, 107.5, … 147.1.

---

## 5. Products, variables, regions

| Product (path) | Variables | Region | Season / years | Purpose |
|---|---|---|---|---|
| **`OFES2_NP/DATA_3D/daily/yYYYY/`** | `temp`, `u_vel`, `v_vel`, `w_vel` (**physics only**) | **Full N. Pacific** (−20–68°N, 100–290°E), `(105,880,1900)` | Daily, **all-season**: `temp` & `w_vel` **full 2000–2019** (365/366 d/yr); `u_vel`/`v_vel` **2000–2006 only** | Physical circulation hindcast — the continuous seasonal record the biology crop lacks (usable for a proper Hobday MHW climatology) |
| **`OFES_test/Data/`** | `temp`, `phy`, `ntr`, `w` (**coupled biology**) | **One 30°×30° crop, `10–40°N, 120–150°E`** (NW Pacific: Kuroshio + subtropical gyre + Kuroshio Extension / subarctic transition), `(28,300,300)`, upper ~150 m | **daily; see §5.2 for exact coverage** | Coupled OFES-BGC (NPZD) — the biology |
| **`clim/tempclim/`** | temperature climatology (`std.MM.DD.dta`) | — | day-of-year | Climatology baseline |
| **`OFES_daily/standard_deviation/typhoontrack/`** | `phy_depth`, `ntr_depth`, `temp_depth` (3×3 profiles) | along a 2018 typhoon track | Sep–Oct 2018 | Derived products for the lab's typhoon study |

Key point: **the physics is full-domain; the biology (`phy`, `ntr`) exists ONLY in the
one crop, October only.** The crop is 300×300 cells ≈ 5.4% of the domain area.

### 5.2 Temporal coverage of the crop (verified by file listing, 2026-07-15)

Raw **daily** `<var>.MM.DD.YYYY.dta` files (excludes `_anom` and `clim`):

| var | October 2000–2019 | extra months | daily total |
|---|---|---|---|
| `phy` (phytoplankton) | 31×20 = 620 | **Sep 2018** (30) | 650 |
| `ntr` (nutrients)     | 620 | **Sep 2018** (30) | 650 |
| `temp`                | 620 | Sep 2018 (30) + **Jan 2010** (31) | 681 |
| `w`                   | 620 | Sep 2018 (30) + Sep 2010 (30) | 680 |

**Key correction:** the biology is **NOT strictly October-only.** For **2018** it runs a
continuous **Sep 1 – Oct 31 (61 days)** — the typhoon-study window. All other years are
October (31 days) only. Physics-only extras: `temp` Jan 2010 (winter), `w` Sep 2010.

**Precomputed `*_anom` files exist for October only** (620 each). That is why
`model_table*.parquet`, built from the anom files, is October-only — a limitation of the
*anomaly* set, not of the raw data. To use Sep 2018 (e.g. for a 61-day marine-heatwave
window) the anomalies must be computed from the raw dailies.

### 5.1 The crop — authoritative definition (from the lab's code)

The crop's shape and location are **defined in `HD-SHU3/script_test/data.ipynb`**, the
lab's extraction notebook. It reads a full-domain field and slices it:

```python
data_in  = np.fromfile(in_path, ">f4").reshape(105, 880, 1900)   # full N. Pacific
data_out = data_in[0:28, 300:600, 200:500]                        # THE CROP
data_out.tofile(out_path)                                          # -> OFES_test/Data/...
# read back elsewhere in the notebook as:  reshape(28, 300, 300)
```

So the crop is `[depth 0:28, lat 300:600, lon 200:500]` of the full grid:

- **Depth:** levels 0–27 → upper ~150 m (surface to ~147 m).
- **Latitude:** index 300–599 → `-19.95 + 0.1·i` = **10.05°N to 39.95°N** (`10–40°N`).
- **Longitude:** index 200–499 → `100.05 + 0.1·i` = **120.05°E to 149.95°E** (`120–150°E`).

**Verified against the data:** reshaped `(28,300,300)`, the October-2000 surface temp is
91% ocean with a clean south→north gradient — 29.5°C at 10°N, 28.5°C at 25°N, 20.4°C at
40°N — and clean coastlines (no diagonal streaks). This confirms the shape.

**Crop temp = `OFES2_NP` slice, exactly (verified 2026-07-15).** The crop's `temp` is
bit-identical to `OFES2_NP[0, 300:600, 200:500]` over every October date tested
(corr 1.0000, bias 0.000, RMSE 0.000). So the biology crop is literally cut from the
physics hindcast — **same run, no mismatch.** Consequence: the **full-year 2000–2019
`OFES2_NP` temperature is exactly the temperature the coupled NPZD biology experienced**,
so marine heatwaves defined from it are exact (not an approximation). (An earlier
"different run" guess was an artifact of the old wrong `(105,150,160)` reshape.)

**Why this matters:** the region spans **tropical (warm, nutrient-limited) in the south to
temperate (cool, Kuroshio-Extension / subarctic-transition) in the north.** The northern
band (≈30–40°N, SST ~20°C) is exactly the temperature-limited regime where a marine-
heatwave → phytoplankton signal is plausible — a regime our earlier (mis-shaped) runs
never actually sampled. The crop's `w` at index 300 (`region["lat"]-300`) confirms the
lat origin offset of 300 (see `ddt_3×3.ipynb`).

## 6. Filename conventions

Within `OFES_test/Data/{temp,phy,ntr}/`:
- `<var>.MM.DD.YYYY.dta` — the daily field (e.g. `phy.10.01.2000.dta`).
- `<var>.MM.DD.clim.dta` — the **daily climatology** (mean over years for that date).
- `<var>_anom.MM.DD.YYYY.dta` — the **anomaly** (field − climatology) for that day.
  (These precomputed anomalies are what `step1b_multiyear_anom.py` uses.)

Full-domain files: `temp.MM.DD.YYYY.dta` etc. (no clim/anom variants there).

---

## 7. Purpose / provenance

- The **physical OFES2** run is the standard eddy-resolving hindcast (temperature,
  currents, MLD, etc.).
- The **OFES-BGC crop** was produced by the lab (Sasai/Yasunaka group) for a
  **2018 typhoon → phytoplankton study** — which is why it is (a) a small NW-Pacific
  box on the typhoon track, (b) autumn/typhoon-season only, and (c) shipped with
  precomputed climatology, anomalies, and along-track depth profiles.

## 8. Known caveats (see `docs/test_results_log.md` for the full list)

- Biology = one 30°×30° region (10–40°N, 120–150°E), October only, single lumped
  phytoplankton compartment (no diatom/nano split; nitrogen-based, no silicate).
- Coupled model → circularity (phy computed from temp/light/nutrients).
- **Crop geometry is now RESOLVED** (§5.1): shape `(28,300,300)`, region 10–40°N /
  120–150°E, from `script_test/data.ipynb`. Earlier runs used the wrong reshape
  `(105,150,160)` — see `test_results_log.md` §"Shape correction" for what that
  affected and what must be re-run.
- Model output, not observations — no in-dataset validation.

---

## 9. JCOPE-EC reanalysis (drive `HD-SHU`, added 2026-07-16)

A **separate, observationally-constrained product** — JCOPE (JAMSTEC, Miyazawa et al.)
is a **data-assimilating** ocean reanalysis (satellite SST + altimetry + in-situ), unlike
free-running OFES. This directly addresses OFES-BGC's lack of observational anchoring.

- **Domain / grid:** `866 lon × 620 lat × 25 depth`, **1/12° (~9 km)**.
  Lon **107.96–180°E**, lat **10.46–62°N** (western + subarctic N. Pacific — covers the
  crop region 120–150°E/10–40°N *and the full Kuroshio–Oyashio system up to 62°N*).
- **Depth:** 25 levels stored **deep→surface**: −2000,−1500,…,−100,−75,−50,−30,−20,−10,
  **0 (surface = index 24)**.
- **Time:** **daily, ~1993–2018** (~25 years), all seasons.
- **Variables** (each = one file, GrADS): physical `t`,`s`; biogeochem `phy` (phyto),
  `zoo`,`det`,`din` (nitrogen/nutrients),`dip`,`dic`,`alk`,`ph`,`ph25`,`pco2`,`aomega`.
- **File format:** GrADS `.ctl` + `.dat`, gzipped. Each `.dat` = **big-endian float32,
  shape `(25, 620, 866)`** (GrADS order z,y,x; x fastest), one variable stored as `val`,
  `undef −99.9999`. ~24 MB/day/variable gzipped (~54 MB raw).
- **Storage / access:** ~5.5 TB inside ~600 GB **tar archives** per 5-yr block
  (`jcope-ec{1993-4,1995-9,2000-4,2005-9,2010-4,2015-8}.tar`; an `old/` set duplicates).
  The on-disk `t/`,`phy/`,… folders are **empty** (not pre-extracted). Extract per-variable
  in ONE sequential pass per tar (`tar xf … --wildcards 'phy.*'`); use `--occurrence=1`
  when pulling a single member (else tar scans the whole archive). **READ-ONLY** source;
  extract only to a writable disk, never back to `HD-SHU`.
- **Region indices** (approx) for 120–150°E/10–40°N: lon `145–505`, lat `0–355`.
- **Units** (not in `.ctl`; inferred from a 2000–2004 pilot, all sane): `t` in **°C**
  (Feb 10–29, Aug 20–32; north 9.6→24.9 seasonally, south ~28→30 — correct Kuroshio/
  Oyashio seasonality); `phy` ≈ **mmol N/m³** (0.02–0.8; winter/spring > summer, N>S —
  correct bloom seasonality); `din` ≈ **mmol N/m³** nitrate (0–22; **Aug S 0.69 vs N 4.86**
  — reproduces the nutrient-rich-north/poor-south contrast). Nitrogen-based ecosystem model.
- **Pilot extracted:** `src/jcope_extract_pilot.py` → `Ocean Project/JCOPE Data/jcope_pilot/
  {t,phy,din}/<var>.YYYYMMDD.npy`, **surface only, 2000–2004** (1827 days each, ~2.8 GB),
  region-cropped `(1,356,362)`. HD-PGF-A is NTFS/read-only on macOS, so output went to the
  laptop. To extend: set `ZK0,ZK1=15,25` (upper 200 m) and/or run the other year-tars.
