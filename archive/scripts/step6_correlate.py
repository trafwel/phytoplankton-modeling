#!/usr/bin/env python3
"""
step6_correlate.py — Phase 6: correlate marine heatwaves with the diatom response.

This is the scientific payload that justifies the forecast: it shows, in the data,
that heatwaves are followed by diatom collapses/blooms. Three complementary views:

  1. LAG-CORRELATION : corr(heatwave at month t-k, diatom anomaly at t) vs lag k.
  2. COMPOSITE       : average diatom anomaly in the months around a heatwave month.
  3. CO-OCCURRENCE   : is a collapse more likely after a heatwave than at baseline?

Works off the merged Phase 3.6 table (which already holds both sides).

Input : data/processed/model_table.parquet   (must contain the MHW features)
Output: figures/phase6_mhw_vs_diato.png  +  data/processed/phase6_stats.csv

Requires: pandas, numpy, matplotlib
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TABLE = "data/processed/model_table.parquet"
FIG   = "figures/phase6_mhw_vs_diato.png"
STATS = "data/processed/phase6_stats.csv"

DRIVER   = "mhw_intensity_cum"   # the heatwave signal
RESPONSE = "phy_anom"            # OFES-BGC phytoplankton response (z-scored log anomaly)
LAGS = range(0, 5)               # months of delay to test
COLLAPSE_Z = -1.0                # phy anomaly below this = "collapse" for co-occurrence


def _sorted(df):
    return df.sort_values(["lat", "lon", "time"])


# ---------------- 1. lag-correlation ----------------
def lag_correlation(df):
    """corr(DRIVER k months earlier, RESPONSE now), pooled over all cells."""
    df = _sorted(df)
    g = df.groupby(["lat", "lon"])
    rows = []
    for k in LAGS:
        drv = g[DRIVER].shift(k)                  # heatwave k months before
        m = drv.notna() & df[RESPONSE].notna()
        r = np.corrcoef(drv[m], df.loc[m, RESPONSE])[0, 1] if m.sum() > 10 else np.nan
        rows.append({"lag_months": k, "correlation": r, "n": int(m.sum())})
    return pd.DataFrame(rows)


# ---------------- 2. composite ----------------
def composite(df, offsets=range(-1, 4), event_col="mhw_days", event_min=1):
    """Average diatom anomaly at t+offset around heatwave 'event' months
    (a month with at least `event_min` MHW days)."""
    df = _sorted(df)
    g = df.groupby(["lat", "lon"])
    event = df[event_col] >= event_min
    rows = []
    for off in offsets:
        resp = g[RESPONSE].shift(-off)            # response `off` months AFTER each row
        m = event & resp.notna()
        rows.append({"offset_months": off,
                     "mean_diato_anom": float(resp[m].mean()) if m.sum() else np.nan,
                     "n": int(m.sum())})
    return pd.DataFrame(rows)


# ---------------- 3. co-occurrence / lift ----------------
def cooccurrence(df, lag=1, event_col="mhw_days", event_min=1):
    """P(collapse at t | heatwave at t-lag) vs baseline P(collapse). Lift>1 => heatwaves
    raise collapse odds."""
    df = _sorted(df)
    g = df.groupby(["lat", "lon"])
    heat_before = g[event_col].shift(lag) >= event_min
    collapse = df[RESPONSE] < COLLAPSE_Z
    base = collapse.mean()
    cond = collapse[heat_before].mean() if heat_before.sum() else np.nan
    return {"baseline_P_collapse": float(base),
            "P_collapse_after_heatwave": float(cond),
            "lift": float(cond / base) if base else np.nan,
            "lag_months": lag}


def main() -> int:
    os.makedirs("figures", exist_ok=True)
    os.makedirs(os.path.dirname(STATS), exist_ok=True)
    df = pd.read_parquet(TABLE)
    for c in (DRIVER, RESPONSE):
        if c not in df.columns:
            raise SystemExit(f"column '{c}' missing — has Phase 4 been merged in 3.6?")

    lc = lag_correlation(df)
    comp = composite(df)
    co = cooccurrence(df, lag=1)

    print("lag-correlation:\n", lc.to_string(index=False))
    print("\ncomposite:\n", comp.to_string(index=False))
    print("\nco-occurrence:", co)

    # save stats
    lc.assign(kind="lag_corr").to_csv(STATS, index=False)
    print(f"saved stats -> {STATS}")

    # figure: lag-correlation bars + composite curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.bar(lc["lag_months"], lc["correlation"])
    ax1.set(xlabel="heatwave lead (months)", ylabel=f"corr with {RESPONSE}",
            title="Lag-correlation")
    ax1.axhline(0, color="k", lw=0.6)
    ax2.plot(comp["offset_months"], comp["mean_diato_anom"], "o-")
    ax2.set(xlabel="months relative to heatwave", ylabel="mean diatom anomaly",
            title="Composite response")
    ax2.axhline(0, color="k", lw=0.6); ax2.axvline(0, color="r", lw=0.6, ls="--")
    fig.tight_layout(); fig.savefig(FIG, dpi=120)
    print(f"saved figure -> {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
