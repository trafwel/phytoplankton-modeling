#!/usr/bin/env python3
"""
step5_model.py — Phase 5/7: phytoplankton collapse/bloom model.

Built BLOCK BY BLOCK. This file grows one section at a time; it is not the
finished script until all blocks are added.

Reads the analysis table from Phase 3.6 and (eventually) trains a calibrated,
interpretable Random Forest that outputs, per (cell, month), the probability of a
diatom collapse/bloom plus its severity.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, classification_report, confusion_matrix

# ---------------- CONFIG ----------------
TABLE = "OFES Data/model_table.parquet"
MAX_ROWS = 1_500_000   # subsample cap for a fast run; set to None to use all 14M rows

# Predictors (features). SST anomaly is here now; the MHW monthly features get
# added once Phase 4 is coded (names TBD — placeholders shown commented).
FEATURES = [
    "ntr_z",                 # nutrients (the dominant driver)
    "temp_z",                # temperature
    "w_z",                   # subsurface vertical velocity (upwelling)
]

# The quantity we turn into collapse/bloom classes (Block 2).
# Combined run: predict phytoplankton from nutrients + physics together.
TARGET_BASE = "phy_z"

# --- Severity classes (Block 2) ---
# Bin the z-scored anomaly into ordered classes using sigma cutoffs (TARGET_BASE is
# already standardized, so the thresholds ARE in std units).
# CLASS_MODE = 3 for a small test (5 classes get too sparse); 5 for the full run.
CLASS_MODE = 3
if CLASS_MODE == 3:
    CLASS_EDGES = [-0.75, 0.75]              # 2 edges -> 3 bins
    CLASS_CODES = [0, 1, 2]
    CLASS_NAMES = {0: "collapse", 1: "normal", 2: "bloom"}
else:
    CLASS_EDGES = [-1.5, -0.5, 0.5, 1.5]     # 4 edges -> 5 bins
    CLASS_CODES = [0, 1, 2, 3, 4]
    CLASS_NAMES = {
        0: "strong_collapse", 1: "mild_collapse", 2: "normal",
        3: "mild_bloom", 4: "strong_bloom",
    }


def load_table(path: str = TABLE) -> pd.DataFrame:
    """Load the Phase 3.6 table and show what's in it."""
    df = pd.read_parquet(path)
    print(f"loaded table: {df.shape[0]} rows x {df.shape[1]} cols")
    print("columns:", list(df.columns))
    return df


def label_severity(df: pd.DataFrame, base: str = TARGET_BASE) -> pd.DataFrame:
    """Add a 'severity_class' column (ordinal 0..4) from the z-scored anomaly.

    Rows where the anomaly is NaN (land / masked / no data) are dropped, since
    they have no target to learn.
    """
    df = df.dropna(subset=[base]).copy()
    # np.digitize returns the bin index for each value given the edges.
    # With 4 edges it yields 0..4 -> exactly our 5 classes.
    df["severity_class"] = np.digitize(df[base].values, bins=CLASS_EDGES)
    df["severity_name"] = df["severity_class"].map(CLASS_NAMES)
    return df


def show_class_balance(df: pd.DataFrame) -> None:
    """Print how many rows fall in each class (imbalance matters a lot later)."""
    counts = df["severity_class"].value_counts().sort_index()
    total = len(df)
    print("class balance:")
    for code in CLASS_CODES:
        n = int(counts.get(code, 0))
        print(f"  {code} {CLASS_NAMES[code]:16s} {n:7d}  ({100*n/total:5.1f}%)")


# ============================ Block 3: features + lags ============================
def build_features(df: pd.DataFrame, lags=(1, 2)):
    """Add lagged versions of each feature (blooms respond to PRIOR months) and
    return (df, feature_cols). Lags are computed WITHIN each cell's time series."""
    base = [f for f in FEATURES if f in df.columns]
    df = df.sort_values(["lat", "lon", "time"]).copy()
    g = df.groupby(["lat", "lon"], sort=False)
    lag_cols = []
    for f in base:
        for L in lags:
            col = f"{f}_lag{L}"
            df[col] = g[f].shift(L)      # value from L months earlier, same cell
            lag_cols.append(col)
    feature_cols = base + lag_cols
    df = df.dropna(subset=feature_cols)  # drop the first L months per cell
    print(f"features ({len(feature_cols)}): {feature_cols}")
    print(f"rows after feature build: {len(df)}")
    return df, feature_cols


# ============================ Block 4: blocked CV =================================
BLOCK_DEG = 40.0  # spatial block size in grid-INDEX units (~4 deg) for the crop

def make_groups(df: pd.DataFrame) -> np.ndarray:
    """Blocked CV groups. If the table spans multiple YEARS (multi-year run), hold
    out whole years -- the honest test of forecasting to an unseen year. Otherwise
    (single season) fall back to SPATIAL tiles so nearby correlated cells are held
    out together and don't leak."""
    if "year" in df.columns and df["year"].nunique() >= 3:
        return df["year"].values                      # year-blocked CV
    latb = np.floor(df["lat"].values / BLOCK_DEG)
    lonb = np.floor(df["lon"].values / BLOCK_DEG)
    return (latb * 1000 + lonb).astype(int)           # spatial fallback


# ============================ Block 5: the model =================================
def make_model() -> RandomForestClassifier:
    """Random Forest: interpretable, non-linear, gives predict_proba.
    class_weight handles the collapse/bloom rarity; min_samples_leaf curbs overfit."""
    return RandomForestClassifier(
        n_estimators=150,        # reduced for the fast test run
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=0,
    )


# ============================ Block 6: out-of-fold predictions ====================
def out_of_fold(df, feature_cols, target="severity_class", n_splits=5):
    """Predict every row with a model that never saw its year-block => honest
    probabilities for the whole dataset."""
    X = df[feature_cols].values
    y = df[target].values
    groups = make_groups(df)
    classes = np.array(sorted(np.unique(y)))
    proba = np.zeros((len(y), len(classes)))
    pred = np.empty(len(y), dtype=int)
    cv = GroupKFold(n_splits=n_splits)
    for k, (tr, te) in enumerate(cv.split(X, y, groups), 1):
        m = make_model().fit(X[tr], y[tr])
        p = m.predict_proba(X[te])
        for j, c in enumerate(m.classes_):          # align to global class order
            proba[te, np.searchsorted(classes, c)] = p[:, j]
        pred[te] = m.predict(X[te])
        print(f"  fold {k}: train={len(tr)} test={len(te)}")
    return classes, proba, pred, y


# ============================ Block 7: evaluation ================================
def evaluate(classes, proba, pred, y):
    """Probability quality + per-class performance (accuracy is misleading here)."""
    p = np.clip(proba, 1e-6, None)
    p = p / p.sum(axis=1, keepdims=True)            # guard log_loss against 0s
    print(f"log-loss: {log_loss(y, p, labels=classes):.4f} (lower=better)")
    names = [CLASS_NAMES[c] for c in classes]
    print(classification_report(y, pred, labels=classes, target_names=names,
                                zero_division=0))
    print("confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y, pred, labels=classes))


# ============================ Block 8: interpretability ==========================
def interpret(df, feature_cols, target="severity_class"):
    """Fit on all data and report what drives the model."""
    m = make_model().fit(df[feature_cols].values, df[target].values)
    imp = pd.Series(m.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("feature importances:\n", imp.to_string())
    try:
        import shap
        expl = shap.TreeExplainer(m)
        sample = df[feature_cols].sample(min(2000, len(df)), random_state=0)
        sv = expl.shap_values(sample)
        print("SHAP computed on", len(sample), "rows (use shap.summary_plot for the figure)")
    except Exception as e:
        print(f"(SHAP skipped: {e})")
    return m


# ============================ Block 9: final fit + calibrate + save ==============
def fit_final_and_save(df, feature_cols, target="severity_class",
                       out="models/rf_severity.joblib"):
    """Calibrate probabilities (so predict_proba is trustworthy) and save.
    NOTE: for full rigor, calibration folds should also be year-blocked; the
    internal cv here is a reasonable first pass."""
    cal = CalibratedClassifierCV(make_model(), method="isotonic", cv=3)
    cal.fit(df[feature_cols].values, df[target].values)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    joblib.dump({"model": cal, "features": feature_cols, "classes": CLASS_NAMES}, out)
    print(f"saved calibrated model -> {out}")
    return cal


def main():
    df = load_table()
    df = label_severity(df)
    show_class_balance(df)
    df, feature_cols = build_features(df)
    if MAX_ROWS and len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=0)
        print(f"subsampled to {len(df):,} rows for a fast run "
              f"(set MAX_ROWS=None for all)")
    classes, proba, pred, y = out_of_fold(df, feature_cols)
    evaluate(classes, proba, pred, y)
    interpret(df, feature_cols)
    fit_final_and_save(df, feature_cols)


if __name__ == "__main__":
    main()
