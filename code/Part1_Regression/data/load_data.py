"""
Load processed regression data from disk.

The preprocessing pipeline (EDA, feature engineering, scaling, train/val/test split)
was completed by member A.  This module provides a single entry-point for all
downstream model code so that every notebook/script uses the same data consistently.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default paths (relative to repository root)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]  # …/supervised-learning-project
_PROCESSED_DIR = _REPO_ROOT / "data" / "processed" / "regression"


def load_regression_data(
    processed_dir: Optional[str] = None,
    use_log_target: bool = True,
) -> dict:
    """Return processed train / val / test splits as NumPy arrays.

    Parameters
    ----------
    processed_dir : str or None
        Override for the default ``data/processed/regression/`` path.
    use_log_target : bool, default True
        If True, use ``y_*_log.csv`` (log-transformed cnt).
        If False, use ``y_*_orig.csv`` (raw cnt).

    Returns
    -------
    dict with keys
        X_train, X_val, X_test   – np.ndarray (float64)
        y_train, y_val, y_test   – np.ndarray (float64, 1-D)
        feature_names            – list[str]
        weights_train, weights_val, weights_test – np.ndarray (for WLS)
        y_train_orig, y_val_orig, y_test_orig    – np.ndarray (raw cnt)
    """
    base = Path(processed_dir) if processed_dir else _PROCESSED_DIR
    if not base.exists():
        raise FileNotFoundError(
            f"Processed data directory not found: {base}\n"
            "Run the preprocessing notebooks first (01_eda.ipynb, 02_preprocessing.ipynb)."
        )

    # --- Features ---
    X_train = pd.read_csv(base / "X_train.csv")
    X_val = pd.read_csv(base / "X_val.csv")
    X_test = pd.read_csv(base / "X_test.csv")
    feature_names = list(X_train.columns)

    # --- Target ---
    if use_log_target:
        y_train = pd.read_csv(base / "y_train_log.csv").squeeze("columns")
        y_val = pd.read_csv(base / "y_val_log.csv").squeeze("columns")
        y_test = pd.read_csv(base / "y_test_log.csv").squeeze("columns")
    else:
        y_train = pd.read_csv(base / "y_train_orig.csv").squeeze("columns")
        y_val = pd.read_csv(base / "y_val_orig.csv").squeeze("columns")
        y_test = pd.read_csv(base / "y_test_orig.csv").squeeze("columns")

    # --- Original target (always load for inverse-transform evaluation) ---
    y_train_orig = pd.read_csv(base / "y_train_orig.csv").squeeze("columns").values
    y_val_orig = pd.read_csv(base / "y_val_orig.csv").squeeze("columns").values
    y_test_orig = pd.read_csv(base / "y_test_orig.csv").squeeze("columns").values

    # --- WLS weights (CSV has a column header "0") ---
    w_train = pd.read_csv(base / "weights_train.csv").squeeze("columns").values
    w_val = pd.read_csv(base / "weights_val.csv").squeeze("columns").values
    w_test = pd.read_csv(base / "weights_test.csv").squeeze("columns").values

    return {
        "X_train": X_train.values.astype(np.float64),
        "X_val": X_val.values.astype(np.float64),
        "X_test": X_test.values.astype(np.float64),
        "y_train": y_train.values.astype(np.float64),
        "y_val": y_val.values.astype(np.float64),
        "y_test": y_test.values.astype(np.float64),
        "feature_names": feature_names,
        "weights_train": w_train.astype(np.float64),
        "weights_val": w_val.astype(np.float64),
        "weights_test": w_test.astype(np.float64),
        "y_train_orig": y_train_orig.astype(np.float64),
        "y_val_orig": y_val_orig.astype(np.float64),
        "y_test_orig": y_test_orig.astype(np.float64),
    }


def add_bias_column(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones (bias / intercept term) to X."""
    return np.column_stack([np.ones(X.shape[0]), X])
