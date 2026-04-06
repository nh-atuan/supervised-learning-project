"""
Evaluation metrics for regression models.

All functions are implemented from scratch (no sklearn).
sklearn equivalents are used only in ``verify_against_sklearn()`` for sanity checks.
"""

from __future__ import annotations

import numpy as np


# ============================================================================
# 1. Core Metrics
# ============================================================================

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (R²)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MSE, RMSE, MAE, R² and return as a dict."""
    return {
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R²": r_squared(y_true, y_pred),
    }


def print_metrics(metrics: dict[str, float], model_name: str = "Model") -> None:
    """Pretty-print a metrics dict."""
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    for name, value in metrics.items():
        print(f"  {name:>6s}: {value:.6f}")
    print(f"{'='*50}\n")


# ============================================================================
# 2. Cross-Validation
# ============================================================================

def k_fold_split(
    n_samples: int,
    k: int = 10,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate k-fold train/val index arrays.

    Returns
    -------
    folds : list of (train_indices, val_indices) tuples
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[: n_samples % k] += 1
    folds = []
    current = 0
    for size in fold_sizes:
        val_idx = indices[current : current + size]
        train_idx = np.concatenate([indices[:current], indices[current + size :]])
        folds.append((train_idx, val_idx))
        current += size
    return folds


def cross_validate_model(
    model_class,
    X: np.ndarray,
    y: np.ndarray,
    k: int = 10,
    seed: int = 42,
    model_kwargs: dict | None = None,
    fit_kwargs: dict | None = None,
) -> dict[str, dict[str, float]]:
    """Perform k-fold CV and return metric statistics.

    Parameters
    ----------
    model_class : class
        Must implement ``fit(X, y, **fit_kwargs)`` and ``predict(X)`` methods.
    X, y : array-like
        Full dataset (train + val combined, NOT test).
    k : int
        Number of folds.
    model_kwargs : dict
        kwargs passed to ``model_class(**model_kwargs)``.
    fit_kwargs : dict
        kwargs passed to ``model.fit()``.

    Returns
    -------
    dict with keys "MSE", "RMSE", "MAE", "R²", each mapping to
    {"mean": float, "std": float, "folds": list[float]}.
    """
    model_kwargs = model_kwargs or {}
    fit_kwargs = fit_kwargs or {}
    folds = k_fold_split(len(X), k=k, seed=seed)

    fold_metrics: dict[str, list[float]] = {
        "MSE": [], "RMSE": [], "MAE": [], "R²": [],
    }

    for train_idx, val_idx in folds:
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        model = model_class(**model_kwargs)
        model.fit(X_tr, y_tr, **fit_kwargs)
        y_pred = model.predict(X_va)

        m = compute_all_metrics(y_va, y_pred)
        for key in fold_metrics:
            fold_metrics[key].append(m[key])

    results: dict[str, dict[str, float]] = {}
    for key, vals in fold_metrics.items():
        arr = np.array(vals)
        results[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "folds": vals,
        }
    return results


def print_cv_results(
    cv_results: dict[str, dict[str, float]],
    model_name: str = "Model",
) -> None:
    """Pretty-print cross-validation results."""
    print(f"\n{'='*55}")
    print(f"  {model_name} — {len(cv_results.get('MSE', {}).get('folds', []))}-fold CV")
    print(f"{'='*55}")
    for metric, stats in cv_results.items():
        print(f"  {metric:>6s}: {stats['mean']:.6f} ± {stats['std']:.6f}")
    print(f"{'='*55}\n")


# ============================================================================
# 3. Verification against sklearn
# ============================================================================

def verify_against_sklearn(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Compare custom metrics with sklearn to validate correctness."""
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    custom = compute_all_metrics(y_true, y_pred)
    sklearn_vals = {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R²": r2_score(y_true, y_pred),
    }

    print("\nMetric verification (custom vs sklearn):")
    all_ok = True
    for key in custom:
        diff = abs(custom[key] - sklearn_vals[key])
        status = "✓" if diff < 1e-10 else "✗"
        if diff >= 1e-10:
            all_ok = False
        print(f"  {key:>6s}  custom={custom[key]:.10f}  sklearn={sklearn_vals[key]:.10f}  diff={diff:.2e}  {status}")
    if all_ok:
        print("  All metrics match sklearn! ✓\n")
    else:
        print("  WARNING: Some metrics differ! ✗\n")
