"""
Nonlinear basis function models for regression.

Classes
-------
- PolynomialRegression    : Polynomial basis expansion + linear regression
- RBFRegression           : Gaussian RBF basis functions
- FourierBasisRegression  : Fourier (sin/cos) basis — good for cyclic data

Functions
---------
- add_interaction_terms   : Generate x_i * x_j interaction features
- run_ablation_study      : Systematically remove feature groups and measure impact
"""

from __future__ import annotations

import time
from itertools import combinations
from typing import Optional

import numpy as np


# ============================================================================
# 1. Polynomial Regression
# ============================================================================

class PolynomialRegression:
    """Polynomial basis expansion followed by linear regression (Normal Equations).

    Expands each feature x_j into [x_j, x_j^2, ..., x_j^degree].
    Optionally includes cross-terms (interaction_only=False, include_interactions=True).
    """

    def __init__(self, degree: int = 2, include_interactions: bool = False):
        self.degree = degree
        self.include_interactions = include_interactions
        self.w: np.ndarray | None = None
        self.fit_time: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "PolynomialRegression":
        t0 = time.perf_counter()
        X_poly = self._transform(X)
        X_b = _add_bias(X_poly)

        # Regularise slightly for numerical stability with high-degree polynomials
        XtX = X_b.T @ X_b + 1e-8 * np.eye(X_b.shape[1])
        Xty = X_b.T @ y
        self.w = np.linalg.solve(XtX, Xty)
        self.fit_time = time.perf_counter() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_poly = self._transform(X)
        return _add_bias(X_poly) @ self.w

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Expand features to polynomial basis."""
        features = [X]
        for d in range(2, self.degree + 1):
            features.append(X ** d)

        if self.include_interactions and self.degree >= 2:
            n_orig = X.shape[1]
            for i, j in combinations(range(n_orig), 2):
                features.append((X[:, i] * X[:, j]).reshape(-1, 1))

        return np.hstack(features)


# ============================================================================
# 2. Gaussian RBF Regression
# ============================================================================

class RBFRegression:
    """Gaussian Radial Basis Function regression.

    ϕ_j(x) = exp(-||x - μ_j||^2 / (2σ²))

    Centers μ_j are chosen from the training data (random subset or k-means).
    Then linear regression on the RBF features.
    """

    def __init__(
        self,
        n_centers: int = 50,
        sigma: float = 1.0,
        center_method: str = "random",  # "random" or "kmeans"
    ):
        self.n_centers = n_centers
        self.sigma = sigma
        self.center_method = center_method
        self.centers: np.ndarray | None = None
        self.w: np.ndarray | None = None
        self.fit_time: float = 0.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed: int = 42,
        **kwargs,
    ) -> "RBFRegression":
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)

        # Select centers
        n_centers = min(self.n_centers, len(X))
        if self.center_method == "kmeans":
            self.centers = _simple_kmeans(X, n_centers, seed=seed)
        else:
            idx = rng.choice(len(X), size=n_centers, replace=False)
            self.centers = X[idx].copy()

        # Compute RBF features
        Phi = self._transform(X)
        Phi_b = _add_bias(Phi)

        # Solve
        XtX = Phi_b.T @ Phi_b + 1e-8 * np.eye(Phi_b.shape[1])
        Xty = Phi_b.T @ y
        self.w = np.linalg.solve(XtX, Xty)
        self.fit_time = time.perf_counter() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Phi = self._transform(X)
        return _add_bias(Phi) @ self.w

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Compute RBF features: Φ[i, j] = exp(-||x_i - μ_j||^2 / (2σ²))."""
        # shape: (n_samples, n_centers)
        diff = X[:, np.newaxis, :] - self.centers[np.newaxis, :, :]  # (n, m, d)
        sq_dist = np.sum(diff ** 2, axis=2)  # (n, m)
        return np.exp(-sq_dist / (2 * self.sigma ** 2))


# ============================================================================
# 3. Fourier Basis Regression
# ============================================================================

class FourierBasisRegression:
    """Fourier (sin/cos) basis function regression.

    Suitable for data with cyclic/periodic patterns (e.g. bike sharing by hour).
    For each feature x_j, generates:
        sin(2πk * x_j), cos(2πk * x_j)  for k = 1, ..., n_harmonics
    """

    def __init__(self, n_harmonics: int = 5, feature_indices: Optional[list[int]] = None):
        """
        Parameters
        ----------
        n_harmonics : int
            Number of Fourier harmonics per feature.
        feature_indices : list[int] or None
            If given, apply Fourier transform only to these features.
            Others are passed through as-is.
        """
        self.n_harmonics = n_harmonics
        self.feature_indices = feature_indices
        self.w: np.ndarray | None = None
        self.fit_time: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "FourierBasisRegression":
        t0 = time.perf_counter()
        Phi = self._transform(X)
        Phi_b = _add_bias(Phi)

        XtX = Phi_b.T @ Phi_b + 1e-8 * np.eye(Phi_b.shape[1])
        Xty = Phi_b.T @ y
        self.w = np.linalg.solve(XtX, Xty)
        self.fit_time = time.perf_counter() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Phi = self._transform(X)
        return _add_bias(Phi) @ self.w

    def _transform(self, X: np.ndarray) -> np.ndarray:
        parts = []

        if self.feature_indices is not None:
            fourier_cols = self.feature_indices
            # Pass-through features
            passthrough = [j for j in range(X.shape[1]) if j not in fourier_cols]
            if passthrough:
                parts.append(X[:, passthrough])
        else:
            fourier_cols = list(range(X.shape[1]))

        for j in fourier_cols:
            for k in range(1, self.n_harmonics + 1):
                parts.append(np.sin(2 * np.pi * k * X[:, j]).reshape(-1, 1))
                parts.append(np.cos(2 * np.pi * k * X[:, j]).reshape(-1, 1))

        return np.hstack(parts)


# ============================================================================
# 4. Interaction Terms
# ============================================================================

def add_interaction_terms(
    X: np.ndarray,
    feature_names: Optional[list[str]] = None,
) -> tuple[np.ndarray, list[str]]:
    """Add pairwise interaction terms x_i * x_j to feature matrix.

    Returns
    -------
    X_int : ndarray of shape (n_samples, n_features + n_interactions)
    names : list of feature names including interaction names
    """
    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]

    interaction_features = []
    interaction_names = []

    for i, j in combinations(range(n_features), 2):
        interaction_features.append((X[:, i] * X[:, j]).reshape(-1, 1))
        interaction_names.append(f"{feature_names[i]}*{feature_names[j]}")

    X_int = np.hstack([X] + interaction_features) if interaction_features else X
    names = list(feature_names) + interaction_names

    return X_int, names


# ============================================================================
# 5. Ablation Study
# ============================================================================

def run_ablation_study(
    model_class,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    feature_groups: dict[str, list[int]],
    model_kwargs: dict | None = None,
    fit_kwargs: dict | None = None,
) -> dict:
    """Ablation study: remove one feature group at a time and measure impact.

    Parameters
    ----------
    feature_groups : dict mapping group name -> list of feature indices
        Example: {"temperature": [0, 1], "time": [2, 3, 4], "weather": [5, 6]}

    Returns
    -------
    dict with keys:
        baseline_mse : float
        results      : list of dicts with 'group', 'remaining_features',
                       'mse', 'delta_mse', 'pct_change'
    """
    from ..evaluation.metrics import mse

    model_kwargs = model_kwargs or {}
    fit_kwargs = fit_kwargs or {}
    all_features = list(range(X_train.shape[1]))

    # Baseline with all features
    model = model_class(**model_kwargs)
    model.fit(X_train, y_train, **fit_kwargs)
    baseline = mse(y_val, model.predict(X_val))

    print(f"\n  Ablation Study")
    print(f"  {'─' * 55}")
    print(f"  Baseline (all features): MSE = {baseline:.6f}")
    print(f"  {'─' * 55}")

    results = []
    for group_name, group_indices in feature_groups.items():
        remaining = [f for f in all_features if f not in group_indices]
        if len(remaining) == 0:
            continue

        model = model_class(**model_kwargs)
        model.fit(X_train[:, remaining], y_train, **fit_kwargs)
        ablated_mse = mse(y_val, model.predict(X_val[:, remaining]))

        delta = ablated_mse - baseline
        pct = (delta / baseline) * 100

        results.append({
            "group": group_name,
            "removed_features": group_indices,
            "remaining_n": len(remaining),
            "mse": ablated_mse,
            "delta_mse": delta,
            "pct_change": pct,
        })

        direction = "↑" if delta > 0 else "↓"
        print(f"  Remove '{group_name}' ({len(group_indices)} features): "
              f"MSE = {ablated_mse:.6f}  ({direction}{abs(pct):.1f}%)")

    # Sort by impact (largest MSE increase = most important group)
    results.sort(key=lambda x: x["delta_mse"], reverse=True)

    return {
        "baseline_mse": baseline,
        "results": results,
    }


# ============================================================================
# Helpers
# ============================================================================

def _add_bias(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0]), X])


def _simple_kmeans(
    X: np.ndarray,
    k: int,
    max_iter: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Simple k-means for selecting RBF centres."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=k, replace=False)
    centers = X[idx].copy()

    for _ in range(max_iter):
        # Assign
        dists = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        labels = dists.argmin(axis=1)

        # Update
        new_centers = np.empty_like(centers)
        for c in range(k):
            mask = labels == c
            if mask.sum() > 0:
                new_centers[c] = X[mask].mean(axis=0)
            else:
                new_centers[c] = centers[c]

        if np.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers

    return centers
