"""
Regularized Regression models implemented from scratch.

Classes
-------
- RidgeRegression      : L2 regularization (closed-form)
- LassoRegression      : L1 regularization (coordinate descent)
- ElasticNetRegression : L1 + L2 (coordinate descent)

Functions
---------
- select_lambda_cv     : Grid search λ via k-fold CV
- compute_lasso_path   : Lasso path (warm start) for regularization path plots
- compute_ridge_path   : Ridge path for regularization path plots
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np


# ============================================================================
# 1. Ridge Regression (L2)
# ============================================================================

class RidgeRegression:
    """Ridge Regression via closed-form solution.

    Minimises:  (1/2n) ||y - Xw||^2  +  (λ/2) ||w||^2

    Closed-form:  w = (X^T X + λ I)^{-1} X^T y
    (Note: bias term is NOT regularized.)
    """

    def __init__(self, lam: float = 1.0):
        self.lam = lam
        self.w: np.ndarray | None = None
        self.fit_time: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "RidgeRegression":
        t0 = time.perf_counter()
        X_b = _add_bias(X)
        n_samples, n_features = X_b.shape

        # Regularize all except bias
        reg_matrix = self.lam * np.eye(n_features)
        reg_matrix[0, 0] = 0  # don't regularize bias

        XtX = X_b.T @ X_b
        Xty = X_b.T @ y
        self.w = np.linalg.solve(XtX + reg_matrix, Xty)
        self.fit_time = time.perf_counter() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _add_bias(X) @ self.w

    @property
    def coefficients(self) -> np.ndarray:
        return self.w[1:] if self.w is not None else np.array([])

    @property
    def intercept(self) -> float:
        return float(self.w[0]) if self.w is not None else 0.0


# ============================================================================
# 2. Lasso Regression (L1) — Coordinate Descent
# ============================================================================

class LassoRegression:
    """Lasso Regression via Coordinate Descent.

    Minimises:  (1/2n) ||y - Xw||^2  +  λ ||w||_1

    Uses the soft-thresholding operator.
    """

    def __init__(
        self,
        lam: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ):
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.w: np.ndarray | None = None
        self.n_iter: int = 0
        self.fit_time: float = 0.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        warm_start_w: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "LassoRegression":
        t0 = time.perf_counter()
        X_b = _add_bias(X)
        n, p = X_b.shape

        if warm_start_w is not None:
            self.w = warm_start_w.copy()
        else:
            self.w = np.zeros(p)

        for iteration in range(self.max_iter):
            w_old = self.w.copy()

            for j in range(p):
                # Partial residual excluding feature j
                residual = y - X_b @ self.w + X_b[:, j] * self.w[j]
                rho = X_b[:, j] @ residual / n

                if j == 0:
                    # Don't regularize bias
                    self.w[j] = rho
                else:
                    z = np.sum(X_b[:, j] ** 2) / n
                    self.w[j] = _soft_threshold(rho, self.lam) / z

            # Convergence check
            if np.max(np.abs(self.w - w_old)) < self.tol:
                self.n_iter = iteration + 1
                break
        else:
            self.n_iter = self.max_iter

        self.fit_time = time.perf_counter() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _add_bias(X) @ self.w

    @property
    def coefficients(self) -> np.ndarray:
        return self.w[1:] if self.w is not None else np.array([])

    @property
    def intercept(self) -> float:
        return float(self.w[0]) if self.w is not None else 0.0

    @property
    def n_nonzero_coefs(self) -> int:
        return int(np.sum(np.abs(self.coefficients) > 1e-10))


# ============================================================================
# 3. Elastic Net (L1 + L2) — Coordinate Descent
# ============================================================================

class ElasticNetRegression:
    """Elastic Net via Coordinate Descent.

    Minimises:  (1/2n) ||y - Xw||^2  +  λ₁ ||w||_1  +  (λ₂/2) ||w||^2
    """

    def __init__(
        self,
        lam1: float = 0.5,
        lam2: float = 0.5,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ):
        self.lam1 = lam1  # L1
        self.lam2 = lam2  # L2
        self.max_iter = max_iter
        self.tol = tol
        self.w: np.ndarray | None = None
        self.n_iter: int = 0
        self.fit_time: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "ElasticNetRegression":
        t0 = time.perf_counter()
        X_b = _add_bias(X)
        n, p = X_b.shape
        self.w = np.zeros(p)

        for iteration in range(self.max_iter):
            w_old = self.w.copy()

            for j in range(p):
                residual = y - X_b @ self.w + X_b[:, j] * self.w[j]
                rho = X_b[:, j] @ residual / n

                if j == 0:
                    self.w[j] = rho
                else:
                    z = np.sum(X_b[:, j] ** 2) / n + self.lam2
                    self.w[j] = _soft_threshold(rho, self.lam1) / z

            if np.max(np.abs(self.w - w_old)) < self.tol:
                self.n_iter = iteration + 1
                break
        else:
            self.n_iter = self.max_iter

        self.fit_time = time.perf_counter() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _add_bias(X) @ self.w

    @property
    def coefficients(self) -> np.ndarray:
        return self.w[1:] if self.w is not None else np.array([])

    @property
    def intercept(self) -> float:
        return float(self.w[0]) if self.w is not None else 0.0


# ============================================================================
# 4. Hyperparameter Selection
# ============================================================================

def select_lambda_cv(
    model_class,
    X: np.ndarray,
    y: np.ndarray,
    lambdas: np.ndarray,
    k: int = 10,
    seed: int = 42,
    model_kwargs: dict | None = None,
    lambda_param_name: str = "lam",
) -> dict:
    """Select best λ via k-fold cross-validation (grid search).

    Returns
    -------
    dict with keys: best_lambda, best_mse, all_mses, all_stds
    """
    from ..evaluation.metrics import k_fold_split, mse

    model_kwargs = model_kwargs or {}
    folds = k_fold_split(len(X), k=k, seed=seed)

    all_mean_mses = []
    all_std_mses = []

    for lam in lambdas:
        fold_mses = []
        for train_idx, val_idx in folds:
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]

            kwargs = {**model_kwargs, lambda_param_name: lam}
            model = model_class(**kwargs)
            model.fit(X_tr, y_tr)
            fold_mses.append(mse(y_va, model.predict(X_va)))

        all_mean_mses.append(float(np.mean(fold_mses)))
        all_std_mses.append(float(np.std(fold_mses)))

    best_idx = int(np.argmin(all_mean_mses))
    return {
        "best_lambda": float(lambdas[best_idx]),
        "best_mse": all_mean_mses[best_idx],
        "all_mses": all_mean_mses,
        "all_stds": all_std_mses,
        "lambdas": lambdas.tolist(),
    }


def compute_ridge_path(
    X: np.ndarray,
    y: np.ndarray,
    lambdas: np.ndarray,
) -> np.ndarray:
    """Compute Ridge coefficient path.

    Returns
    -------
    coefs : ndarray of shape (len(lambdas), n_features)
    """
    coefs = []
    for lam in lambdas:
        model = RidgeRegression(lam=lam)
        model.fit(X, y)
        coefs.append(model.coefficients.copy())
    return np.array(coefs)


def compute_lasso_path(
    X: np.ndarray,
    y: np.ndarray,
    lambdas: np.ndarray,
    max_iter: int = 2000,
    tol: float = 1e-6,
) -> np.ndarray:
    """Compute Lasso coefficient path with warm start.

    Lambdas should be sorted from largest to smallest for best warm-start
    performance.

    Returns
    -------
    coefs : ndarray of shape (len(lambdas), n_features)
    """
    # Sort lambdas descending for warm start
    sorted_idx = np.argsort(-lambdas)
    lambdas_sorted = lambdas[sorted_idx]

    coefs = np.zeros((len(lambdas), X.shape[1]))
    current_w = None

    for i, lam in enumerate(lambdas_sorted):
        model = LassoRegression(lam=lam, max_iter=max_iter, tol=tol)
        model.fit(X, y, warm_start_w=current_w)
        original_idx = sorted_idx[i]
        coefs[original_idx] = model.coefficients.copy()
        current_w = model.w.copy()  # pass full w (including bias) for warm start

    return coefs


# ============================================================================
# Helpers
# ============================================================================

def _add_bias(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0]), X])


def _soft_threshold(rho: float, lam: float) -> float:
    """Soft-thresholding operator: S(ρ, λ) = sign(ρ) max(|ρ| - λ, 0)."""
    if rho > lam:
        return rho - lam
    elif rho < -lam:
        return rho + lam
    else:
        return 0.0
