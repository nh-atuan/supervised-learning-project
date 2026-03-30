"""
Linear Regression models implemented from scratch.

Classes
-------
- NormalEquationLR   : Closed-form solution w = (X^T X)^{-1} X^T y
- MiniBatchGDLR      : Mini-batch Gradient Descent with learning rate schedules
- WeightedLeastSquares : WLS for heteroscedastic data

Functions
---------
- gauss_markov_diagnostics : Residual plot, QQ-plot, Breusch–Pagan test
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np


# ============================================================================
# 1. Normal Equations
# ============================================================================

class NormalEquationLR:
    """Linear Regression via the Normal Equations (closed-form).

    Solves  w = (X^T X)^{-1} X^T y  where X is augmented with a bias column.
    Does NOT use sklearn.LinearRegression.
    """

    def __init__(self):
        self.w: np.ndarray | None = None
        self.fit_time: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "NormalEquationLR":
        t0 = time.perf_counter()
        X_b = _add_bias(X)
        # Use np.linalg.lstsq for numerical stability instead of direct inverse
        # but implement the formula explicitly:
        #   w = (X^T X)^{-1} X^T y
        XtX = X_b.T @ X_b
        Xty = X_b.T @ y
        self.w = np.linalg.solve(XtX, Xty)
        self.fit_time = time.perf_counter() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_b = _add_bias(X)
        return X_b @ self.w

    @property
    def coefficients(self) -> np.ndarray:
        """Return weights excluding bias."""
        return self.w[1:] if self.w is not None else np.array([])

    @property
    def intercept(self) -> float:
        return float(self.w[0]) if self.w is not None else 0.0


# ============================================================================
# 2. Mini-batch Gradient Descent
# ============================================================================

class MiniBatchGDLR:
    """Linear Regression via Mini-batch Gradient Descent.

    Supports two learning rate schedules:
      - "step_decay"        : lr *= decay_rate every decay_steps epochs
      - "cosine_annealing"  : lr follows cosine schedule to lr_min
    """

    def __init__(
        self,
        lr: float = 0.01,
        n_epochs: int = 500,
        batch_size: int = 64,
        lr_schedule: str = "constant",   # "constant", "step_decay", "cosine_annealing"
        decay_rate: float = 0.5,
        decay_steps: int = 100,
        lr_min: float = 1e-5,
        verbose: bool = False,
    ):
        self.lr_init = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr_schedule = lr_schedule
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.lr_min = lr_min
        self.verbose = verbose

        self.w: np.ndarray | None = None
        self.loss_history: list[float] = []
        self.lr_history: list[float] = []
        self.fit_time: float = 0.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed: int = 42,
        **kwargs,
    ) -> "MiniBatchGDLR":
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        X_b = _add_bias(X)
        n_samples, n_features = X_b.shape

        # Xavier-like init
        self.w = rng.standard_normal(n_features) * np.sqrt(2.0 / n_features)
        self.loss_history = []
        self.lr_history = []

        for epoch in range(self.n_epochs):
            lr = self._get_lr(epoch)
            self.lr_history.append(lr)

            # Shuffle
            perm = rng.permutation(n_samples)
            X_b_shuf = X_b[perm]
            y_shuf = y[perm]

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                X_batch = X_b_shuf[start:end]
                y_batch = y_shuf[start:end]
                m = end - start

                # Gradient: (1/m) * X^T (X w - y)
                residual = X_batch @ self.w - y_batch
                grad = (1.0 / m) * (X_batch.T @ residual)
                self.w -= lr * grad

            # Epoch loss (full training set)
            y_pred = X_b @ self.w
            epoch_loss = float(np.mean((y - y_pred) ** 2))
            self.loss_history.append(epoch_loss)

            if self.verbose and (epoch % 50 == 0 or epoch == self.n_epochs - 1):
                print(f"  Epoch {epoch:4d} | lr={lr:.6f} | MSE={epoch_loss:.6f}")

        self.fit_time = time.perf_counter() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_b = _add_bias(X)
        return X_b @ self.w

    @property
    def coefficients(self) -> np.ndarray:
        return self.w[1:] if self.w is not None else np.array([])

    @property
    def intercept(self) -> float:
        return float(self.w[0]) if self.w is not None else 0.0

    # --- Learning rate schedules ---

    def _get_lr(self, epoch: int) -> float:
        if self.lr_schedule == "step_decay":
            factor = self.decay_rate ** (epoch // self.decay_steps)
            return max(self.lr_init * factor, self.lr_min)
        elif self.lr_schedule == "cosine_annealing":
            return self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (
                1 + np.cos(np.pi * epoch / self.n_epochs)
            )
        else:  # constant
            return self.lr_init


# ============================================================================
# 3. Weighted Least Squares (WLS)
# ============================================================================

class WeightedLeastSquares:
    """Weighted Least Squares for heteroscedastic data.

    Minimises  sum_i  w_i * (y_i - x_i^T β)^2
    Closed-form: β = (X^T W X)^{-1} X^T W y
    where W = diag(weights).
    """

    def __init__(self):
        self.w: np.ndarray | None = None
        self.fit_time: float = 0.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "WeightedLeastSquares":
        """Fit WLS.  If weights is None, falls back to OLS."""
        t0 = time.perf_counter()
        X_b = _add_bias(X)

        if weights is None:
            weights = np.ones(len(y))

        # W is diagonal → efficient to use element-wise multiplication
        # X^T W X = (sqrt(W) X)^T (sqrt(W) X)
        sqrt_w = np.sqrt(weights)
        X_w = X_b * sqrt_w[:, np.newaxis]
        y_w = y * sqrt_w

        XtX = X_w.T @ X_w
        Xty = X_w.T @ y_w
        self.w = np.linalg.solve(XtX, Xty)
        self.fit_time = time.perf_counter() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_b = _add_bias(X)
        return X_b @ self.w

    @property
    def coefficients(self) -> np.ndarray:
        return self.w[1:] if self.w is not None else np.array([])

    @property
    def intercept(self) -> float:
        return float(self.w[0]) if self.w is not None else 0.0


# ============================================================================
# 4. Gauss–Markov Diagnostics
# ============================================================================

def gauss_markov_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: np.ndarray,
    plot: bool = True,
) -> dict:
    """Check Gauss–Markov assumptions.

    1. Residual plot  → visual check for homoscedasticity
    2. QQ-plot        → visual check for normality
    3. Breusch–Pagan  → statistical test for heteroscedasticity

    Parameters
    ----------
    X : design matrix (without bias column)

    Returns
    -------
    dict with Breusch–Pagan test results and diagnostics.
    """
    import statsmodels.stats.diagnostic as smd

    residuals = y_true - y_pred

    # --- Breusch–Pagan test ---
    X_b = _add_bias(X)
    bp_stat, bp_pvalue, bp_f, bp_f_pvalue = smd.het_breuschpagan(residuals, X_b)

    result = {
        "bp_statistic": float(bp_stat),
        "bp_p_value": float(bp_pvalue),
        "bp_f_statistic": float(bp_f),
        "bp_f_p_value": float(bp_f_pvalue),
        "heteroscedastic": bp_pvalue < 0.05,
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std()),
    }

    print("\n=== Gauss–Markov Diagnostics ===")
    print(f"  Residual mean : {result['residual_mean']:.6f}  (should be ≈ 0)")
    print(f"  Residual std  : {result['residual_std']:.6f}")
    print(f"\n  Breusch–Pagan test:")
    print(f"    LM statistic : {result['bp_statistic']:.4f}")
    print(f"    p-value      : {result['bp_p_value']:.6f}")
    print(f"    F statistic  : {result['bp_f_statistic']:.4f}")
    print(f"    F p-value    : {result['bp_f_p_value']:.6f}")
    if result["heteroscedastic"]:
        print("    → Heteroscedasticity DETECTED (p < 0.05)")
        print("    → Consider using Weighted Least Squares (WLS)")
    else:
        print("    → No significant heteroscedasticity (p >= 0.05)")

    if plot:
        from evaluation.plots import plot_residuals, plot_qq

        plot_residuals(y_true, y_pred, title="Gauss–Markov: Residual Analysis")
        plot_qq(y_true, y_pred, title="Gauss–Markov: QQ-Plot")

    return result


# ============================================================================
# Helpers
# ============================================================================

def _add_bias(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones to X."""
    return np.column_stack([np.ones(X.shape[0]), X])
