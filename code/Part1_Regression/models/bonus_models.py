"""
Bonus regression models for extra credit.

Classes
-------
- BayesianLinearRegression    : Full posterior p(w|t), predictive distribution
- EvidenceMaximization        : EM-style α, β optimisation
- KernelRidgeRegression       : Kernel Ridge with RBF / polynomial kernels
- GaussianProcessRegression   : GP with RBF kernel + log-marginal-likelihood
- RobustRegression            : IRLS with Huber loss

Functions
---------
- bias_variance_bootstrap     : Bootstrap 200x to estimate bias² & variance
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np


# ============================================================================
# 1. Bayesian Linear Regression
# ============================================================================

class BayesianLinearRegression:
    """Bayesian Linear Regression with conjugate Gaussian prior.

    Prior:     p(w) = N(0, α⁻¹ I)
    Likelihood: p(t|w) = N(Xw, β⁻¹ I)
    Posterior:  p(w|t) = N(m_N, S_N)
      - S_N = (α I + β X^T X)^{-1}
      - m_N = β S_N X^T t

    Predictive: p(t*|x*) = N(m_N^T x*, σ²_N(x*))
      - σ²_N(x*) = 1/β + x*^T S_N x*
    """

    def __init__(self, alpha: float = 1.0, beta: float = 25.0):
        self.alpha = alpha  # precision of prior
        self.beta = beta    # precision of noise
        self.m_N: np.ndarray | None = None
        self.S_N: np.ndarray | None = None
        self.fit_time: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "BayesianLinearRegression":
        t0 = time.perf_counter()
        X_b = _add_bias(X)
        n, d = X_b.shape

        # Posterior covariance
        self.S_N = np.linalg.inv(self.alpha * np.eye(d) + self.beta * X_b.T @ X_b)
        # Posterior mean
        self.m_N = self.beta * self.S_N @ X_b.T @ y
        self.fit_time = time.perf_counter() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Point prediction (posterior mean)."""
        X_b = _add_bias(X)
        return X_b @ self.m_N

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return mean prediction and standard deviation.

        Returns
        -------
        y_mean : ndarray (n,)
        y_std  : ndarray (n,) — predictive standard deviation
        """
        X_b = _add_bias(X)
        y_mean = X_b @ self.m_N
        # Predictive variance: 1/β + x^T S_N x
        y_var = 1.0 / self.beta + np.sum((X_b @ self.S_N) * X_b, axis=1)
        y_std = np.sqrt(np.maximum(y_var, 0))
        return y_mean, y_std

    @property
    def coefficients(self) -> np.ndarray:
        return self.m_N[1:] if self.m_N is not None else np.array([])

    @property
    def intercept(self) -> float:
        return float(self.m_N[0]) if self.m_N is not None else 0.0


# ============================================================================
# 2. Evidence Maximization (Empirical Bayes)
# ============================================================================

class EvidenceMaximization:
    """Optimise α, β by maximising log marginal likelihood (evidence).

    Uses the EM-style re-estimation equations:
        γ    = Σ_i λ_i / (α + λ_i)
        α_new = γ / m_N^T m_N
        β_new = (N - γ) / ||t - Φ m_N||^2

    where λ_i are eigenvalues of β Φ^T Φ.
    """

    def __init__(
        self,
        alpha_init: float = 1.0,
        beta_init: float = 25.0,
        max_iter: int = 200,
        tol: float = 1e-6,
    ):
        self.alpha = alpha_init
        self.beta = beta_init
        self.max_iter = max_iter
        self.tol = tol
        self.model: BayesianLinearRegression | None = None
        self.history: list[dict] = []
        self.fit_time: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "EvidenceMaximization":
        t0 = time.perf_counter()
        X_b = _add_bias(X)
        n, d = X_b.shape

        for iteration in range(self.max_iter):
            # E-step: compute posterior with current α, β
            S_N_inv = self.alpha * np.eye(d) + self.beta * X_b.T @ X_b
            S_N = np.linalg.inv(S_N_inv)
            m_N = self.beta * S_N @ X_b.T @ y

            # Eigenvalues of β Φ^T Φ
            eigvals = np.linalg.eigvalsh(self.beta * X_b.T @ X_b)

            # γ = sum(λ_i / (α + λ_i))
            gamma = float(np.sum(eigvals / (self.alpha + eigvals)))

            # M-step
            alpha_new = gamma / (m_N @ m_N)
            residual = y - X_b @ m_N
            beta_new = (n - gamma) / (residual @ residual)

            # Log marginal likelihood (for tracking)
            log_det_A = np.sum(np.log(self.alpha + eigvals))
            log_ml = (
                0.5 * d * np.log(self.alpha)
                + 0.5 * n * np.log(self.beta)
                - 0.5 * self.beta * (residual @ residual)
                - 0.5 * self.alpha * (m_N @ m_N)
                - 0.5 * log_det_A
                - 0.5 * n * np.log(2 * np.pi)
            )

            self.history.append({
                "iteration": iteration,
                "alpha": float(alpha_new),
                "beta": float(beta_new),
                "gamma": gamma,
                "log_marginal_likelihood": float(log_ml),
            })

            # Convergence check
            if (abs(alpha_new - self.alpha) < self.tol and
                    abs(beta_new - self.beta) < self.tol):
                self.alpha = float(alpha_new)
                self.beta = float(beta_new)
                break

            self.alpha = float(alpha_new)
            self.beta = float(beta_new)

        # Final model
        self.model = BayesianLinearRegression(alpha=self.alpha, beta=self.beta)
        self.model.fit(X, y)
        self.fit_time = time.perf_counter() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_with_uncertainty(self, X: np.ndarray):
        return self.model.predict_with_uncertainty(X)


# ============================================================================
# 3. Kernel Ridge Regression
# ============================================================================

class KernelRidgeRegression:
    """Kernel Ridge Regression.

    Solves:  α = (K + λI)^{-1} y
    Prediction: f(x*) = k(x*, X_train)^T α

    Supports 'rbf' and 'polynomial' kernels.
    """

    def __init__(
        self,
        kernel: str = "rbf",     # "rbf" or "polynomial"
        lam: float = 1.0,
        gamma: float = 0.1,     # for RBF: k(x,y) = exp(-γ ||x-y||²)
        degree: int = 3,        # for polynomial
        coef0: float = 1.0,     # for polynomial
    ):
        self.kernel = kernel
        self.lam = lam
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha_: np.ndarray | None = None
        self.X_train: np.ndarray | None = None
        self.fit_time: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "KernelRidgeRegression":
        t0 = time.perf_counter()
        self.X_train = X.copy()
        K = self._compute_kernel(X, X)
        n = K.shape[0]
        self.alpha_ = np.linalg.solve(K + self.lam * np.eye(n), y)
        self.fit_time = time.perf_counter() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        K = self._compute_kernel(X, self.X_train)
        return K @ self.alpha_

    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if self.kernel == "rbf":
            sq_dist = (
                np.sum(X1 ** 2, axis=1, keepdims=True)
                + np.sum(X2 ** 2, axis=1, keepdims=True).T
                - 2 * X1 @ X2.T
            )
            return np.exp(-self.gamma * sq_dist)
        elif self.kernel == "polynomial":
            return (self.coef0 + X1 @ X2.T) ** self.degree
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")


# ============================================================================
# 4. Gaussian Process Regression
# ============================================================================

class GaussianProcessRegression:
    """Gaussian Process Regression with RBF kernel.

    k(x, x') = σ² exp(-||x - x'||² / (2 l²))

    Optimises log-marginal-likelihood via gradient ascent to learn
    kernel parameters (signal variance σ² and length scale l).
    """

    def __init__(
        self,
        signal_variance: float = 1.0,
        length_scale: float = 1.0,
        noise_variance: float = 0.1,
        optimize: bool = True,
        n_opt_iter: int = 100,
        opt_lr: float = 0.01,
    ):
        self.signal_variance = signal_variance
        self.length_scale = length_scale
        self.noise_variance = noise_variance
        self.optimize = optimize
        self.n_opt_iter = n_opt_iter
        self.opt_lr = opt_lr
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.K_inv: np.ndarray | None = None
        self.alpha_: np.ndarray | None = None
        self.opt_history: list[dict] = []
        self.fit_time: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "GaussianProcessRegression":
        t0 = time.perf_counter()
        self.X_train = X.copy()
        self.y_train = y.copy()

        if self.optimize:
            self._optimize_hyperparams()

        K = self._rbf_kernel(X, X) + self.noise_variance * np.eye(len(X))
        self.K_inv = np.linalg.inv(K)
        self.alpha_ = self.K_inv @ y
        self.fit_time = time.perf_counter() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        K_star = self._rbf_kernel(X, self.X_train)
        return K_star @ self.alpha_

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        K_star = self._rbf_kernel(X, self.X_train)
        K_ss = self._rbf_kernel(X, X)

        y_mean = K_star @ self.alpha_
        y_var = np.diag(K_ss) - np.sum((K_star @ self.K_inv) * K_star, axis=1)
        y_std = np.sqrt(np.maximum(y_var, 0))
        return y_mean, y_std

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        sq_dist = (
            np.sum(X1 ** 2, axis=1, keepdims=True)
            + np.sum(X2 ** 2, axis=1, keepdims=True).T
            - 2 * X1 @ X2.T
        )
        return self.signal_variance * np.exp(-sq_dist / (2 * self.length_scale ** 2))

    def _log_marginal_likelihood(self, X, y) -> float:
        K = self._rbf_kernel(X, X) + self.noise_variance * np.eye(len(X))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -1e10
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        lml = (
            -0.5 * y @ alpha
            - np.sum(np.log(np.diag(L)))
            - 0.5 * len(y) * np.log(2 * np.pi)
        )
        return float(lml)

    def _optimize_hyperparams(self):
        """Gradient ascent on log-marginal-likelihood (numerical gradients)."""
        X, y = self.X_train, self.y_train
        params = np.log(np.array([
            self.signal_variance, self.length_scale, self.noise_variance
        ]))

        eps = 1e-5
        for i in range(self.n_opt_iter):
            self.signal_variance = np.exp(params[0])
            self.length_scale = np.exp(params[1])
            self.noise_variance = np.exp(params[2])

            lml = self._log_marginal_likelihood(X, y)

            # Numerical gradient
            grad = np.zeros(3)
            for j in range(3):
                params_plus = params.copy()
                params_plus[j] += eps
                self.signal_variance = np.exp(params_plus[0])
                self.length_scale = np.exp(params_plus[1])
                self.noise_variance = np.exp(params_plus[2])
                lml_plus = self._log_marginal_likelihood(X, y)
                grad[j] = (lml_plus - lml) / eps

            params += self.opt_lr * grad

            self.opt_history.append({
                "iter": i,
                "lml": lml,
                "signal_var": np.exp(params[0]),
                "length_scale": np.exp(params[1]),
                "noise_var": np.exp(params[2]),
            })

        # Set final params
        self.signal_variance = np.exp(params[0])
        self.length_scale = np.exp(params[1])
        self.noise_variance = np.exp(params[2])


# ============================================================================
# 5. Robust Regression (IRLS with Huber Loss)
# ============================================================================

class RobustRegression:
    """Iteratively Reweighted Least Squares (IRLS) with Huber loss.

    Huber loss weights: w_i = 1 if |r_i| ≤ δ, else δ / |r_i|
    This down-weights outliers, making the regression more robust.
    """

    def __init__(
        self,
        delta: float = 1.345,  # Huber threshold (1.345 → 95% efficiency at Gaussian)
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        self.delta = delta
        self.max_iter = max_iter
        self.tol = tol
        self.w: np.ndarray | None = None
        self.n_iter: int = 0
        self.weight_history: list[np.ndarray] = []
        self.fit_time: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "RobustRegression":
        t0 = time.perf_counter()
        X_b = _add_bias(X)
        n, p = X_b.shape

        # Initialise with OLS
        self.w = np.linalg.solve(X_b.T @ X_b, X_b.T @ y)
        weights = np.ones(n)

        for iteration in range(self.max_iter):
            w_old = self.w.copy()

            # Compute residuals and Huber weights
            residuals = y - X_b @ self.w
            abs_res = np.abs(residuals)
            weights = np.where(abs_res <= self.delta, 1.0, self.delta / abs_res)
            self.weight_history.append(weights.copy())

            # Weighted least squares step
            W = np.diag(weights)
            self.w = np.linalg.solve(X_b.T @ W @ X_b, X_b.T @ W @ y)

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
# 6. Bias-Variance Bootstrap Analysis
# ============================================================================

def bias_variance_bootstrap(
    model_class,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bootstrap: int = 200,
    seed: int = 42,
    model_kwargs: dict | None = None,
    fit_kwargs: dict | None = None,
) -> dict:
    """Estimate bias² and variance via bootstrap.

    For each bootstrap sample, train a model and collect predictions
    on X_test.  Then:
        - bias² = mean((y_test - mean_prediction)²)
        - variance = mean(var_of_predictions_per_sample)
        - total_error ≈ bias² + variance + noise
    """
    model_kwargs = model_kwargs or {}
    fit_kwargs = fit_kwargs or {}
    rng = np.random.default_rng(seed)
    n_train = len(X_train)
    n_test = len(X_test)

    # Collect predictions: shape (n_bootstrap, n_test)
    predictions = np.zeros((n_bootstrap, n_test))

    for b in range(n_bootstrap):
        idx = rng.choice(n_train, size=n_train, replace=True)
        X_b, y_b = X_train[idx], y_train[idx]

        model = model_class(**model_kwargs)
        model.fit(X_b, y_b, **fit_kwargs)
        predictions[b] = model.predict(X_test)

    # Mean prediction across bootstraps for each test point
    mean_pred = predictions.mean(axis=0)  # (n_test,)

    # Bias² = E[(y - E[f(x)])²]
    bias_sq = float(np.mean((y_test - mean_pred) ** 2))

    # Variance = E[Var[f(x)]]
    variance = float(np.mean(predictions.var(axis=0)))

    # Total expected error
    total_error = bias_sq + variance

    print(f"\n  Bias-Variance Decomposition ({n_bootstrap} bootstrap samples)")
    print(f"  {'─' * 40}")
    print(f"  Bias²    : {bias_sq:.6f}")
    print(f"  Variance : {variance:.6f}")
    print(f"  Total    : {total_error:.6f}")

    return {
        "bias_squared": bias_sq,
        "variance": variance,
        "total_error": total_error,
        "mean_prediction": mean_pred,
        "all_predictions": predictions,
    }


# ============================================================================
# Helpers
# ============================================================================

def _add_bias(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0]), X])
