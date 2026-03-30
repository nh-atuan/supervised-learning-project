"""
Visualisation helpers for regression evaluation.

Every function produces publication-ready plots with proper titles, axis labels,
legends, and appropriate figure sizes so they can go directly into the report.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------
_FIG_DEFAULTS = dict(dpi=120)

sns.set_theme(style="whitegrid", font_scale=1.1, rc={
    "figure.figsize": (8, 5),
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


# ============================================================================
# 1. Learning curves
# ============================================================================

def plot_learning_curves(
    model_class,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric_fn=None,
    fractions: Optional[list[float]] = None,
    model_kwargs: dict | None = None,
    fit_kwargs: dict | None = None,
    title: str = "Learning Curve",
    ylabel: str = "MSE",
    ax: Optional[plt.Axes] = None,
    seed: int = 42,
) -> plt.Axes:
    """Plot train / validation loss as a function of training-set size.

    Parameters
    ----------
    model_class : class implementing fit(X, y) and predict(X)
    metric_fn : callable(y_true, y_pred) -> float.  Defaults to MSE.
    fractions : list of floats in (0, 1] — fractions of training data to use.
    """
    from .metrics import mse as _mse

    if metric_fn is None:
        metric_fn = _mse

    model_kwargs = model_kwargs or {}
    fit_kwargs = fit_kwargs or {}

    if fractions is None:
        fractions = np.linspace(0.1, 1.0, 10).tolist()

    n_total = len(X_train)
    train_scores, val_scores, sizes = [], [], []

    rng = np.random.default_rng(seed)

    for frac in fractions:
        n = max(10, int(n_total * frac))
        idx = rng.choice(n_total, size=n, replace=False)
        X_tr, y_tr = X_train[idx], y_train[idx]

        model = model_class(**model_kwargs)
        model.fit(X_tr, y_tr, **fit_kwargs)

        train_scores.append(metric_fn(y_tr, model.predict(X_tr)))
        val_scores.append(metric_fn(y_val, model.predict(X_val)))
        sizes.append(n)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5), **_FIG_DEFAULTS)
    ax.plot(sizes, train_scores, "o-", label="Train")
    ax.plot(sizes, val_scores, "s-", label="Validation")
    ax.set_xlabel("Number of Training Samples")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax


# ============================================================================
# 2. Residuals
# ============================================================================

def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Analysis",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Scatter plot of residuals + histogram side panel."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), **_FIG_DEFAULTS)

    # Scatter
    axes[0].scatter(y_pred, residuals, alpha=0.4, s=12, edgecolors="none")
    axes[0].axhline(y=0, color="red", linewidth=1.2, linestyle="--")
    axes[0].set_xlabel("Predicted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title(f"{title} — Residuals vs Predicted")
    axes[0].grid(True, alpha=0.3)

    # Histogram
    axes[1].hist(residuals, bins=40, edgecolor="white", alpha=0.8, color="steelblue")
    axes[1].set_xlabel("Residual Value")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"{title} — Residual Distribution")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================================
# 3. Predicted vs Actual
# ============================================================================

def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Scatter plot of y_pred vs y_true with 45° reference line."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7), **_FIG_DEFAULTS)

    ax.scatter(y_true, y_pred, alpha=0.35, s=12, edgecolors="none", label="Predictions")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "r--", linewidth=1.2, label="Ideal (y = x)")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax


# ============================================================================
# 4. QQ-Plot
# ============================================================================

def plot_qq(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "QQ-Plot of Residuals",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Quantile-Quantile plot for residual normality check."""
    residuals = y_true - y_pred
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), **_FIG_DEFAULTS)
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax


# ============================================================================
# 5. Regularization path
# ============================================================================

def plot_regularization_path(
    lambdas: np.ndarray,
    coefs: np.ndarray,
    feature_names: Optional[list[str]] = None,
    title: str = "Regularization Path",
    top_k: int = 10,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot coefficient values as a function of log(λ).

    Parameters
    ----------
    lambdas : 1-D array of regularization strengths
    coefs : 2-D array, shape (len(lambdas), n_features)
    top_k : int — only label the top-k features with largest magnitude at
             the smallest λ for readability.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6), **_FIG_DEFAULTS)

    log_lambdas = np.log10(lambdas)

    # Determine top-k features for labelling
    last_coef = np.abs(coefs[0])  # smallest lambda => most coefficients
    top_indices = np.argsort(last_coef)[-top_k:]

    for j in range(coefs.shape[1]):
        label = None
        lw = 0.7
        alpha = 0.4
        if j in top_indices and feature_names:
            label = feature_names[j]
            lw = 1.5
            alpha = 0.9
        ax.plot(log_lambdas, coefs[:, j], linewidth=lw, alpha=alpha, label=label)

    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("log₁₀(λ)")
    ax.set_ylabel("Coefficient Value")
    ax.set_title(title)
    if feature_names:
        ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax


# ============================================================================
# 6. Validation curve
# ============================================================================

def plot_validation_curve(
    complexities: list,
    train_scores: list[float],
    val_scores: list[float],
    xlabel: str = "Model Complexity",
    ylabel: str = "MSE",
    title: str = "Validation Curve",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot train/val metric vs model complexity (e.g. polynomial degree)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5), **_FIG_DEFAULTS)

    ax.plot(complexities, train_scores, "o-", label="Train")
    ax.plot(complexities, val_scores, "s-", label="Validation")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax


# ============================================================================
# 7. Convergence plot (for gradient descent)
# ============================================================================

def plot_convergence(
    losses: list[float] | dict[str, list[float]],
    xlabel: str = "Epoch",
    ylabel: str = "Loss (MSE)",
    title: str = "Convergence Plot",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot training loss over epochs.

    Parameters
    ----------
    losses : list or dict
        If list, single curve.  If dict, multiple curves {label: list}.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5), **_FIG_DEFAULTS)

    if isinstance(losses, dict):
        for label, vals in losses.items():
            ax.plot(vals, label=label)
    else:
        ax.plot(losses, label="Training Loss")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax


# ============================================================================
# 8. Model comparison bar chart
# ============================================================================

def plot_model_comparison(
    results_df,
    metric: str = "R²",
    title: str = "Model Comparison",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Horizontal bar chart comparing models on a single metric.

    Parameters
    ----------
    results_df : pd.DataFrame with 'Model' column and metric columns.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, max(4, len(results_df) * 0.5)), **_FIG_DEFAULTS)

    sorted_df = results_df.sort_values(metric, ascending=True)
    colors = sns.color_palette("viridis", n_colors=len(sorted_df))
    ax.barh(sorted_df["Model"], sorted_df[metric], color=colors)
    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    return ax


# ============================================================================
# 9. Bias-Variance trade-off plot
# ============================================================================

def plot_bias_variance(
    complexities,
    bias_sq: list[float],
    variance: list[float],
    mse_total: Optional[list[float]] = None,
    xlabel: str = "Model Complexity",
    title: str = "Bias–Variance Trade-off",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot bias², variance, and optionally total MSE vs complexity."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5), **_FIG_DEFAULTS)

    ax.plot(complexities, bias_sq, "o-", label="Bias²", color="tab:blue")
    ax.plot(complexities, variance, "s-", label="Variance", color="tab:orange")
    if mse_total is not None:
        ax.plot(complexities, mse_total, "^-", label="Total MSE", color="tab:red")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Error")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax


# ============================================================================
# 10. Heatmap helper (for Elastic Net lambda grid, etc.)
# ============================================================================

def plot_heatmap(
    data: np.ndarray,
    x_labels: list,
    y_labels: list,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "Heatmap",
    cmap: str = "viridis",
    fmt: str = ".4f",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Generic annotated heatmap using seaborn."""
    if ax is None:
        _, ax = plt.subplots(
            figsize=(max(6, len(x_labels) * 0.8), max(5, len(y_labels) * 0.6)),
            **_FIG_DEFAULTS,
        )
    sns.heatmap(
        data, annot=True, fmt=fmt, cmap=cmap,
        xticklabels=x_labels, yticklabels=y_labels, ax=ax,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    return ax
