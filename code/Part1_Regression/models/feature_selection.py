"""
Feature selection methods for regression.

Functions
---------
- forward_stepwise_selection  : Greedy forward selection
- backward_elimination        : Greedy backward elimination
- lasso_feature_selection     : Select features with non-zero Lasso coefficients
"""

from __future__ import annotations

import numpy as np


def forward_stepwise_selection(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_features: int | None = None,
    verbose: bool = True,
) -> dict:
    """Forward Stepwise Selection.

    Greedily adds the feature that most reduces validation MSE at each step.

    Returns
    -------
    dict with keys:
        selected_features : list[int]  — indices of selected features
        val_mses          : list[float] — validation MSE at each step
        feature_order     : list[int]   — order features were added
    """
    from ..evaluation.metrics import mse
    from .linear_regression import NormalEquationLR

    n_features = X.shape[1]
    if max_features is None:
        max_features = n_features

    selected: list[int] = []
    remaining = list(range(n_features))
    val_mses: list[float] = []
    feature_order: list[int] = []

    best_overall_mse = float("inf")
    best_overall_selected: list[int] = []

    for step in range(min(max_features, n_features)):
        best_mse = float("inf")
        best_feature = -1

        for f in remaining:
            current = selected + [f]
            model = NormalEquationLR()
            model.fit(X[:, current], y)
            score = mse(y_val, model.predict(X_val[:, current]))

            if score < best_mse:
                best_mse = score
                best_feature = f

        selected.append(best_feature)
        remaining.remove(best_feature)
        val_mses.append(best_mse)
        feature_order.append(best_feature)

        if best_mse < best_overall_mse:
            best_overall_mse = best_mse
            best_overall_selected = selected.copy()

        if verbose:
            print(f"  Step {step + 1:3d} | Added feature {best_feature:3d} | Val MSE: {best_mse:.6f}")

    return {
        "selected_features": best_overall_selected,
        "val_mses": val_mses,
        "feature_order": feature_order,
        "best_val_mse": best_overall_mse,
    }


def backward_elimination(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    min_features: int = 1,
    verbose: bool = True,
) -> dict:
    """Backward Elimination.

    Starts with all features, greedily removes the one whose removal
    causes the least increase in validation MSE.

    Returns
    -------
    dict with keys:
        selected_features : list[int]
        val_mses          : list[float]
        removal_order     : list[int]
    """
    from ..evaluation.metrics import mse
    from .linear_regression import NormalEquationLR

    n_features = X.shape[1]
    current_features = list(range(n_features))
    val_mses: list[float] = []
    removal_order: list[int] = []

    # Initial MSE with all features
    model = NormalEquationLR()
    model.fit(X[:, current_features], y)
    initial_mse = mse(y_val, model.predict(X_val[:, current_features]))
    val_mses.append(initial_mse)

    best_overall_mse = initial_mse
    best_overall_selected = current_features.copy()

    if verbose:
        print(f"  Start   | {n_features} features | Val MSE: {initial_mse:.6f}")

    while len(current_features) > min_features:
        best_mse = float("inf")
        worst_feature = -1

        for f in current_features:
            reduced = [c for c in current_features if c != f]
            model = NormalEquationLR()
            model.fit(X[:, reduced], y)
            score = mse(y_val, model.predict(X_val[:, reduced]))

            if score < best_mse:
                best_mse = score
                worst_feature = f

        current_features.remove(worst_feature)
        removal_order.append(worst_feature)
        val_mses.append(best_mse)

        if best_mse < best_overall_mse:
            best_overall_mse = best_mse
            best_overall_selected = current_features.copy()

        if verbose:
            print(f"  Step {len(removal_order):3d} | Removed feature {worst_feature:3d} "
                  f"| {len(current_features)} remaining | Val MSE: {best_mse:.6f}")

    return {
        "selected_features": best_overall_selected,
        "val_mses": val_mses,
        "removal_order": removal_order,
        "best_val_mse": best_overall_mse,
    }


def lasso_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    threshold: float = 1e-10,
    verbose: bool = True,
) -> dict:
    """Select features based on non-zero Lasso coefficients.

    Parameters
    ----------
    lam : float
        Lasso regularization strength.
    threshold : float
        Coefficients with |w| > threshold are considered non-zero.

    Returns
    -------
    dict with keys:
        selected_features : list[int]
        coefficients      : np.ndarray
        n_selected        : int
    """
    from .regularized_regression import LassoRegression

    model = LassoRegression(lam=lam, max_iter=2000)
    model.fit(X, y)
    coefs = model.coefficients

    selected = [i for i in range(len(coefs)) if abs(coefs[i]) > threshold]

    if verbose:
        print(f"\n  Lasso Feature Selection (λ = {lam})")
        print(f"  Non-zero coefficients: {len(selected)} / {len(coefs)}")
        if len(selected) <= 20:
            for idx in selected:
                print(f"    Feature {idx:3d}: {coefs[idx]:+.6f}")

    return {
        "selected_features": selected,
        "coefficients": coefs,
        "n_selected": len(selected),
    }
