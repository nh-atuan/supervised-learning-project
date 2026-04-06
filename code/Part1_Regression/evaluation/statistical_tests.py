"""
Statistical tests for comparing regression models.

Implements paired t-test and Wilcoxon signed-rank test to determine
whether performance differences between models are statistically significant.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def paired_t_test(
    scores_a: list[float] | np.ndarray,
    scores_b: list[float] | np.ndarray,
    alpha: float = 0.05,
    metric_name: str = "metric",
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> dict:
    """Two-sided paired t-test on per-fold metric scores.

    Parameters
    ----------
    scores_a, scores_b : array-like of shape (k,)
        Per-fold metric values for model A and model B.
    alpha : float
        Significance level.

    Returns
    -------
    dict with keys: statistic, p_value, significant, mean_diff, ci_lower, ci_upper
    """
    a, b = np.asarray(scores_a), np.asarray(scores_b)
    assert len(a) == len(b), "Both score arrays must have the same length (same k)."

    diff = a - b
    n = len(diff)
    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1))
    se = std_diff / np.sqrt(n)

    t_stat = mean_diff / se if se > 0 else 0.0
    p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1))  # two-sided

    # Confidence interval for mean difference
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    significant = p_value < alpha

    result = {
        "test": "Paired t-test",
        "metric": metric_name,
        "model_a": model_a_name,
        "model_b": model_b_name,
        "statistic": float(t_stat),
        "p_value": p_value,
        "significant": significant,
        "alpha": alpha,
        "mean_diff": mean_diff,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }

    _print_result(result)
    return result


def wilcoxon_test(
    scores_a: list[float] | np.ndarray,
    scores_b: list[float] | np.ndarray,
    alpha: float = 0.05,
    metric_name: str = "metric",
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> dict:
    """Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    Parameters
    ----------
    scores_a, scores_b : array-like of shape (k,)
        Per-fold metric values for model A and model B.
    alpha : float
        Significance level.

    Returns
    -------
    dict with keys: statistic, p_value, significant
    """
    a, b = np.asarray(scores_a), np.asarray(scores_b)
    assert len(a) == len(b), "Both score arrays must have the same length."

    diff = a - b
    # Wilcoxon requires non-zero differences
    nonzero_mask = diff != 0
    if nonzero_mask.sum() < 2:
        result = {
            "test": "Wilcoxon signed-rank",
            "metric": metric_name,
            "model_a": model_a_name,
            "model_b": model_b_name,
            "statistic": float("nan"),
            "p_value": 1.0,
            "significant": False,
            "alpha": alpha,
            "note": "Not enough non-zero differences for Wilcoxon test.",
        }
        _print_result(result)
        return result

    stat, p_value = stats.wilcoxon(a, b, alternative="two-sided")

    significant = p_value < alpha

    result = {
        "test": "Wilcoxon signed-rank",
        "metric": metric_name,
        "model_a": model_a_name,
        "model_b": model_b_name,
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": significant,
        "alpha": alpha,
    }

    _print_result(result)
    return result


def compare_models(
    cv_results_a: dict,
    cv_results_b: dict,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    metrics: list[str] | None = None,
    alpha: float = 0.05,
) -> list[dict]:
    """Run both statistical tests for multiple metrics.

    Parameters
    ----------
    cv_results_a, cv_results_b : dict
        Output of ``cross_validate_model()`` from metrics.py.
        Each has keys like "MSE", "RMSE", etc. mapping to
        {"mean", "std", "folds"}.
    metrics : list of str or None
        Which metrics to compare.  Defaults to all shared keys.

    Returns
    -------
    list of test result dicts
    """
    if metrics is None:
        metrics = list(set(cv_results_a.keys()) & set(cv_results_b.keys()))

    all_results = []
    for metric in metrics:
        folds_a = cv_results_a[metric]["folds"]
        folds_b = cv_results_b[metric]["folds"]

        print(f"\n{'─'*60}")
        print(f"  Comparing {model_a_name} vs {model_b_name} on {metric}")
        print(f"{'─'*60}")

        t_result = paired_t_test(
            folds_a, folds_b, alpha=alpha,
            metric_name=metric,
            model_a_name=model_a_name,
            model_b_name=model_b_name,
        )
        w_result = wilcoxon_test(
            folds_a, folds_b, alpha=alpha,
            metric_name=metric,
            model_a_name=model_a_name,
            model_b_name=model_b_name,
        )
        all_results.extend([t_result, w_result])

    return all_results


def _print_result(result: dict) -> None:
    """Pretty-print a single test result."""
    sig_label = "YES ✓" if result.get("significant") else "NO"
    print(f"  [{result['test']}]  {result['model_a']} vs {result['model_b']}")
    print(f"    Metric    : {result['metric']}")
    print(f"    Statistic : {result['statistic']:.4f}")
    print(f"    p-value   : {result['p_value']:.6f}")
    print(f"    Significant (α={result['alpha']}): {sig_label}")
    if "mean_diff" in result:
        print(f"    Mean diff : {result['mean_diff']:.6f}")
        print(f"    95% CI    : [{result['ci_lower']:.6f}, {result['ci_upper']:.6f}]")
    if "note" in result:
        print(f"    Note      : {result['note']}")
