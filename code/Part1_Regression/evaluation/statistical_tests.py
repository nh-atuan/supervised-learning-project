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


# ============================================================================
# Friedman Test (Omnibus Non-Parametric Test)
# ============================================================================

def friedman_test(
    cv_results_dict: dict[str, dict],
    metric: str = "R²",
    alpha: float = 0.05,
) -> dict:
    """Friedman test — non-parametric omnibus test across *all* models.

    The Friedman test is the non-parametric equivalent of a one-way
    repeated-measures ANOVA.  It tests the null hypothesis:

        H0: All models perform equally (no overall difference).
        H1: At least one model is significantly different.

    This should be run **before** conducting pairwise comparisons
    (e.g. Wilcoxon) to control the family-wise error rate.  If the
    Friedman test is *not* significant, pairwise comparisons should
    be interpreted with caution.

    Parameters
    ----------
    cv_results_dict : dict
        Keys = model names, values = output of ``cross_validate_model()``.
        Each value must contain ``metric`` as a key, mapping to a dict
        with a ``"folds"`` list.  All fold lists must have the same length.
    metric : str
        Which metric to test (e.g. "MSE", "R²").
    alpha : float
        Significance level.

    Returns
    -------
    dict with keys:
        test, metric, n_models, n_folds, statistic, p_value, significant,
        alpha, model_names, ranks (average ranks per model).
    """
    model_names = list(cv_results_dict.keys())
    n_models = len(model_names)

    if n_models < 3:
        print("  [Friedman] Need at least 3 models. Skipping.")
        return {
            "test": "Friedman",
            "metric": metric,
            "n_models": n_models,
            "n_folds": 0,
            "statistic": float("nan"),
            "p_value": 1.0,
            "significant": False,
            "alpha": alpha,
            "model_names": model_names,
            "ranks": {},
            "note": "Friedman test requires >= 3 groups.",
        }

    # Build matrix: rows = folds, columns = models
    folds_per_model = [
        cv_results_dict[name][metric]["folds"] for name in model_names
    ]
    n_folds = len(folds_per_model[0])
    assert all(len(f) == n_folds for f in folds_per_model), \
        "All models must have the same number of folds."

    # scipy.stats.friedmanchisquare expects each group as a separate array
    stat, p_value = stats.friedmanchisquare(*folds_per_model)

    # Compute average ranks across folds (lower rank = better for error
    # metrics, higher rank = better for R²-like metrics)
    from scipy.stats import rankdata as _rankdata

    score_matrix = np.array(folds_per_model).T  # shape (n_folds, n_models)
    # For error metrics (lower is better), rank ascending.
    # For R²-like metrics (higher is better), rank descending.
    is_higher_better = metric.startswith("R") or "R²" in metric
    if is_higher_better:
        # Negate so that highest value gets rank 1
        rank_matrix = np.apply_along_axis(
            lambda row: _rankdata(-row, method="average"), axis=1, arr=score_matrix,
        )
    else:
        rank_matrix = np.apply_along_axis(
            lambda row: _rankdata(row, method="average"), axis=1, arr=score_matrix,
        )
    avg_ranks = rank_matrix.mean(axis=0)  # shape (n_models,)
    rank_dict = {name: float(r) for name, r in zip(model_names, avg_ranks)}

    significant = p_value < alpha

    result = {
        "test": "Friedman",
        "metric": metric,
        "n_models": n_models,
        "n_folds": n_folds,
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": significant,
        "alpha": alpha,
        "model_names": model_names,
        "ranks": rank_dict,
    }

    # Pretty-print
    print(f"\n{'='*65}")
    print(f"  Friedman Test — {metric}  ({n_models} models × {n_folds} folds)")
    print(f"{'='*65}")
    print(f"  Statistic (χ²) : {stat:.4f}")
    print(f"  p-value         : {p_value:.6f}")
    sig_label = "YES — reject H0 (models differ)" if significant \
        else "NO — fail to reject H0 (no evidence of difference)"
    print(f"  Significant (α={alpha}): {sig_label}")
    print(f"\n  Average Ranks (1 = best):")
    for name, rank in sorted(rank_dict.items(), key=lambda x: x[1]):
        print(f"    {name:<25s}  rank = {rank:.2f}")
    print(f"{'='*65}\n")

    return result


def pairwise_wilcoxon_posthoc(
    cv_results_dict: dict[str, dict],
    metric: str = "R²",
    alpha: float = 0.05,
    correction: str = "bonferroni",
) -> list[dict]:
    """Post-hoc pairwise Wilcoxon tests with multiple-comparison correction.

    Should only be run **after** a significant Friedman test to identify
    *which* specific pairs of models differ significantly.

    Parameters
    ----------
    cv_results_dict : dict
        Same format as for ``friedman_test()``.
    metric : str
        Which metric to test.
    alpha : float
        Family-wise significance level.
    correction : str
        ``"bonferroni"`` divides α by the number of comparisons.

    Returns
    -------
    list of dicts, one per pair, with keys:
        model_a, model_b, metric, statistic, p_value, p_adjusted,
        significant, alpha_adjusted.
    """
    model_names = list(cv_results_dict.keys())
    n_models = len(model_names)
    n_comparisons = n_models * (n_models - 1) // 2

    if correction == "bonferroni":
        alpha_adj = alpha / n_comparisons
    else:
        alpha_adj = alpha  # no correction

    results = []
    print(f"\n{'─'*65}")
    print(f"  Pairwise Wilcoxon Post-hoc — {metric}")
    print(f"  Correction: {correction} (α_adj = {alpha_adj:.6f}, "
          f"{n_comparisons} comparisons)")
    print(f"{'─'*65}")

    for i in range(n_models):
        for j in range(i + 1, n_models):
            name_a, name_b = model_names[i], model_names[j]
            folds_a = np.asarray(cv_results_dict[name_a][metric]["folds"])
            folds_b = np.asarray(cv_results_dict[name_b][metric]["folds"])

            diff = folds_a - folds_b
            nonzero = np.count_nonzero(diff)
            if nonzero < 2:
                stat, p_val = float("nan"), 1.0
            else:
                stat, p_val = stats.wilcoxon(folds_a, folds_b,
                                             alternative="two-sided")
                stat, p_val = float(stat), float(p_val)

            # Bonferroni-adjusted p-value (capped at 1.0)
            p_adjusted = min(p_val * n_comparisons, 1.0) \
                if correction == "bonferroni" else p_val

            significant = p_adjusted < alpha

            row = {
                "model_a": name_a,
                "model_b": name_b,
                "metric": metric,
                "test": "Wilcoxon (post-hoc)",
                "statistic": stat,
                "p_value": p_val,
                "p_adjusted": p_adjusted,
                "significant": significant,
                "alpha_adjusted": alpha_adj,
            }
            results.append(row)

            sig_mark = "***" if significant else "ns"
            print(f"  {name_a:<20s} vs {name_b:<20s}  "
                  f"p={p_val:.4f}  p_adj={p_adjusted:.4f}  {sig_mark}")

    print(f"{'─'*65}\n")
    return results


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
