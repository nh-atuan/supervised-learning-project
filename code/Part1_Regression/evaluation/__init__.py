"""Part 1 – Regression evaluation package.

Modules
-------
metrics : Core evaluation metrics (MSE, RMSE, MAE, R², MedAE, Adjusted R²)
          and k-fold cross-validation utilities.
plots : Publication-ready visualisation helpers (learning curves, residuals,
        predicted vs actual, QQ-plot, boxplots, violin plots, jointplots, etc.).
statistical_tests : Paired t-test, Wilcoxon signed-rank test, Friedman omnibus
                    test, and pairwise post-hoc testing with Bonferroni correction.
"""

from .metrics import (
    mse,
    rmse,
    mae,
    r_squared,
    medae,
    adjusted_r_squared,
    compute_all_metrics,
    print_metrics,
    k_fold_split,
    cross_validate_model,
    print_cv_results,
    verify_against_sklearn,
)

from .plots import (
    plot_learning_curves,
    plot_residuals,
    plot_predicted_vs_actual,
    plot_qq,
    plot_regularization_path,
    plot_validation_curve,
    plot_convergence,
    plot_model_comparison,
    plot_bias_variance,
    plot_heatmap,
    plot_cv_boxplot,
    plot_multi_residuals,
    plot_residual_violin,
    plot_predicted_vs_actual_joint,
)

from .statistical_tests import (
    paired_t_test,
    wilcoxon_test,
    compare_models,
    friedman_test,
    pairwise_wilcoxon_posthoc,
)

__all__ = [
    # --- metrics ---
    "mse", "rmse", "mae", "r_squared", "medae", "adjusted_r_squared",
    "compute_all_metrics", "print_metrics",
    "k_fold_split", "cross_validate_model", "print_cv_results",
    "verify_against_sklearn",
    # --- plots ---
    "plot_learning_curves", "plot_residuals", "plot_predicted_vs_actual",
    "plot_qq", "plot_regularization_path", "plot_validation_curve",
    "plot_convergence", "plot_model_comparison", "plot_bias_variance",
    "plot_heatmap", "plot_cv_boxplot", "plot_multi_residuals",
    "plot_residual_violin", "plot_predicted_vs_actual_joint",
    # --- statistical tests ---
    "paired_t_test", "wilcoxon_test", "compare_models",
    "friedman_test", "pairwise_wilcoxon_posthoc",
]
