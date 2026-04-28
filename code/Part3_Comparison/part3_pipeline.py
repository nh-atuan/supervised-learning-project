from __future__ import annotations

import json
import sys
import time
import tracemalloc
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SkLDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as SkQDA
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression as SkLogReg
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def resolve_repo_root() -> Path:
    candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    for candidate in candidates:
        if (candidate / "docs" / "REQUIREMENT.md").exists():
            return candidate
    raise FileNotFoundError("Cannot locate repository root.")


def _load_part1_modules(repo_root: Path):
    part1_root = repo_root / "code" / "Part1_Regression"
    if str(part1_root) not in sys.path:
        sys.path.insert(0, str(part1_root))

    from eda_preprocessing.load_data import load_regression_data  # type: ignore[import-not-found]
    from models.linear_regression import MiniBatchGDLR, NormalEquationLR  # type: ignore[import-not-found]
    from models.nonlinear_models import PolynomialRegression  # type: ignore[import-not-found]
    from models.regularized_regression import LassoRegression, RidgeRegression  # type: ignore[import-not-found]

    return {
        "load_regression_data": load_regression_data,
        "MiniBatchGDLR": MiniBatchGDLR,
        "NormalEquationLR": NormalEquationLR,
        "PolynomialRegression": PolynomialRegression,
        "LassoRegression": LassoRegression,
        "RidgeRegression": RidgeRegression,
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse_val = mean_squared_error(y_true, y_pred)
    return {
        "MSE_test": float(mse_val),
        "RMSE_test": float(np.sqrt(mse_val)),
        "MAE_test": float(mean_absolute_error(y_true, y_pred)),
        "R2_test": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    return {
        "Model": model_name,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _fit_predict_profile(model, X_train, y_train, X_test):
    tracemalloc.start()
    fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - fit_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    pred_start = time.perf_counter()
    y_pred = model.predict(X_test)
    pred_time = time.perf_counter() - pred_start

    return y_pred, float(fit_time), float(pred_time), float(peak / (1024**2))


def _add_gaussian_noise(X: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    if sigma <= 0:
        return X.copy()
    rng = np.random.default_rng(seed)
    return X + rng.normal(loc=0.0, scale=sigma, size=X.shape)


def _corrupt_features(X: np.ndarray, corruption_rate: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X_corrupt = X.copy().astype(float)
    mask = rng.random(X_corrupt.shape) < corruption_rate
    X_corrupt[mask] = np.nan
    return X_corrupt


def _impute_with_strategy(X_train: np.ndarray, X_test_corrupt: np.ndarray, strategy: str) -> np.ndarray:
    if strategy == "knn":
        imputer = KNNImputer(n_neighbors=5)
    else:
        imputer = SimpleImputer(strategy=strategy)
    imputer.fit(X_train)
    return imputer.transform(X_test_corrupt)


def _to_native(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def run_all(seed: int = 42, max_class_samples: int = 30000) -> dict:
    np.random.seed(seed)
    repo_root = resolve_repo_root()
    modules = _load_part1_modules(repo_root)

    part2_results = repo_root / "code" / "Part2_Classification" / "results"
    part3_results = repo_root / "code" / "Part3_Comparison" / "results"
    part3_results.mkdir(parents=True, exist_ok=True)

    load_regression_data = modules["load_regression_data"]
    MiniBatchGDLR = modules["MiniBatchGDLR"]
    NormalEquationLR = modules["NormalEquationLR"]
    PolynomialRegression = modules["PolynomialRegression"]
    LassoRegression = modules["LassoRegression"]
    RidgeRegression = modules["RidgeRegression"]

    # --- Load baseline artifacts ---
    reg_data = load_regression_data(use_log_target=True)
    X_reg_all = np.vstack([reg_data["X_train"], reg_data["X_val"], reg_data["X_test"]])
    y_reg_all = np.concatenate([reg_data["y_train"], reg_data["y_val"], reg_data["y_test"]])

    reg_prediction_files = [
        repo_root / "code" / "Part1_Regression" / "models" / "results" / "predictions_03_linear_regression.joblib",
        repo_root / "code" / "Part1_Regression" / "models" / "results" / "predictions_04_regularized_regression.joblib",
        repo_root / "code" / "Part1_Regression" / "models" / "results" / "predictions_05_nonlinear_models.joblib",
    ]

    reg_predictions = {}
    for file_path in reg_prediction_files:
        reg_predictions.update(joblib.load(file_path))

    reg_rows = []
    for model_name, payload in reg_predictions.items():
        m = regression_metrics(reg_data["y_test"], np.asarray(payload["y_pred_test"]))
        reg_rows.append({"Model": model_name, **m})
    reg_summary_df = pd.DataFrame(reg_rows).sort_values("R2_test", ascending=False).reset_index(drop=True)

    cls_artifact_path = part2_results / "evaluation_artifacts_from_03.joblib"
    if not cls_artifact_path.exists():
        raise FileNotFoundError(f"Missing artifact: {cls_artifact_path}")

    cls_artifacts = joblib.load(cls_artifact_path)
    X_cls_all = np.vstack([
        cls_artifacts["X_train_processed"],
        cls_artifacts["X_val_processed"],
        cls_artifacts["X_test_processed"],
    ])
    y_cls_all = np.concatenate([
        cls_artifacts["y_train"],
        cls_artifacts["y_val"],
        cls_artifacts["y_test"],
    ])

    cls_predictions = {}
    for name, pred in cls_artifacts.get("multiclass_test_predictions", {}).items():
        cls_predictions[f"Logistic {name}"] = np.asarray(pred)
    if "preds_lda" in cls_artifacts:
        cls_predictions["LDA"] = np.asarray(cls_artifacts["preds_lda"])
    if "preds_qda" in cls_artifacts:
        cls_predictions["QDA"] = np.asarray(cls_artifacts["preds_qda"])
    for name, pred in cls_artifacts.get("task3_custom_test_predictions", {}).items():
        cls_predictions[name] = np.asarray(pred)

    y_cls_test = np.asarray(cls_artifacts["y_test"])
    cls_rows = [classification_metrics(y_cls_test, pred, model_name) for model_name, pred in cls_predictions.items()]
    cls_summary_df = pd.DataFrame(cls_rows).sort_values("F1_macro", ascending=False).reset_index(drop=True)

    # --- Comprehensive comparison table ---
    best_linear = reg_summary_df[reg_summary_df["Model"].isin(["OLS (Normal Eq.)", "Mini-batch GD"])].iloc[0]
    best_ridge_lasso = reg_summary_df[reg_summary_df["Model"].isin(["Ridge", "Lasso", "ElasticNet"])].iloc[0]
    best_logistic = cls_summary_df[cls_summary_df["Model"].str.contains("Logistic", regex=False)].iloc[0]
    best_lda_qda = cls_summary_df[cls_summary_df["Model"].isin(["LDA", "QDA"])].iloc[0]

    comparison_table = pd.DataFrame(
        [
            {
                "Tiêu chí": "Giả thiết phân phối",
                "Linear Reg.": "Gaussian noise, E[y|X] tuyến tính",
                "Ridge/Lasso": "Như Linear Reg. + regularization prior",
                "Logistic Reg.": "Bernoulli/Categorical, logit/softmax link",
                "LDA/QDA": "Gaussian theo lớp; LDA chung covariance, QDA riêng covariance",
            },
            {
                "Tiêu chí": "Nghiệm dạng đóng",
                "Linear Reg.": "Có (Normal Equation)",
                "Ridge/Lasso": "Ridge: có; Lasso: không (coord-descent)",
                "Logistic Reg.": "Không (lặp: GD/Newton)",
                "LDA/QDA": "Có (ước lượng thống kê + Bayes rule)",
            },
            {
                "Tiêu chí": "Độ phức tạp huấn luyện",
                "Linear Reg.": "Trung bình (giải hệ tuyến tính)",
                "Ridge/Lasso": "Cao hơn Linear (CV + regularization)",
                "Logistic Reg.": "Trung bình đến cao (iterative optimization)",
                "LDA/QDA": "LDA nhẹ; QDA nặng hơn do covariance theo lớp",
            },
            {
                "Tiêu chí": "Khả năng giải thích",
                "Linear Reg.": "Rất cao",
                "Ridge/Lasso": "Cao (Lasso có tính chọn đặc trưng)",
                "Logistic Reg.": "Cao (hệ số theo log-odds)",
                "LDA/QDA": "Trung bình-cao (tham số phân phối theo lớp)",
            },
            {
                "Tiêu chí": "Nhạy với outlier",
                "Linear Reg.": "Cao (MSE rất nhạy)",
                "Ridge/Lasso": "Ridge/Lasso giảm một phần nhưng vẫn nhạy",
                "Logistic Reg.": "Nhạy với outlier leverage cao",
                "LDA/QDA": "Nhạy do phụ thuộc ước lượng mean/covariance",
            },
            {
                "Tiêu chí": "Hiệu năng thực nghiệm",
                "Linear Reg.": f"Best R2={best_linear['R2_test']:.4f} ({best_linear['Model']})",
                "Ridge/Lasso": f"Best R2={best_ridge_lasso['R2_test']:.4f} ({best_ridge_lasso['Model']})",
                "Logistic Reg.": f"Best F1-macro={best_logistic['F1_macro']:.4f} ({best_logistic['Model']})",
                "LDA/QDA": f"Best F1-macro={best_lda_qda['F1_macro']:.4f} ({best_lda_qda['Model']})",
            },
        ]
    )

    comparison_table_path = part3_results / "part3_comprehensive_comparison_table.csv"
    comparison_table.to_csv(comparison_table_path, index=False)

    # --- Model builders ---
    best_params_04 = joblib.load(repo_root / "code" / "Part1_Regression" / "models" / "results" / "best_params_04.joblib")
    best_params_05 = joblib.load(repo_root / "code" / "Part1_Regression" / "models" / "results" / "best_params_05.joblib")

    def make_regression_models():
        return {
            "OLS": NormalEquationLR(),
            "Ridge": RidgeRegression(lam=float(best_params_04["Ridge"]["lam"])),
            "Lasso": LassoRegression(lam=float(best_params_04["Lasso"]["lam"]), max_iter=800, tol=1e-5),
            "Polynomial": PolynomialRegression(degree=int(best_params_05["Polynomial"]["degree"]), include_interactions=False),
        }

    def make_classification_models():
        return {
            "LogisticReg": SkLogReg(max_iter=300, solver="lbfgs", random_state=seed),
            "LDA": SkLDA(solver="lsqr", shrinkage="auto"),
            "QDA": SkQDA(reg_param=0.1),
        }

    # --- Split sensitivity ---
    split_records = []
    split_indices_payload = {}
    ratios = [0.6, 0.7, 0.8]

    cls_idx_full = np.arange(len(y_cls_all))
    if len(cls_idx_full) > max_class_samples:
        cls_subset_idx, _ = train_test_split(
            cls_idx_full,
            train_size=max_class_samples,
            stratify=y_cls_all,
            random_state=seed,
        )
    else:
        cls_subset_idx = cls_idx_full

    X_cls_exp = X_cls_all[cls_subset_idx]
    y_cls_exp = y_cls_all[cls_subset_idx]
    split_indices_payload["cls_subset"] = cls_subset_idx

    for ratio in ratios:
        rs = seed + int(ratio * 100)

        # Regression split
        reg_idx = np.arange(len(y_reg_all))
        reg_train_idx, reg_test_idx = train_test_split(
            reg_idx,
            train_size=ratio,
            random_state=rs,
            shuffle=True,
        )
        split_indices_payload[f"reg_train_{int(ratio * 100)}"] = reg_train_idx
        split_indices_payload[f"reg_test_{int(ratio * 100)}"] = reg_test_idx

        X_reg_train, y_reg_train = X_reg_all[reg_train_idx], y_reg_all[reg_train_idx]
        X_reg_test, y_reg_test = X_reg_all[reg_test_idx], y_reg_all[reg_test_idx]

        for model_name, model in make_regression_models().items():
            y_pred, fit_t, pred_t, peak_mb = _fit_predict_profile(model, X_reg_train, y_reg_train, X_reg_test)
            m = regression_metrics(y_reg_test, y_pred)
            split_records.append(
                {
                    "dataset": "regression",
                    "model": model_name,
                    "experiment": "split_sensitivity",
                    "train_ratio": ratio,
                    "metric_primary": "R2_test",
                    "primary_score": m["R2_test"],
                    **m,
                    "fit_time_sec": fit_t,
                    "predict_time_sec": pred_t,
                    "peak_mem_mb": peak_mb,
                }
            )

        # Classification split
        cls_local_idx = np.arange(len(y_cls_exp))
        cls_train_local, cls_test_local = train_test_split(
            cls_local_idx,
            train_size=ratio,
            random_state=rs,
            stratify=y_cls_exp,
        )

        split_indices_payload[f"cls_train_{int(ratio * 100)}"] = cls_subset_idx[cls_train_local]
        split_indices_payload[f"cls_test_{int(ratio * 100)}"] = cls_subset_idx[cls_test_local]

        X_cls_train, y_cls_train = X_cls_exp[cls_train_local], y_cls_exp[cls_train_local]
        X_cls_test, y_cls_test_local = X_cls_exp[cls_test_local], y_cls_exp[cls_test_local]

        for model_name, model in make_classification_models().items():
            y_pred, fit_t, pred_t, peak_mb = _fit_predict_profile(model, X_cls_train, y_cls_train, X_cls_test)
            m = classification_metrics(y_cls_test_local, y_pred, model_name)
            split_records.append(
                {
                    "dataset": "classification",
                    "model": model_name,
                    "experiment": "split_sensitivity",
                    "train_ratio": ratio,
                    "metric_primary": "F1_macro",
                    "primary_score": m["F1_macro"],
                    "Accuracy": m["Accuracy"],
                    "Precision_macro": m["Precision_macro"],
                    "Recall_macro": m["Recall_macro"],
                    "F1_macro": m["F1_macro"],
                    "fit_time_sec": fit_t,
                    "predict_time_sec": pred_t,
                    "peak_mem_mb": peak_mb,
                }
            )

    split_df = pd.DataFrame(split_records)

    # --- Noise + corruption ---
    noise_records = []
    corruption_records = []
    base_ratio = 0.7

    reg_idx = np.arange(len(y_reg_all))
    reg_train_idx_base, reg_test_idx_base = train_test_split(reg_idx, train_size=base_ratio, random_state=seed, shuffle=True)
    X_reg_train_base, y_reg_train_base = X_reg_all[reg_train_idx_base], y_reg_all[reg_train_idx_base]
    X_reg_test_base, y_reg_test_base = X_reg_all[reg_test_idx_base], y_reg_all[reg_test_idx_base]

    cls_local_idx = np.arange(len(y_cls_exp))
    cls_train_base, cls_test_base = train_test_split(
        cls_local_idx,
        train_size=base_ratio,
        random_state=seed,
        stratify=y_cls_exp,
    )
    X_cls_train_base, y_cls_train_base = X_cls_exp[cls_train_base], y_cls_exp[cls_train_base]
    X_cls_test_base, y_cls_test_base = X_cls_exp[cls_test_base], y_cls_exp[cls_test_base]

    trained_reg_models = make_regression_models()
    for model in trained_reg_models.values():
        model.fit(X_reg_train_base, y_reg_train_base)

    trained_cls_models = make_classification_models()
    for model in trained_cls_models.values():
        model.fit(X_cls_train_base, y_cls_train_base)

    for sigma in [0.0, 0.1, 0.5, 1.0]:
        X_reg_noisy = _add_gaussian_noise(X_reg_test_base, sigma=sigma, seed=seed + int(1000 * sigma))
        X_cls_noisy = _add_gaussian_noise(X_cls_test_base, sigma=sigma, seed=seed + int(2000 * sigma))

        for model_name, model in trained_reg_models.items():
            pred_start = time.perf_counter()
            y_pred = model.predict(X_reg_noisy)
            pred_t = time.perf_counter() - pred_start
            m = regression_metrics(y_reg_test_base, y_pred)
            noise_records.append(
                {
                    "dataset": "regression",
                    "model": model_name,
                    "experiment": "noise_injection",
                    "sigma": sigma,
                    "metric_primary": "R2_test",
                    "primary_score": m["R2_test"],
                    **m,
                    "predict_time_sec": float(pred_t),
                }
            )

        for model_name, model in trained_cls_models.items():
            pred_start = time.perf_counter()
            y_pred = model.predict(X_cls_noisy)
            pred_t = time.perf_counter() - pred_start
            m = classification_metrics(y_cls_test_base, y_pred, model_name)
            noise_records.append(
                {
                    "dataset": "classification",
                    "model": model_name,
                    "experiment": "noise_injection",
                    "sigma": sigma,
                    "metric_primary": "F1_macro",
                    "primary_score": m["F1_macro"],
                    "Accuracy": m["Accuracy"],
                    "Precision_macro": m["Precision_macro"],
                    "Recall_macro": m["Recall_macro"],
                    "F1_macro": m["F1_macro"],
                    "predict_time_sec": float(pred_t),
                }
            )

    noise_df = pd.DataFrame(noise_records)

    for rate in [0.1, 0.2, 0.3]:
        for strategy in ["mean", "median", "knn"]:
            X_reg_corrupt = _corrupt_features(X_reg_test_base, corruption_rate=rate, seed=seed + int(rate * 100) + 5)
            X_cls_corrupt = _corrupt_features(X_cls_test_base, corruption_rate=rate, seed=seed + int(rate * 100) + 55)

            X_reg_repaired = _impute_with_strategy(X_reg_train_base, X_reg_corrupt, strategy)
            X_cls_repaired = _impute_with_strategy(X_cls_train_base, X_cls_corrupt, strategy)

            for model_name, model in trained_reg_models.items():
                y_pred = model.predict(X_reg_repaired)
                m = regression_metrics(y_reg_test_base, y_pred)
                corruption_records.append(
                    {
                        "dataset": "regression",
                        "model": model_name,
                        "experiment": "feature_corruption",
                        "corruption_rate": rate,
                        "imputation": strategy,
                        "metric_primary": "R2_test",
                        "primary_score": m["R2_test"],
                        **m,
                    }
                )

            for model_name, model in trained_cls_models.items():
                y_pred = model.predict(X_cls_repaired)
                m = classification_metrics(y_cls_test_base, y_pred, model_name)
                corruption_records.append(
                    {
                        "dataset": "classification",
                        "model": model_name,
                        "experiment": "feature_corruption",
                        "corruption_rate": rate,
                        "imputation": strategy,
                        "metric_primary": "F1_macro",
                        "primary_score": m["F1_macro"],
                        "Accuracy": m["Accuracy"],
                        "Precision_macro": m["Precision_macro"],
                        "Recall_macro": m["Recall_macro"],
                        "F1_macro": m["F1_macro"],
                    }
                )

    corruption_df = pd.DataFrame(corruption_records)

    # --- Convergence ---
    gd_reg = MiniBatchGDLR(
        lr=0.03,
        n_epochs=120,
        batch_size=256,
        lr_schedule="cosine_annealing",
        lr_min=1e-4,
        verbose=False,
    )
    gd_reg.fit(X_reg_train_base, y_reg_train_base, seed=seed)
    reg_curve = np.asarray(gd_reg.loss_history, dtype=float)

    gd_cls_curve = np.asarray(cls_artifacts.get("gd_binary_loss_history", {}).get("loss", []), dtype=float)
    newton_cls_curve = np.asarray(cls_artifacts.get("newton_binary_loss_history", {}).get("loss", []), dtype=float)

    curves = {"Regression Mini-batch GD (MSE)": reg_curve}
    if gd_cls_curve.size > 0:
        curves["Classification Logistic GD (BCE)"] = gd_cls_curve
    if newton_cls_curve.size > 0:
        curves["Classification Logistic Newton-IRLS (BCE)"] = newton_cls_curve

    convergence_rows = []
    for name, curve in curves.items():
        convergence_rows.append(
            {
                "algorithm": name,
                "iterations": int(len(curve)),
                "best_iteration": int(np.argmin(curve) + 1),
                "best_loss": float(np.min(curve)),
                "final_loss": float(curve[-1]),
            }
        )
    convergence_df = pd.DataFrame(convergence_rows).sort_values("best_loss")

    # --- Exports ---
    split_df_path = part3_results / "split_sensitivity_results.csv"
    noise_df_path = part3_results / "noise_robustness_results.csv"
    corruption_df_path = part3_results / "corruption_robustness_results.csv"
    convergence_df_path = part3_results / "convergence_summary.csv"
    log_path = part3_results / "experiment_log.jsonl"
    split_indices_path = part3_results / "split_indices.npz"
    env_path = part3_results / "environment_snapshot.txt"
    manifest_path = part3_results / "part3_manifest.json"

    split_df.to_csv(split_df_path, index=False)
    noise_df.to_csv(noise_df_path, index=False)
    corruption_df.to_csv(corruption_df_path, index=False)
    convergence_df.to_csv(convergence_df_path, index=False)

    all_records = []
    all_records.extend(split_df.to_dict(orient="records"))
    all_records.extend(noise_df.to_dict(orient="records"))
    all_records.extend(corruption_df.to_dict(orient="records"))

    with open(log_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            normalized = {k: _to_native(v) for k, v in rec.items()}
            normalized["seed"] = seed
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")

    np.savez(split_indices_path, **split_indices_payload)

    packages = [
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "scipy",
        "statsmodels",
        "joblib",
        "ipykernel",
        "jupyter",
        "nbformat",
        "tqdm",
    ]

    import platform
    from datetime import datetime, timezone, timedelta

    tz_vn = timezone(timedelta(hours=7))
    timestamp = datetime.now(tz=tz_vn).strftime("%Y-%m-%dT%H:%M:%S+07:00")

    env_lines = [
        f"Python: {sys.version}",
        f"Platform: {sys.platform}",
        f"Node: {platform.node()}",
        f"Processor: {platform.processor()} ({platform.machine()})",
        f"OS: {platform.platform()}",
        f"Seed: {seed}",
        f"Timestamp: {timestamp}",
        "",
        "Package versions:",
    ]
    for pkg in packages:
        try:
            env_lines.append(f"- {pkg}=={version(pkg)}")
        except PackageNotFoundError:
            env_lines.append(f"- {pkg}: not found")
    env_path.write_text("\n".join(env_lines), encoding="utf-8")


    # Determinism sanity check
    idx_train, idx_test = train_test_split(np.arange(len(y_reg_all)), train_size=0.7, random_state=seed, shuffle=True)
    m1 = NormalEquationLR().fit(X_reg_all[idx_train], y_reg_all[idx_train])
    m2 = NormalEquationLR().fit(X_reg_all[idx_train], y_reg_all[idx_train])
    p1 = m1.predict(X_reg_all[idx_test])
    p2 = m2.predict(X_reg_all[idx_test])
    reproducible = bool(np.allclose(p1, p2, atol=1e-12))

    def _to_rel(p: Path) -> str:
        try:
            return str(p.relative_to(repo_root.parent)).replace("\\", "/")
        except ValueError:
            return str(p).replace("\\", "/")

    manifest = {
        "seed": seed,
        "reproducible_sanity_check": reproducible,
        "files": {
            "comparison_table": _to_rel(comparison_table_path),
            "split_results": _to_rel(split_df_path),
            "noise_results": _to_rel(noise_df_path),
            "corruption_results": _to_rel(corruption_df_path),
            "convergence_summary": _to_rel(convergence_df_path),
            "experiment_log": _to_rel(log_path),
            "split_indices": _to_rel(split_indices_path),
            "environment_snapshot": _to_rel(env_path),
        },
        "counts": {
            "split_records": int(len(split_df)),
            "noise_records": int(len(noise_df)),
            "corruption_records": int(len(corruption_df)),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "repo_root": repo_root,
        "part3_results": part3_results,
        "reg_summary_df": reg_summary_df,
        "cls_summary_df": cls_summary_df,
        "comparison_table": comparison_table,
        "split_df": split_df,
        "noise_df": noise_df,
        "corruption_df": corruption_df,
        "convergence_df": convergence_df,
        "curves": curves,
        "manifest": manifest,
        "cls_artifacts": cls_artifacts,
    }
