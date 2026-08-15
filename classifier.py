"""
Sections 3.4 / 3.5 / 4.2 / 4.3 — XGBoost Classification

- Trains an XGBoost binary classifier (Benign=0 / Malignant=1) on the
  25-dim quantitative feature vectors extracted per lesion.
- Stratified K-fold cross-validation (Section 3.5, Table 13).
- Bootstrap 95% confidence intervals for accuracy/precision/recall/F1/specificity
  (Section 4.2, Table 9).
- Feature importance ranking (Section 4.3, Table 11).
"""

import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)
from scipy.stats import wilcoxon

from config import xgb_cfg, dataset_cfg


def build_xgb_classifier() -> xgb.XGBClassifier:
    """Instantiate XGBoost with the exact hyperparameters from Table 5/6."""
    return xgb.XGBClassifier(
        learning_rate=xgb_cfg.learning_rate,
        max_depth=xgb_cfg.max_depth,
        min_child_weight=xgb_cfg.min_child_weight,
        subsample=xgb_cfg.subsample,
        colsample_bytree=xgb_cfg.colsample_bytree,
        gamma=xgb_cfg.gamma,
        reg_lambda=xgb_cfg.reg_lambda,
        reg_alpha=xgb_cfg.reg_alpha,
        n_estimators=xgb_cfg.n_estimators,
        objective=xgb_cfg.objective,
        eval_metric=xgb_cfg.eval_metric,
        random_state=xgb_cfg.random_state,
    )


def z_score_normalize(X_train: np.ndarray, X_test: np.ndarray = None):
    """Section 3.3 (final paragraph): z-score normalization applied to all 25 features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    return X_train_scaled, scaler


def compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "precision": precision_score(y_true, y_pred, zero_division=0) * 100,
        "recall": recall_score(y_true, y_pred, zero_division=0) * 100,
        "f1": f1_score(y_true, y_pred, zero_division=0) * 100,
        "specificity": compute_specificity(y_true, y_pred) * 100,
    }


def stratified_kfold_cv(X: np.ndarray, y: np.ndarray, n_folds: int = None,
                         random_state: int = None) -> dict:
    """Section 3.5 / Table 13 — stratified K-fold CV preserving class ratio per fold."""
    k = dataset_cfg.n_cv_folds if n_folds is None else n_folds
    rs = dataset_cfg.random_state if random_state is None else random_state

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=rs)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_s, X_val_s, _ = z_score_normalize(X_train, X_val)

        model = build_xgb_classifier()
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_val_s)

        metrics = evaluate_predictions(y_val, y_pred)
        metrics["fold"] = fold_idx
        fold_results.append(metrics)

    keys = ["accuracy", "precision", "recall", "f1", "specificity"]
    summary = {
        k_: {
            "mean": float(np.mean([f[k_] for f in fold_results])),
            "std": float(np.std([f[k_] for f in fold_results])),
        }
        for k_ in keys
    }
    return {"folds": fold_results, "summary": summary}


def bootstrap_confidence_intervals(y_true: np.ndarray, y_pred: np.ndarray,
                                    n_iterations: int = None,
                                    ci: float = 95.0,
                                    random_state: int = None) -> dict:
    """Section 4.2 — bootstrap (1000 iterations) 95% CIs for each metric (Table 9)."""
    n_iter = dataset_cfg.bootstrap_iterations if n_iterations is None else n_iterations
    rs = dataset_cfg.random_state if random_state is None else random_state
    rng = np.random.default_rng(rs)

    n = len(y_true)
    boot_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "specificity": []}

    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        m = evaluate_predictions(yt, yp)
        for k_ in boot_metrics:
            boot_metrics[k_].append(m[k_])

    alpha = (100 - ci) / 2.0
    result = {}
    for k_, vals in boot_metrics.items():
        vals = np.array(vals)
        if vals.size == 0:
            result[k_] = {"value": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
            continue
        result[k_] = {
            "value": float(np.mean(vals)),
            "ci_lower": float(np.percentile(vals, alpha)),
            "ci_upper": float(np.percentile(vals, 100 - alpha)),
        }
    return result


def wilcoxon_significance_test(y_true: np.ndarray, y_pred: np.ndarray,
                                random_pred: np.ndarray = None):
    """Section 4.2 — Wilcoxon signed-rank test vs. a random classifier baseline."""
    correct = (y_pred == y_true).astype(int)
    if random_pred is None:
        rng = np.random.default_rng(dataset_cfg.random_state)
        random_pred = rng.integers(0, 2, size=len(y_true))
    random_correct = (random_pred == y_true).astype(int)

    if np.all(correct == random_correct):
        return {"statistic": None, "p_value": 1.0}

    stat, p_value = wilcoxon(correct, random_correct, zero_method="wilcox",
                              correction=True, alternative="greater")
    return {"statistic": float(stat), "p_value": float(p_value)}


def feature_importance_report(model: xgb.XGBClassifier, feature_names: list,
                               top_n: int = 10, category_map: dict = None) -> list:
    """Section 4.3 / Table 11 — feature importance ranking with category grouping."""
    importances = model.feature_importances_
    # normalize to sum to 1 for interpretability, matching the paper's reported scores
    importances = importances / (importances.sum() + 1e-12)

    order = np.argsort(importances)[::-1][:top_n]
    report = []
    for rank, idx in enumerate(order, start=1):
        name = feature_names[idx]
        cat = category_map.get(name, "Unknown") if category_map else "Unknown"
        report.append({
            "rank": rank,
            "feature": name,
            "importance": float(importances[idx]),
            "category": cat,
        })
    return report


FEATURE_CATEGORY_MAP = {
    # morphological
    "area": "Morphological", "perimeter": "Morphological", "circularity": "Morphological",
    "compactness": "Morphological", "eccentricity": "Morphological", "solidity": "Morphological",
    "extent": "Morphological", "major_axis_length": "Morphological",
    "minor_axis_length": "Morphological", "equivalent_diameter": "Morphological",
    "orientation": "Morphological",
    # texture (GLCM)
    "glcm_homogeneity": "Texture", "glcm_contrast": "Texture", "glcm_energy": "Texture",
    "glcm_correlation": "Texture", "glcm_entropy": "Texture", "glcm_dissimilarity": "Texture",
    "glcm_asm": "Texture",
    # intensity
    "mean_intensity": "Intensity", "std_intensity": "Intensity", "skewness": "Intensity",
    "kurtosis": "Intensity", "min_intensity": "Intensity", "max_intensity": "Intensity",
    "intensity_range": "Intensity", "median_intensity": "Intensity", "iqr_intensity": "Intensity",
}


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n_samples, n_features = 150, 25
    X = rng.normal(0, 1, size=(n_samples, n_features))
    y = (X[:, 0] + X[:, 1] * 0.5 + rng.normal(0, 0.3, n_samples) > 0).astype(int)
    feature_names = [f"feat_{i}" for i in range(n_features)]

    cv_result = stratified_kfold_cv(X, y)
    print("CV summary:", cv_result["summary"])

    X_s, scaler = z_score_normalize(X)
    model = build_xgb_classifier()
    model.fit(X_s, y)
    y_pred = model.predict(X_s)

    ci_result = bootstrap_confidence_intervals(y, y_pred, n_iterations=200)
    print("Bootstrap CI:", ci_result)

    sig = wilcoxon_significance_test(y, y_pred)
    print("Wilcoxon test:", sig)

    fi = feature_importance_report(model, feature_names, top_n=5)
    print("Top features:", fi)
