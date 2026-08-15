"""
Sections 3.3 / 3.4 / 3.5 / 4.2 / 4.3 — Full classification pipeline

For each lesion case:
  1. Obtain (image, tumor_mask) — either from U-Net segmentation output or
     ground-truth masks (useful when isolating classifier performance from
     segmentation error, as the paper's classification module does using a
     consolidated 150-case dataset with radiologist-assigned labels).
  2. Extract the 25-feature vector (morphological + texture + intensity).
  3. Train/evaluate XGBoost with stratified K-fold CV, bootstrap CIs, and
     feature importance ranking.

Because "Benign" vs. "Malignant" ground-truth labels are not present in the
synthetic segmentation masks (nor in the raw LiTS masks — see the paper's
"Clarification on Ground Truth Labels", Section 3.1.1), this script's
synthetic-data path assigns pathology labels via a simple heuristic
correlated with texture/intensity features, purely so the classification
stage can be run and validated end-to-end. On real data, ground-truth
Benign/Malignant labels should come from the 3Dircadb dataset and/or
radiologist annotation as described in the paper.
"""

import argparse
import json
import numpy as np

from data import SyntheticLiverCTDataset, _make_synthetic_ct_slice
from feature_extraction import extract_all_features, features_to_vector
from classifier import (
    build_xgb_classifier, z_score_normalize, stratified_kfold_cv,
    bootstrap_confidence_intervals, wilcoxon_significance_test,
    feature_importance_report, evaluate_predictions, FEATURE_CATEGORY_MAP,
)
from config import dataset_cfg


FEATURE_ORDER = [
    "area", "perimeter", "circularity", "compactness", "eccentricity",
    "solidity", "extent", "major_axis_length", "minor_axis_length",
    "equivalent_diameter", "orientation",
    "glcm_homogeneity", "glcm_contrast", "glcm_energy", "glcm_correlation",
    "glcm_entropy", "glcm_dissimilarity", "glcm_asm",
    "mean_intensity", "std_intensity", "skewness", "kurtosis",
    "min_intensity", "max_intensity", "intensity_range",
]


def build_synthetic_lesion_dataset(n_cases: int = None, seed: int = None):
    """Builds a synthetic 150-case lesion dataset (70 benign / 80 malignant,
    matching Section 4.2 / Table 10 totals) with extracted feature vectors
    and pathology labels, for demonstrating the full classification pipeline.
    """
    n = dataset_cfg.n_classification_cases if n_cases is None else n_cases
    rs = dataset_cfg.random_state if seed is None else seed
    rng = np.random.default_rng(rs)

    # preserve the paper's benign:malignant ratio (70:80) at whatever
    # total case count is requested, guaranteeing at least 1 of each class
    benign_frac = dataset_cfg.n_benign / dataset_cfg.n_classification_cases
    n_benign = max(1, min(n - 1, round(n * benign_frac)))
    n_malignant = n - n_benign
    labels_plan = [0] * n_benign + [1] * n_malignant
    rng.shuffle(labels_plan)

    X_rows, y_rows = [], []
    for i, label in enumerate(labels_plan):
        local_seed = int(rng.integers(0, 1_000_000)) + i
        local_rng = np.random.default_rng(local_seed)

        size_cat = local_rng.choice(["small", "medium", "large"], p=[18 / 70, 32 / 70, 20 / 70])
        image, mask = _make_synthetic_ct_slice(local_rng, tumor_size_category=size_cat)
        tumor_mask = (mask == 2)

        if tumor_mask.sum() < 5:
            # ensure a non-trivial region exists
            tumor_mask = (mask == 1)

        # Malignant lesions: bias toward higher GLCM contrast / lower
        # homogeneity / more irregular shape (heuristic surrogate labels,
        # since ground truth pathology requires 3Dircadb / radiologist input)
        if label == 1:
            noise = local_rng.normal(0, 8, size=image.shape)
            image = np.where(tumor_mask, image + noise * 2.5, image)

        feats = extract_all_features(image, tumor_mask.astype(np.uint8))
        X_rows.append(features_to_vector(feats, FEATURE_ORDER))
        y_rows.append(label)

    X = np.array(X_rows)
    y = np.array(y_rows)
    return X, y, FEATURE_ORDER


def run_full_classification_pipeline(n_cases: int = None, output_json: str = None):
    print("Building lesion feature dataset (morphological + GLCM texture + intensity)...")
    X, y, feature_names = build_synthetic_lesion_dataset(n_cases=n_cases)
    print(f"Dataset: {X.shape[0]} cases, {X.shape[1]} features, "
          f"{int((y == 0).sum())} benign / {int((y == 1).sum())} malignant")

    # ---------------- Stratified K-fold CV (Table 13 style) ----------------
    print("\nRunning 5-fold stratified cross-validation...")
    cv_results = stratified_kfold_cv(X, y)
    print("Per-fold results:")
    for f in cv_results["folds"]:
        print(f"  Fold {f['fold']}: acc={f['accuracy']:.2f}%  "
              f"prec={f['precision']:.2f}%  rec={f['recall']:.2f}%  f1={f['f1']:.2f}%")
    print("Mean ± Std across folds:")
    for k, v in cv_results["summary"].items():
        print(f"  {k:12s}: {v['mean']:.2f} ± {v['std']:.2f}")

    # ---------------- Final model on full data + bootstrap CI (Table 9) --
    print("\nTraining final model on full dataset for feature importance / bootstrap CI...")
    X_scaled, scaler = z_score_normalize(X)
    model = build_xgb_classifier()
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    point_metrics = evaluate_predictions(y, y_pred)
    print("Point-estimate metrics (in-sample, for feature-importance context):")
    for k, v in point_metrics.items():
        print(f"  {k:12s}: {v:.2f}%")

    print("\nComputing bootstrap 95% confidence intervals (1000 iterations)...")
    ci_results = bootstrap_confidence_intervals(y, y_pred)
    print(f"{'Metric':<15}{'Value (%)':<12}{'95% CI'}")
    for k, v in ci_results.items():
        print(f"{k:<15}{v['value']:<12.2f}[{v['ci_lower']:.1f} - {v['ci_upper']:.1f}]")

    print("\nWilcoxon signed-rank test vs. random classifier baseline:")
    sig = wilcoxon_significance_test(y, y_pred)
    print(f"  statistic={sig['statistic']}, p-value={sig['p_value']:.6f}")

    # ---------------- Feature importance (Table 11 style) ----------------
    print("\nTop 10 features by importance:")
    fi = feature_importance_report(model, feature_names, top_n=10,
                                    category_map=FEATURE_CATEGORY_MAP)
    print(f"{'Rank':<6}{'Feature':<25}{'Importance':<14}{'Category'}")
    for row in fi:
        print(f"{row['rank']:<6}{row['feature']:<25}{row['importance']:<14.4f}{row['category']}")

    category_totals = {}
    for row in fi:
        category_totals[row["category"]] = category_totals.get(row["category"], 0) + row["importance"]
    print("\nCategory contribution (top-10 features):")
    for cat, total in sorted(category_totals.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:<15}: {total * 100:.1f}%")

    results = {
        "dataset_size": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "cv_results": cv_results,
        "point_metrics": point_metrics,
        "bootstrap_ci": ci_results,
        "wilcoxon_test": sig,
        "feature_importance": fi,
    }

    if output_json:
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved full results to {output_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cases", type=int, default=150)
    parser.add_argument("--output-json", default="../outputs/classification_results.json")
    args = parser.parse_args()

    run_full_classification_pipeline(n_cases=args.n_cases, output_json=args.output_json)
