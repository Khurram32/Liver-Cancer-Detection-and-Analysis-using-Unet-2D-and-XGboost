"""
Lightweight sanity tests for each pipeline component. Run with:
    python3 test_pipeline.py
"""

import numpy as np
import torch

from preprocessing import (
    apply_windowing, normalize_intensity, median_denoise,
    histogram_equalize, resize_image, preprocess_slice,
)
from unet import UNet2D
from losses import HybridDiceBCELoss, DiceLoss
from metrics_segmentation import dice_coefficient, iou_score, voe_score, rvd_score
from feature_extraction import extract_all_features
from classifier import build_xgb_classifier, z_score_normalize
from data import SyntheticLiverCTDataset


def test_windowing():
    hu = np.array([[-2000, 0, 2000], [40, 240, -160]], dtype=np.float32)
    out = apply_windowing(hu, window_level=40, window_width=400)
    assert out.min() >= -160 and out.max() <= 240
    print("PASS: windowing clips to [WL-WW/2, WL+WW/2]")


def test_normalization():
    img = np.random.normal(50, 10, size=(32, 32))
    norm = normalize_intensity(img)
    assert abs(norm.mean()) < 1e-6
    assert abs(norm.std() - 1.0) < 1e-6
    print("PASS: normalization gives zero mean, unit std")


def test_median_denoise_shape():
    img = np.random.rand(32, 32).astype(np.float32)
    out = median_denoise(img)
    assert out.shape == img.shape
    print("PASS: median denoise preserves shape")


def test_histogram_equalize_range():
    img = np.random.normal(100, 20, size=(64, 64))
    out = histogram_equalize(img)
    assert out.min() >= 0
    print("PASS: histogram equalization produces valid output")


def test_resize():
    img = np.random.rand(100, 100).astype(np.float32)
    out = resize_image(img)
    assert out.shape == (512, 512)
    print("PASS: resize produces 512x512 output")


def test_full_preprocess_pipeline():
    hu = np.random.normal(30, 200, size=(256, 256)).astype(np.float32)
    mask = (np.random.rand(256, 256) > 0.7).astype(np.int64)
    img, m = preprocess_slice(hu, apply_augmentation=True, mask=mask)
    assert img.shape == (512, 512)
    assert m.shape == (512, 512)
    print("PASS: full preprocessing pipeline (windowing->norm->denoise->"
          "equalize->resize->augment)")


def test_unet_output_shape():
    model = UNet2D()
    x = torch.randn(1, 1, 128, 128)
    out = model(x)
    assert out.shape == (1, 3, 128, 128)  # C=3: background, liver, tumor
    print("PASS: U-Net outputs (B, 3, H, W) logits")


def test_unet_softmax_sums_to_one():
    model = UNet2D()
    x = torch.randn(1, 1, 64, 64)
    proba = model.predict_proba(x)
    sums = proba.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    print("PASS: U-Net softmax output sums to 1 per pixel (Eq. 7)")


def test_dice_loss_range():
    loss_fn = DiceLoss(num_classes=3)
    logits = torch.randn(2, 3, 16, 16)
    targets = torch.randint(0, 3, (2, 16, 16))
    loss = loss_fn(logits, targets)
    assert 0.0 <= loss.item() <= 1.5
    print("PASS: Dice loss in expected range")


def test_hybrid_loss():
    loss_fn = HybridDiceBCELoss(lam=0.5)
    logits = torch.randn(2, 3, 16, 16)
    targets = torch.randint(0, 3, (2, 16, 16))
    loss = loss_fn(logits, targets)
    assert loss.item() > 0
    print("PASS: hybrid Dice+BCE loss (Eq. 8) computed")


def test_segmentation_metrics_perfect_match():
    mask = (np.random.rand(50, 50) > 0.5).astype(np.uint8)
    assert abs(dice_coefficient(mask, mask) - 1.0) < 1e-5
    assert abs(iou_score(mask, mask) - 1.0) < 1e-5
    assert abs(voe_score(mask, mask)) < 1e-3
    assert abs(rvd_score(mask, mask)) < 1e-3
    print("PASS: segmentation metrics are correct for perfect prediction "
          "(Dice=IoU=1, VOE=RVD=0)")


def test_feature_extraction_count():
    image = np.random.normal(100, 20, size=(64, 64))
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:45] = 1
    feats = extract_all_features(image, mask)
    assert len(feats) >= 25
    assert "circularity" in feats and "glcm_contrast" in feats and "mean_intensity" in feats
    print(f"PASS: feature extraction yields {len(feats)} features "
          f"(morphological+texture+intensity)")


def test_xgboost_classifier_fit_predict():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 25))
    y = (X[:, 0] > 0).astype(int)
    X_s, _ = z_score_normalize(X)
    model = build_xgb_classifier()
    model.fit(X_s, y)
    preds = model.predict(X_s)
    assert preds.shape == y.shape
    print("PASS: XGBoost classifier trains and predicts")


def test_synthetic_dataset():
    ds = SyntheticLiverCTDataset(n_samples=4)
    img, mask = ds[0]
    assert img.shape == (1, 512, 512)
    assert mask.shape == (512, 512)
    assert set(np.unique(mask).tolist()).issubset({0, 1, 2})
    print("PASS: synthetic dataset produces valid (image, 3-class mask) pairs")


if __name__ == "__main__":
    tests = [
        test_windowing, test_normalization, test_median_denoise_shape,
        test_histogram_equalize_range, test_resize, test_full_preprocess_pipeline,
        test_unet_output_shape, test_unet_softmax_sums_to_one,
        test_dice_loss_range, test_hybrid_loss,
        test_segmentation_metrics_perfect_match, test_feature_extraction_count,
        test_xgboost_classifier_fit_predict, test_synthetic_dataset,
    ]
    print(f"Running {len(tests)} sanity tests...\n")
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed.")
