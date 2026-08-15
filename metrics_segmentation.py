"""
Section 4.1 — Segmentation Performance Metrics (Table 7)

Dice        = 2|X ∩ Y| / (|X| + |Y|)
IoU         = |X ∩ Y| / |X ∪ Y|
VOE         = 1 - IoU  (Volumetric Overlap Error, expressed as %)
RVD         = (|X| - |Y|) / |Y|  (Relative Volume Difference, expressed as %, unsigned)
"""

import numpy as np


def dice_coefficient(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    return (2.0 * intersection + eps) / (pred.sum() + gt.sum() + eps)


def iou_score(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return (intersection + eps) / (union + eps)


def voe_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Volumetric Overlap Error (%) = (1 - IoU) * 100."""
    return (1.0 - iou_score(pred_mask, gt_mask)) * 100.0


def rvd_score(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7) -> float:
    """Relative Volume Difference (%) = |(|pred| - |gt|) / |gt|| * 100."""
    pred_vol = pred_mask.astype(bool).sum()
    gt_vol = gt_mask.astype(bool).sum()
    return abs(pred_vol - gt_vol) / (gt_vol + eps) * 100.0


def segmentation_report(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    return {
        "dice_pct": dice_coefficient(pred_mask, gt_mask) * 100.0,
        "iou_pct": iou_score(pred_mask, gt_mask) * 100.0,
        "voe_pct": voe_score(pred_mask, gt_mask),
        "rvd_pct": rvd_score(pred_mask, gt_mask),
    }


def categorize_tumor_size(area_pixels: int, pixel_spacing_mm: float = 1.0) -> str:
    """Table 8: bucket a tumor's diameter into Small(<2cm) / Medium(2-5cm) / Large(>5cm).

    Approximates an equivalent circular diameter from area:
      diameter = 2 * sqrt(area / pi)
    """
    diameter_mm = 2.0 * np.sqrt(area_pixels * (pixel_spacing_mm ** 2) / np.pi)
    diameter_cm = diameter_mm / 10.0
    if diameter_cm < 2.0:
        return "small"
    elif diameter_cm <= 5.0:
        return "medium"
    else:
        return "large"


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    gt = (rng.random((100, 100)) > 0.7).astype(np.uint8)
    pred = gt.copy()
    pred[:10, :10] = 1 - pred[:10, :10]  # introduce some errors
    print(segmentation_report(pred, gt))
