"""
Section 4.1 — Segmentation Performance Evaluation

Reproduces the style of Table 7 (overall Dice/IoU/VOE/RVD for liver and
tumor) and Table 8 (breakdown by tumor size category) using a trained
U-Net checkpoint evaluated on held-out data.
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from unet import UNet2D
from data import SyntheticLiverCTDataset, LiTSVolumeDataset
from metrics_segmentation import segmentation_report, categorize_tumor_size
from config import unet_train_cfg


def load_model(checkpoint_path: str, device):
    model = UNet2D().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_dataset(model, dataset, device, batch_size: int = 4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    liver_reports, tumor_reports = [], []
    size_bucket_tumor_dice = {"small": [], "medium": [], "large": []}

    idx_offset = 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs_dev = imgs.to(device)
            logits = model(imgs_dev)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            masks_np = masks.numpy()

            for b in range(preds.shape[0]):
                pred = preds[b]
                gt = masks_np[b]

                pred_liver = (pred >= 1)
                gt_liver = (gt >= 1)
                liver_reports.append(segmentation_report(pred_liver, gt_liver))

                pred_tumor = (pred == 2)
                gt_tumor = (gt == 2)
                if gt_tumor.sum() > 0:
                    rep = segmentation_report(pred_tumor, gt_tumor)
                    tumor_reports.append(rep)

                    cat = categorize_tumor_size(int(gt_tumor.sum()))
                    size_bucket_tumor_dice[cat].append(rep["dice_pct"])

                idx_offset += 1

    def summarize(reports, key):
        vals = [r[key] for r in reports]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (0.0, 0.0)

    liver_summary = {k: summarize(liver_reports, k) for k in ["dice_pct", "iou_pct", "voe_pct", "rvd_pct"]}
    tumor_summary = {k: summarize(tumor_reports, k) for k in ["dice_pct", "iou_pct", "voe_pct", "rvd_pct"]}

    size_summary = {}
    for cat, vals in size_bucket_tumor_dice.items():
        size_summary[cat] = {
            "n_cases": len(vals),
            "tumor_dice_mean": float(np.mean(vals)) if vals else 0.0,
            "tumor_dice_std": float(np.std(vals)) if vals else 0.0,
        }

    return {
        "liver": liver_summary,
        "tumor": tumor_summary,
        "by_size": size_summary,
    }


def print_report(results: dict):
    print("\n=== Table 7 style: Segmentation Performance Metrics ===")
    print(f"{'Task':<20}{'Dice (%)':<18}{'IoU (%)':<18}{'VOE (%)':<18}{'RVD (%)':<18}")
    for task, key in [("Liver Segmentation", "liver"), ("Tumor Segmentation", "tumor")]:
        s = results[key]
        print(f"{task:<20}"
              f"{s['dice_pct'][0]:6.2f} ± {s['dice_pct'][1]:<8.2f}"
              f"{s['iou_pct'][0]:6.2f} ± {s['iou_pct'][1]:<8.2f}"
              f"{s['voe_pct'][0]:6.2f} ± {s['voe_pct'][1]:<8.2f}"
              f"{s['rvd_pct'][0]:6.2f} ± {s['rvd_pct'][1]:<8.2f}")

    print("\n=== Table 8 style: Segmentation Performance by Tumor Size ===")
    print(f"{'Tumor Size':<20}{'Cases':<10}{'Tumor Dice (%)':<20}")
    for cat in ["small", "medium", "large"]:
        s = results["by_size"][cat]
        print(f"{cat:<20}{s['n_cases']:<10}{s['tumor_dice_mean']:6.2f} ± {s['tumor_dice_std']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="../checkpoints/unet_best.pt")
    parser.add_argument("--data-source", choices=["synthetic", "lits"], default="synthetic")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--n-samples", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    if args.data_source == "lits" and args.data_dir is not None:
        dataset = LiTSVolumeDataset(args.data_dir, augment=False)
    else:
        dataset = SyntheticLiverCTDataset(n_samples=args.n_samples, augment=False)

    results = evaluate_dataset(model, dataset, device)
    print_report(results)
