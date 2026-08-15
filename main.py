"""
Main entry point — runs the full dual-stage pipeline described in the paper:

  1. U-Net 2D segmentation training (Section 3.2)
  2. Segmentation evaluation, Table 7 / Table 8 style report (Section 4.1)
  3. Feature extraction + XGBoost classification, Table 9 / 10 / 11 style
     report (Sections 3.3-3.5, 4.2-4.3)

By default this runs on the synthetic dataset (see data.py) so it works
out-of-the-box without the (large, gated) LiTS2017 / 3Dircadb downloads.
Pass --data-source lits --data-dir /path/to/lits to run on real data.
"""

import argparse
import gc
import os

from train_segmentation import train
from evaluate_segmentation import evaluate_dataset, print_report
from run_classification import run_full_classification_pipeline
from data import SyntheticLiverCTDataset, LiTSVolumeDataset
import torch


def main():
    parser = argparse.ArgumentParser(description="Dual-Stage Liver Tumor Characterization Pipeline")
    parser.add_argument("--data-source", choices=["synthetic", "lits"], default="synthetic")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--n-samples", type=int, default=40,
                         help="Number of synthetic segmentation samples")
    parser.add_argument("--n-cases", type=int, default=150,
                         help="Number of synthetic classification lesion cases")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--quick-test", action="store_true",
                         help="Run a fast smoke test (few epochs/samples)")
    parser.add_argument("--checkpoint", default=os.path.join(
        os.path.dirname(__file__), "..", "checkpoints", "unet_best.pt"))
    parser.add_argument("--output-json", default=os.path.join(
        os.path.dirname(__file__), "..", "outputs", "classification_results.json"))
    parser.add_argument("--skip-segmentation", action="store_true",
                         help="Skip U-Net training/eval, run classification stage only")
    args = parser.parse_args()

    if args.quick_test:
        args.n_samples = min(args.n_samples, 6)
        args.n_cases = min(args.n_cases, 16)
        args.epochs = 1

    if not args.skip_segmentation:
        print("=" * 70)
        print("STAGE 1: U-Net 2D Segmentation Training (Section 3.2)")
        print("=" * 70)
        model, history = train(
            data_source=args.data_source,
            data_dir=args.data_dir,
            n_samples=args.n_samples,
            epochs=args.epochs,
            checkpoint_path=args.checkpoint,
            quick_test=args.quick_test,
        )

        print("\n" + "=" * 70)
        print("STAGE 2: Segmentation Evaluation (Section 4.1, Tables 7-8)")
        print("=" * 70)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.data_source == "lits" and args.data_dir is not None:
            eval_dataset = LiTSVolumeDataset(args.data_dir, augment=False)
        else:
            eval_dataset = SyntheticLiverCTDataset(n_samples=max(10, args.n_samples // 2), augment=False)
        seg_results = evaluate_dataset(model, eval_dataset, device, batch_size=2)
        print_report(seg_results)

        # free memory before the classification stage (important on
        # memory-constrained machines, since both stages hold large
        # 512x512 tensors/arrays in a single process)
        del model, history, eval_dataset, seg_results
        gc.collect()
    else:
        print("Skipping segmentation stage (--skip-segmentation).")

    print("\n" + "=" * 70)
    print("STAGE 3: Feature Extraction + XGBoost Classification "
          "(Sections 3.3-3.5, 4.2-4.3)")
    print("=" * 70)
    clf_results = run_full_classification_pipeline(
        n_cases=args.n_cases, output_json=args.output_json,
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
