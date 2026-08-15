# Dual-Stage Liver Tumor Characterization

Implementation of **"Dual-Stage Computational Approach for Liver Tumor
Characterization: U-Net 2D Spatial Analysis and XGBoost-Based Classification"**
(Gautum, Izhar, Gautam, Sardana, Rai — *Procedia Computer Science* 282 (2026)
952–966, FTNCT08).

The paper's pipeline has two stages:
1. **U-Net 2D** segments liver and tumor regions from abdominal CT slices.
2. **XGBoost**, fed 25 morphological/texture/intensity features extracted
   from the segmented lesions, classifies each lesion as benign or malignant.

## Project structure

```
liver_tumor_pipeline/
├── src/
│   ├── config.py                  # All hyperparameters (Tables 4, 5, 6)
│   ├── preprocessing.py           # Windowing, norm, denoise, equalize, augment (Sec 3.1.2, Eqs 1-3)
│   ├── unet.py                    # U-Net 2D architecture (Sec 3.2, Eqs 4-7)
│   ├── losses.py                  # Hybrid Dice+BCE loss (Sec 3.2.3, Eqs 8-10)
│   ├── metrics_segmentation.py    # Dice, IoU, VOE, RVD (Sec 4.1)
│   ├── feature_extraction.py      # Morphological/GLCM/intensity features (Sec 3.3)
│   ├── classifier.py              # XGBoost + CV + bootstrap CI + feature importance (Sec 3.4/3.5/4.2/4.3)
│   ├── data.py                    # LiTS NIfTI loader + synthetic data generator
│   ├── train_segmentation.py      # U-Net training script (Table 4)
│   ├── evaluate_segmentation.py   # Table 7/8-style segmentation report
│   ├── run_classification.py      # Table 9/10/11-style classification report
│   ├── main.py                    # End-to-end pipeline runner
│   └── test_pipeline.py           # Sanity test suite (14 tests)
├── checkpoints/                   # Saved U-Net weights
├── outputs/                       # JSON results
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Running

**Full pipeline (synthetic data, quick smoke test):**
```bash
cd src
python3 main.py --quick-test
```

**Full pipeline with more realistic sample sizes** (slower; increase
`--n-samples`/`--n-cases`/`--epochs` toward the paper's actual scale —
131 training volumes, 150 classification cases, 50 epochs — as your
hardware allows):
```bash
python3 main.py --n-samples 60 --n-cases 150 --epochs 50
```

**On real LiTS2017 / 3Dircadb data** (once downloaded — see note below):
```bash
python3 main.py --data-source lits --data-dir /path/to/lits --n-samples 60
```

**Individual stages:**
```bash
python3 train_segmentation.py --n-samples 40 --epochs 50
python3 evaluate_segmentation.py --checkpoint ../checkpoints/unet_best.pt
python3 run_classification.py --n-cases 150
```

**Tests:**
```bash
python3 test_pipeline.py
```

## Important note on data

The real **LiTS2017** dataset (131 training / 70 testing CT volumes) and
**3Dircadb** dataset (20 patients with pathology labels) referenced by the
paper are large (tens of GB), require registration/gated downloads, and
are not available in this environment. Per the paper's own "Clarification
on Ground Truth Labels" (Section 3.1.1), the raw LiTS masks only label
Liver vs. Lesion — Benign/Malignant pathology labels come from 3Dircadb
and radiologist annotation.

To let the full pipeline run and be verified end-to-end without those
downloads, `data.py` includes:
- `LiTSVolumeDataset` — a real loader for LiTS-format `.nii`/`.nii.gz`
  volumes, ready to use once you have the data (see
  https://competitions.codalab.org/competitions/17094 or the LiTS
  benchmark paper [24] for access).
- `SyntheticLiverCTDataset` — a structurally faithful synthetic
  generator (same 512×512 resolution, same 3-class label scheme, same
  small/medium/large tumor-size distribution as Table 8) so the U-Net
  training loop, evaluation metrics, feature extraction, and XGBoost
  classifier can all be exercised and validated.

Swap in real data via `--data-source lits --data-dir ...` once available;
no other code changes are needed.

## What's implemented from the paper

| Paper section | Component | File |
|---|---|---|
| 3.1.2, Eqs 1-3 | Windowing, normalization, median denoise, histogram equalization | `preprocessing.py` |
| 3.2, Eqs 4-7 | U-Net 2D (encoder/decoder, skip connections, softmax output) | `unet.py` |
| 3.2.3, Eqs 8-10 | Hybrid Dice+BCE loss | `losses.py` |
| Table 4 | Adam, lr=1e-4, batch=16, 50 epochs, early stopping (patience=10) | `train_segmentation.py`, `config.py` |
| 3.3 | Morphological, GLCM texture, intensity feature extraction (25 features) | `feature_extraction.py` |
| 3.4/3.5, Tables 5-6 | XGBoost hyperparameters, stratified K-fold CV | `classifier.py` |
| 4.1, Tables 7-8 | Dice/IoU/VOE/RVD, tumor-size stratified reporting | `metrics_segmentation.py`, `evaluate_segmentation.py` |
| 4.2, Table 9-10 | Bootstrap 95% CIs, Wilcoxon significance test, confusion matrix | `classifier.py`, `run_classification.py` |
| 4.3, Table 11 | XGBoost feature importance ranking | `classifier.py` |
| 4.5, Table 13 | 5-fold cross-validation | `classifier.py` |

## Known limitations of this implementation

- **Synthetic data only reproduces structure, not the paper's exact
  reported numbers.** Reported metrics (Dice 96.84%/71.35%, classification
  accuracy 96.78%, etc.) reflect training on the real LiTS2017/3Dircadb
  data over 50 epochs; they cannot be reproduced from synthetic data.
  Code correctness was instead validated via the unit test suite
  (`test_pipeline.py`) and end-to-end smoke runs.
- **Compute**: the paper trains on 512×512 CT slices with a ~31M-parameter
  U-Net; full training (Table 4: up to 50 epochs, batch size 16) is
  intended for GPU hardware. On CPU-only or memory-constrained machines,
  reduce `--n-samples`, batch size, and epoch count, or use `--quick-test`.
- **Benign/Malignant surrogate labels**: `run_classification.py`'s
  synthetic path assigns pathology labels via a texture/intensity
  heuristic (documented in-code) purely so the classifier stage can be
  exercised; real experiments require actual radiologist/3Dircadb labels.
