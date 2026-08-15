"""
Configuration for Dual-Stage Liver Tumor Characterization Pipeline
(U-Net 2D Segmentation + XGBoost Classification)

All hyperparameters are taken directly from the paper:
"Dual-Stage Computational Approach for Liver Tumor Characterization:
U-Net 2D Spatial Analysis and XGBoost-Based Classification"
(Gautum, Izhar, Gautam, Sardana, Rai — Procedia Computer Science 282 (2026) 952-966)
"""

from dataclasses import dataclass, field
from typing import Tuple


# ----------------------------------------------------------------------
# Section 3.1.2 — Image Preprocessing
# ----------------------------------------------------------------------
@dataclass
class PreprocessConfig:
    # 1. Windowing (Hounsfield Units)
    window_level: float = 40.0     # W_L
    window_width: float = 400.0    # W_W

    # 3. Noise reduction
    median_kernel_size: int = 3    # 3x3 median filter

    # 4. Contrast enhancement
    num_gray_levels: int = 256     # L, for histogram equalization

    # 5. Resizing and augmentation
    image_size: Tuple[int, int] = (512, 512)
    augmentation_prob: float = 0.5
    rotation_deg: float = 15.0     # +/- 15 degrees
    elastic_alpha: float = 34.0
    elastic_sigma: float = 4.0


# ----------------------------------------------------------------------
# Table 4 — U-Net Training Protocol and Hyperparameter Configuration
# ----------------------------------------------------------------------
@dataclass
class UNetTrainConfig:
    optimizer: str = "adam"
    learning_rate: float = 1e-4
    batch_size: int = 16
    epochs: int = 50
    augmentation_prob: float = 0.5
    early_stopping_patience: int = 10
    dice_bce_lambda: float = 0.5   # lambda weighting Dice vs BCE (Eq. 8)
    input_size: Tuple[int, int] = (512, 512)
    num_classes: int = 3           # background, liver, tumor (Eq. 7)
    dice_smooth_eps: float = 1e-7  # epsilon in Eq. 9


# ----------------------------------------------------------------------
# U-Net Architecture (Section 3.2.1 / 3.2.2)
# ----------------------------------------------------------------------
@dataclass
class UNetArchConfig:
    in_channels: int = 1
    base_channels: int = 64        # 64 -> 128 -> 256 -> 512 -> 1024
    depth: int = 4                 # number of encoder downsampling stages
    num_classes: int = 3


# ----------------------------------------------------------------------
# Tables 5 / 6 — XGBoost Hyperparameter Configuration
# ----------------------------------------------------------------------
@dataclass
class XGBoostConfig:
    learning_rate: float = 0.1     # eta
    max_depth: int = 6
    min_child_weight: int = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0.1
    reg_lambda: float = 1.0        # L2
    reg_alpha: float = 0.0         # L1
    n_estimators: int = 100
    objective: str = "binary:logistic"
    eval_metric: str = "logloss"
    random_state: int = 42


# ----------------------------------------------------------------------
# Feature extraction (Section 3.3) — 25 features total
# ----------------------------------------------------------------------
@dataclass
class FeatureConfig:
    glcm_distances: Tuple[int, ...] = (1,)
    glcm_angles: Tuple[float, ...] = (0.0, 0.785398, 1.570796, 2.356194)  # 0,45,90,135 deg
    glcm_levels: int = 256
    total_features: int = 25


# ----------------------------------------------------------------------
# Dataset config (Section 3.1.1)
# ----------------------------------------------------------------------
@dataclass
class DatasetConfig:
    n_training_volumes: int = 131
    n_testing_volumes: int = 70
    n_classification_cases: int = 150   # LiTS + 3Dircadb consolidated
    n_benign: int = 70
    n_malignant: int = 80
    n_cv_folds: int = 5
    bootstrap_iterations: int = 1000
    random_state: int = 42


preprocess_cfg = PreprocessConfig()
unet_train_cfg = UNetTrainConfig()
unet_arch_cfg = UNetArchConfig()
xgb_cfg = XGBoostConfig()
feature_cfg = FeatureConfig()
dataset_cfg = DatasetConfig()
