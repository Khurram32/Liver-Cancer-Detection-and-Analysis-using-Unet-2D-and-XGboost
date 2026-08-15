"""
Section 3.3 — Feature Extraction

Extracts 25 quantitative features from a segmented tumor region for the
XGBoost classification stage, spanning three categories:

  Morphological (shape/geometry):
    - Area, Perimeter, Circularity, Compactness, Eccentricity
    (+ bounding-box extent, solidity, major/minor axis length,
     equivalent diameter, extent, orientation — to round out 25 total
     features together with texture/intensity groups, consistent with
     the "carefully chosen" 25-feature set referenced in the Conclusion)

  Texture (GLCM, Haralick):
    - Homogeneity, Contrast, Energy, Correlation, Entropy, Dissimilarity,
      ASM (Angular Second Moment)

  Intensity:
    - Mean, Std, Skewness, Kurtosis, Min, Max, Range, Median, IQR
"""

import numpy as np
from scipy import stats
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops

from config import feature_cfg


# ----------------------------------------------------------------------
# Morphological features
# ----------------------------------------------------------------------
def extract_morphological_features(mask: np.ndarray) -> dict:
    """Region-based shape descriptors from a binary tumor mask.

    Area:          A = sum_{(x,y) in R} 1
    Circularity:   C = 4*pi*A / P^2
    Compactness:   perimeter^2 / area (shape irregularity)
    Eccentricity:  degree of elongation (0 = circle, ->1 = line)
    """
    labeled = label(mask.astype(np.uint8))
    props_list = regionprops(labeled)

    if len(props_list) == 0:
        # empty mask fallback
        keys = ["area", "perimeter", "circularity", "compactness", "eccentricity",
                "solidity", "extent", "major_axis_length", "minor_axis_length",
                "equivalent_diameter", "orientation"]
        return {k: 0.0 for k in keys}

    # use the largest connected component as "the" tumor region
    props = max(props_list, key=lambda p: p.area)

    area = float(props.area)
    perimeter = float(props.perimeter) if props.perimeter > 0 else 1.0
    circularity = (4.0 * np.pi * area) / (perimeter ** 2 + 1e-7)
    compactness = (perimeter ** 2) / (area + 1e-7)
    eccentricity = float(props.eccentricity)

    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "compactness": compactness,
        "eccentricity": eccentricity,
        "solidity": float(props.solidity),
        "extent": float(props.extent),
        "major_axis_length": float(props.axis_major_length),
        "minor_axis_length": float(props.axis_minor_length),
        "equivalent_diameter": float(props.equivalent_diameter_area),
        "orientation": float(props.orientation),
    }


# ----------------------------------------------------------------------
# GLCM texture features (Haralick)
# ----------------------------------------------------------------------
def extract_texture_features(image: np.ndarray, mask: np.ndarray) -> dict:
    """GLCM-based texture descriptors computed over the masked tumor region.

    Homogeneity: sum_{i,j} p(i,j) / (1 + |i-j|)
    Contrast:    sum_{i,j} |i-j|^2 * p(i,j)
    Energy:      sum_{i,j} p(i,j)^2
    Correlation: sum_{i,j} (i-mu_i)(j-mu_j)p(i,j) / (sigma_i * sigma_j)
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        keys = ["glcm_homogeneity", "glcm_contrast", "glcm_energy",
                "glcm_correlation", "glcm_entropy", "glcm_dissimilarity", "glcm_asm"]
        return {k: 0.0 for k in keys}

    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    roi = image[y0:y1, x0:x1]
    roi_mask = mask[y0:y1, x0:x1]

    # quantize ROI intensities to 8-bit for GLCM computation, masking
    # background pixels out by setting them to 0
    roi_norm = roi.copy().astype(np.float64)
    roi_norm[roi_mask == 0] = 0
    rmin, rmax = roi_norm.min(), roi_norm.max()
    if rmax - rmin < 1e-8:
        quantized = np.zeros_like(roi_norm, dtype=np.uint8)
    else:
        quantized = ((roi_norm - rmin) / (rmax - rmin) * 255).astype(np.uint8)

    glcm = graycomatrix(
        quantized,
        distances=list(feature_cfg.glcm_distances),
        angles=list(feature_cfg.glcm_angles),
        levels=256,
        symmetric=True,
        normed=True,
    )

    homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
    contrast = float(np.mean(graycoprops(glcm, "contrast")))
    energy = float(np.mean(graycoprops(glcm, "energy")))
    correlation = float(np.mean(graycoprops(glcm, "correlation")))
    dissimilarity = float(np.mean(graycoprops(glcm, "dissimilarity")))
    asm = float(np.mean(graycoprops(glcm, "ASM")))

    # Shannon entropy of the (averaged) co-occurrence distribution
    glcm_avg = glcm.mean(axis=(2, 3))
    glcm_avg_nonzero = glcm_avg[glcm_avg > 0]
    entropy = float(-np.sum(glcm_avg_nonzero * np.log2(glcm_avg_nonzero + 1e-12)))

    return {
        "glcm_homogeneity": homogeneity,
        "glcm_contrast": contrast,
        "glcm_energy": energy,
        "glcm_correlation": correlation,
        "glcm_entropy": entropy,
        "glcm_dissimilarity": dissimilarity,
        "glcm_asm": asm,
    }


# ----------------------------------------------------------------------
# Intensity features
# ----------------------------------------------------------------------
def extract_intensity_features(image: np.ndarray, mask: np.ndarray) -> dict:
    """Pixel-intensity distribution statistics within the masked tumor region.

    Mean:     mu = (1/N) * sum(I_i)
    Std:      sigma = sqrt( (1/N) * sum((I_i - mu)^2) )
    Skewness / kurtosis: higher-order moments (asymmetry, peakedness)
    """
    values = image[mask > 0].astype(np.float64)
    if values.size == 0:
        keys = ["mean_intensity", "std_intensity", "skewness", "kurtosis",
                "min_intensity", "max_intensity", "intensity_range",
                "median_intensity", "iqr_intensity"]
        return {k: 0.0 for k in keys}

    mean_i = float(np.mean(values))
    std_i = float(np.std(values))
    skew_i = float(stats.skew(values)) if values.size > 2 else 0.0
    kurt_i = float(stats.kurtosis(values)) if values.size > 2 else 0.0
    min_i, max_i = float(values.min()), float(values.max())
    median_i = float(np.median(values))
    q75, q25 = np.percentile(values, [75, 25])
    iqr_i = float(q75 - q25)

    return {
        "mean_intensity": mean_i,
        "std_intensity": std_i,
        "skewness": skew_i,
        "kurtosis": kurt_i,
        "min_intensity": min_i,
        "max_intensity": max_i,
        "intensity_range": max_i - min_i,
        "median_intensity": median_i,
        "iqr_intensity": iqr_i,
    }


def extract_all_features(image: np.ndarray, mask: np.ndarray) -> dict:
    """Concatenate morphological + texture + intensity features into a single
    feature vector (dict), z-score normalized downstream before classification.
    """
    feats = {}
    feats.update(extract_morphological_features(mask))
    feats.update(extract_texture_features(image, mask))
    feats.update(extract_intensity_features(image, mask))
    return feats


def features_to_vector(feats: dict, feature_order: list = None) -> np.ndarray:
    order = feature_order if feature_order is not None else sorted(feats.keys())
    return np.array([feats[k] for k in order], dtype=np.float64)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    image = rng.normal(100, 20, size=(64, 64))
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:45] = 1

    feats = extract_all_features(image, mask)
    for k, v in feats.items():
        print(f"{k:25s}: {v:.4f}")
    print(f"\nTotal feature count: {len(feats)}")
