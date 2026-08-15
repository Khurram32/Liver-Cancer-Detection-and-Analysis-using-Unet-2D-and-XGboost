"""
Section 3.1.2 — Image Preprocessing

Implements the strict-order preprocessing pipeline described in the paper:
  1. Windowing (HU)                       -> apply_windowing
  2. Normalization (Eq. 1)                -> normalize_intensity
  3. Noise reduction (median, Eq. 2)      -> median_denoise
  4. Contrast enhancement (Eq. 3)         -> histogram_equalize
  5. Resizing and augmentation            -> resize_image, augment
"""

import numpy as np
import cv2

from config import preprocess_cfg


def apply_windowing(hu_image: np.ndarray, window_level: float = None,
                     window_width: float = None) -> np.ndarray:
    """Step 1: Hounsfield Unit windowing.

    WL=40, WW=400 (standard abdominal soft-tissue window), as specified
    in Section 3.1.2 item 1 of the paper.
    """
    wl = preprocess_cfg.window_level if window_level is None else window_level
    ww = preprocess_cfg.window_width if window_width is None else window_width

    lower = wl - ww / 2.0
    upper = wl + ww / 2.0
    windowed = np.clip(hu_image, lower, upper)
    return windowed


def normalize_intensity(image: np.ndarray, mean: float = None,
                         std: float = None) -> np.ndarray:
    """Step 2: Eq. (1) — I_norm = (I - mu) / sigma."""
    mu = image.mean() if mean is None else mean
    sigma = image.std() if std is None else std
    sigma = sigma if sigma > 1e-8 else 1e-8
    return (image - mu) / sigma


def median_denoise(image: np.ndarray, kernel_size: int = None) -> np.ndarray:
    """Step 3: Eq. (2) — median filtering, kernel 3x3, edge-preserving."""
    k = preprocess_cfg.median_kernel_size if kernel_size is None else kernel_size
    # cv2.medianBlur requires float32 or uint8; kernel size must be odd
    img32 = image.astype(np.float32)
    return cv2.medianBlur(img32, k)


def histogram_equalize(image: np.ndarray, num_gray_levels: int = None) -> np.ndarray:
    """Step 4: Eq. (3) — histogram equalization for contrast enhancement.

    I_enhanced(i,j) = (L-1)/(M*N) * sum_{k=0}^{I(i,j)} h(k)
    """
    L = preprocess_cfg.num_gray_levels if num_gray_levels is None else num_gray_levels

    # Rescale to [0, L-1] uint8 domain to compute the histogram/CDF
    img_min, img_max = image.min(), image.max()
    if img_max - img_min < 1e-8:
        return image.copy()
    scaled = (image - img_min) / (img_max - img_min) * (L - 1)
    scaled_uint8 = scaled.astype(np.uint8)

    M, N = image.shape[:2]
    hist, _ = np.histogram(scaled_uint8.flatten(), bins=L, range=(0, L - 1))
    cdf = np.cumsum(hist)

    enhanced_lut = (L - 1) / (M * N) * cdf
    enhanced = enhanced_lut[scaled_uint8]
    return enhanced.astype(np.float32)


def resize_image(image: np.ndarray, size=None, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    """Step 5a: Resize to 512x512 (paper's fixed input resolution)."""
    target = preprocess_cfg.image_size if size is None else size
    return cv2.resize(image, target, interpolation=interpolation)


def random_rotation(image: np.ndarray, mask: np.ndarray = None, max_deg: float = None):
    """Random rotation in [-max_deg, +max_deg]."""
    deg = preprocess_cfg.rotation_deg if max_deg is None else max_deg
    angle = np.random.uniform(-deg, deg)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rot_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
    rot_mask = None
    if mask is not None:
        rot_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_REFLECT)
    return rot_img, rot_mask


def random_horizontal_flip(image: np.ndarray, mask: np.ndarray = None):
    flipped_img = np.fliplr(image).copy()
    flipped_mask = np.fliplr(mask).copy() if mask is not None else None
    return flipped_img, flipped_mask


def elastic_deformation(image: np.ndarray, mask: np.ndarray = None,
                         alpha: float = None, sigma: float = None):
    """Elastic deformation augmentation (Simard-style), per Section 3.1.2 item 5."""
    from scipy.ndimage import gaussian_filter, map_coordinates

    a = preprocess_cfg.elastic_alpha if alpha is None else alpha
    s = preprocess_cfg.elastic_sigma if sigma is None else sigma

    shape = image.shape[:2]
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), s, mode="constant") * a
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), s, mode="constant") * a

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)))

    deformed_img = map_coordinates(image, indices, order=1, mode="reflect").reshape(shape)
    deformed_mask = None
    if mask is not None:
        deformed_mask = map_coordinates(mask, indices, order=0, mode="reflect").reshape(shape)
    return deformed_img, deformed_mask


def augment(image: np.ndarray, mask: np.ndarray = None, prob: float = None):
    """Step 5b: On-the-fly augmentation, applied independently with prob p=0.5
    per technique, per image (rotation, horizontal flip, elastic deformation).
    """
    p = preprocess_cfg.augmentation_prob if prob is None else prob

    if np.random.rand() < p:
        image, mask = random_rotation(image, mask)
    if np.random.rand() < p:
        image, mask = random_horizontal_flip(image, mask)
    if np.random.rand() < p:
        image, mask = elastic_deformation(image, mask)
    return image, mask


def preprocess_slice(hu_slice: np.ndarray, apply_augmentation: bool = False,
                      mask: np.ndarray = None):
    """Full preprocessing pipeline in the strict order specified by the paper:

    windowing -> normalization -> median denoise -> histogram equalization
    -> resize -> (optional) augmentation
    """
    img = apply_windowing(hu_slice)
    img = normalize_intensity(img)
    img = median_denoise(img)
    img = histogram_equalize(img)
    img = resize_image(img)

    resized_mask = None
    if mask is not None:
        resized_mask = resize_image(mask.astype(np.float32), interpolation=cv2.INTER_NEAREST)
        resized_mask = resized_mask.astype(np.int64)

    if apply_augmentation:
        img, resized_mask = augment(img, resized_mask)

    return img, resized_mask
