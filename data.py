"""
Section 3.1.1 — Dataset Description and Label Clarification

Real LiTS2017 volumes are NIfTI (.nii/.nii.gz) files: `volume-XX.nii` (CT) and
`segmentation-XX.nii` (mask with labels 0=background, 1=liver, 2=tumor).
`LiTSVolumeDataset` reads these directly if a data directory is supplied.

Since the raw LiTS/3Dircadb files are large, gated downloads and are not
available in this environment, `SyntheticLiverCTDataset` also provides a
structurally faithful synthetic stand-in (same 512x512, 3-class label scheme,
same size distribution used in Table 8) so the full pipeline can be run,
tested, and demonstrated end-to-end without external data.
"""

import os
import glob
import numpy as np
from torch.utils.data import Dataset

from preprocessing import preprocess_slice
from config import preprocess_cfg


# ----------------------------------------------------------------------
# Real LiTS-format loader
# ----------------------------------------------------------------------
class LiTSVolumeDataset(Dataset):
    """Loads (CT slice, mask slice) pairs from LiTS-format NIfTI volumes.

    Expected directory layout:
        data_dir/volume-0.nii(.gz), data_dir/segmentation-0.nii(.gz), ...

    Labels in the segmentation mask: 0=background, 1=liver, 2=tumor,
    matching the paper's C=3 output classes (Eq. 7).
    """

    def __init__(self, data_dir: str, augment: bool = False, liver_only_slices: bool = True):
        import nibabel as nib
        self.nib = nib
        self.data_dir = data_dir
        self.augment = augment

        vol_paths = sorted(glob.glob(os.path.join(data_dir, "volume-*.nii*")))
        self.index = []  # list of (volume_path, seg_path, slice_idx)

        for vp in vol_paths:
            case_id = os.path.basename(vp).split("volume-")[-1].split(".nii")[0]
            sp = None
            for ext in [".nii.gz", ".nii"]:
                candidate = os.path.join(data_dir, f"segmentation-{case_id}{ext}")
                if os.path.exists(candidate):
                    sp = candidate
                    break
            if sp is None:
                continue

            seg_img = nib.load(sp)
            n_slices = seg_img.shape[2]
            seg_data = seg_img.get_fdata() if not liver_only_slices else None

            for s in range(n_slices):
                if liver_only_slices:
                    if seg_data is None:
                        seg_data = seg_img.get_fdata()
                    if seg_data[:, :, s].max() == 0:
                        continue
                self.index.append((vp, sp, s))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        vp, sp, s = self.index[idx]
        ct_slice = self.nib.load(vp).dataobj[:, :, s].astype(np.float32)
        mask_slice = self.nib.load(sp).dataobj[:, :, s].astype(np.int64)
        img, mask = preprocess_slice(ct_slice, apply_augmentation=self.augment, mask=mask_slice)
        return img[None, ...].astype(np.float32), mask.astype(np.int64)


# ----------------------------------------------------------------------
# Synthetic data generator (structurally faithful stand-in)
# ----------------------------------------------------------------------
def _make_synthetic_ct_slice(rng: np.random.Generator, size=(512, 512),
                              tumor_size_category: str = None):
    """Generates one synthetic abdominal CT slice with a liver region and
    an optional tumor lesion, in Hounsfield Units.

    Approximate HU ranges used: background air/tissue ~ -1000 to 0,
    liver parenchyma ~ 40-70 HU, tumor lesion ~ 10-40 HU (hypodense) or
    70-110 HU (hyperdense), consistent with typical contrast-enhanced CT.
    """
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2 + rng.uniform(-30, 30), w / 2 + rng.uniform(-30, 30)

    # background body tissue
    image = rng.normal(-20, 15, size=size)

    # liver: irregular ellipse blob
    liver_a = rng.uniform(140, 180)
    liver_b = rng.uniform(110, 150)
    angle = rng.uniform(0, np.pi)
    xr = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
    yr = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
    liver_mask = ((xr / liver_a) ** 2 + (yr / liver_b) ** 2) <= 1.0

    liver_hu = rng.normal(55, 8, size=size)
    image = np.where(liver_mask, liver_hu, image)

    tumor_mask = np.zeros(size, dtype=bool)
    if tumor_size_category is not None:
        size_ranges_cm = {"small": (0.5, 2.0), "medium": (2.0, 5.0), "large": (5.0, 8.0)}
        lo, hi = size_ranges_cm[tumor_size_category]
        diameter_cm = rng.uniform(lo, hi)
        # arbitrary pixel-per-cm scale for the synthetic slice
        radius_px = max(3, (diameter_cm / 2.0) * 15.0)

        # place tumor center inside the liver region
        liver_ys, liver_xs = np.where(liver_mask)
        if len(liver_ys) > 0:
            pick = rng.integers(0, len(liver_ys))
            tcy, tcx = liver_ys[pick], liver_xs[pick]
            dist = np.sqrt((yy - tcy) ** 2 + (xx - tcx) ** 2)
            # irregular boundary via sinusoidal perturbation
            theta = np.arctan2(yy - tcy, xx - tcx)
            perturb = 1.0 + 0.15 * np.sin(4 * theta + rng.uniform(0, 2 * np.pi))
            tumor_mask = dist <= (radius_px * perturb)
            tumor_mask &= liver_mask

            tumor_hu = rng.normal(rng.choice([25, 90]), 10, size=size)
            image = np.where(tumor_mask, tumor_hu, image)

    label_mask = np.zeros(size, dtype=np.int64)
    label_mask[liver_mask] = 1
    label_mask[tumor_mask] = 2

    return image.astype(np.float32), label_mask


class SyntheticLiverCTDataset(Dataset):
    """Synthetic dataset matching the paper's 3-class segmentation scheme and
    Table 8 tumor-size distribution (18 small / 32 medium / 20 large), used
    to run and validate the full pipeline without the real LiTS/3Dircadb data.
    """

    SIZE_DISTRIBUTION = {"small": 18, "medium": 32, "large": 20}

    def __init__(self, n_samples: int = 70, augment: bool = False, seed: int = 42,
                 include_no_tumor_fraction: float = 0.1):
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.n_samples = n_samples

        categories = []
        total_labeled = sum(self.SIZE_DISTRIBUTION.values())
        for cat, count in self.SIZE_DISTRIBUTION.items():
            frac = count / total_labeled
            categories += [cat] * max(1, int(round(frac * n_samples * (1 - include_no_tumor_fraction))))
        n_no_tumor = max(0, n_samples - len(categories))
        categories += [None] * n_no_tumor
        self.rng.shuffle(categories)
        self.categories = categories[:n_samples]
        while len(self.categories) < n_samples:
            self.categories.append(self.rng.choice(["small", "medium", "large"]))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        cat = self.categories[idx]
        seed = int(self.rng.integers(0, 1_000_000)) + idx
        local_rng = np.random.default_rng(seed)
        ct_slice, mask = _make_synthetic_ct_slice(local_rng, tumor_size_category=cat)
        img, mask_resized = preprocess_slice(ct_slice, apply_augmentation=self.augment, mask=mask)
        return img[None, ...].astype(np.float32), mask_resized.astype(np.int64)

    def get_category(self, idx):
        return self.categories[idx]


if __name__ == "__main__":
    ds = SyntheticLiverCTDataset(n_samples=5)
    for i in range(len(ds)):
        img, mask = ds[i]
        print(f"sample {i}: img={img.shape}, mask={mask.shape}, "
              f"category={ds.get_category(i)}, unique_labels={np.unique(mask)}")
