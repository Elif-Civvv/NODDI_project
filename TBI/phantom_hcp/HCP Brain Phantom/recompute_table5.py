#!/usr/bin/env python
"""
Recompute Table 5: per-ROI recovery (bias, MAE, RMSE) for the HCP protocol,
read from the full-WM posterior-median maps.

Lesion voxels are roi_labels > 0 (the same voxels as the per-ROI fit), so this
should reproduce the existing Table 5 numbers. Ground truth is read from the
*_gt.nii files rather than hard-coded, so any mismatch is caught.

Median-map volume order (confirmed): [v_ic, ODI, v_glia, v_iso, T2_t, ...]
  idx 0 = v_ic, idx 1 = ODI, idx 2 = v_glia, idx 3 = v_iso
"""

import os
import numpy as np
import nibabel as nib

BASE = "results/glia"
PROTOCOL = "hcp"
PATHOLOGIES = ["astrogliosis", "edema"]

# parameter name -> (median-map volume index, gt-file token)
PARAMS = [
    ("v_ic",   0, "vic"),
    ("ODI",    1, "odi"),
    ("v_glia", 2, "vglia"),
    ("v_iso",  3, "viso"),
]


def load_slice(path):
    arr = nib.load(path).get_fdata()
    return np.squeeze(arr, axis=2) if arr.shape[2] == 1 else arr


def run(pathology):
    d = os.path.join(BASE, pathology)
    median = load_slice(os.path.join(d, f"phantom_{pathology}_{PROTOCOL}_mcmc_median.nii"))
    labels = load_slice(os.path.join(d, f"phantom_{pathology}_roi_labels.nii"))

    wm = median[..., 0] != 0
    lesion = (labels > 0) & wm
    n = int(lesion.sum())

    print(f"\n=== {pathology}  (HCP, n={n} lesion voxels) ===")
    print(f"{'Param':<8}{'Truth':>8}{'Bias':>9}{'MAE':>8}{'RMSE':>8}")
    for name, idx, token in PARAMS:
        est = median[..., idx][lesion]
        gt_path = os.path.join(d, f"phantom_{pathology}_{token}_gt.nii")
        truth_map = load_slice(gt_path)
        truth = truth_map[lesion]

        # gt may be constant in ROI; report the mean true value for the row
        true_val = float(np.mean(truth))
        err = est - truth
        bias = float(np.mean(err))
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        print(f"{name:<8}{true_val:>8.3f}{bias:>+9.3f}{mae:>8.3f}{rmse:>8.3f}")


if __name__ == "__main__":
    for p in PATHOLOGIES:
        run(p)
    print("\nCompare against the existing Table 5; numbers should match the per-ROI fit.")
