#!/usr/bin/env python
"""
Full recovery audit: per-ROI (lesion) AND healthy-WM recovery, for both protocols.

For each pathology x protocol it reports, per parameter:
  - lesion recovery   (roi_labels > 0): bias / MAE / RMSE   -> this is Table 5
  - healthy recovery  (roi_labels == 0): bias / MAE / RMSE   -> baseline / leakage
Ground truth read from *_gt.nii (so healthy v_glia truth = 0, etc.).

Median-map volume order (confirmed): [v_ic, ODI, v_glia, v_iso, ...]
Novel protocol is skipped automatically if its full median map is absent.
"""

import os
import numpy as np
import nibabel as nib

BASE = "results/glia"
PROTOCOLS = ["hcp", "novel"]
PATHOLOGIES = ["astrogliosis", "edema"]
PARAMS = [("v_ic", 0, "vic"), ("ODI", 1, "odi"),
          ("v_glia", 2, "vglia"), ("v_iso", 3, "viso")]


def load_slice(path):
    arr = nib.load(path).get_fdata()
    return np.squeeze(arr, axis=2) if arr.shape[2] == 1 else arr


def stats(est, truth):
    err = est - truth
    return (float(np.mean(err)), float(np.mean(np.abs(err))),
            float(np.sqrt(np.mean(err ** 2))))


def report(pathology, protocol):
    d = os.path.join(BASE, pathology)
    median_path = os.path.join(d, f"phantom_{pathology}_{protocol}_mcmc_median.nii")
    if not os.path.exists(median_path):
        print(f"\n[skip] {pathology} {protocol}: no full median map (job still running?)")
        return

    median = load_slice(median_path)
    labels = load_slice(os.path.join(d, f"phantom_{pathology}_roi_labels.nii"))
    wm = median[..., 0] != 0
    lesion = (labels > 0) & wm
    healthy = (labels == 0) & wm

    print(f"\n=== {pathology}  ({protocol.upper()})  "
          f"lesion n={int(lesion.sum())}, healthy n={int(healthy.sum())} ===")
    print(f"{'Param':<8}{'Truth':>7} | "
          f"{'L-bias':>8}{'L-MAE':>7}{'L-RMSE':>8} | "
          f"{'H-bias':>8}{'H-MAE':>7}{'H-RMSE':>8}")
    for name, idx, token in PARAMS:
        emap = median[..., idx]
        tmap = load_slice(os.path.join(d, f"phantom_{pathology}_{token}_gt.nii"))
        lb, lm, lr = stats(emap[lesion], tmap[lesion])
        hb, hm, hr = stats(emap[healthy], tmap[healthy])
        true_les = float(np.mean(tmap[lesion]))
        print(f"{name:<8}{true_les:>7.3f} | "
              f"{lb:>+8.3f}{lm:>7.3f}{lr:>8.3f} | "
              f"{hb:>+8.3f}{hm:>7.3f}{hr:>8.3f}")


if __name__ == "__main__":
    for path in PATHOLOGIES:
        for proto in PROTOCOLS:
            report(path, proto)
    print("\nL = lesion ROI (Table 5).  H = healthy WM (baseline / spurious signal).")
