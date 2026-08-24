"""
visualise_roi_comparison.py
===========================
Three-panel spatial comparison per pathology, over the FULL white-matter slice:

  [ True ROI (+ WM backdrop) ]  [ NODDI predicted ]  [ Novel predicted ]

Predicted lesion = (fitted v_glia >= optimal Youden's-J threshold) within WM.
Each protocol uses its OWN optimal threshold (computed from its own ROC), matching
visualise_roi_identification.py. Reads the no-suffix full-WM maps by default
(phantom_<path>_<proto>_mcmc_median.nii); pass --roi to use the _roi maps instead.

Run:
  python visualise_roi_comparison.py \
      --root ".../results" \
      --gt-root ".../results/glia" \
      --out ".../figures"
"""

import argparse
import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

PARAM_INDEX = 2  # v_glia in [v_ic, odi, v_glia, v_iso, theta, phi]


def _find(folders, name):
    for fld in folders:
        if fld is None:
            continue
        p = os.path.join(fld, name)
        if os.path.exists(p):
            return p
    return None


def load_map(path):
    return nib.load(path).get_fdata()[:, :, 0, :]  # (X, Y, 6)


def _load_wm(gt_search, shape):
    wm_path = _find(gt_search, "wm_mask_axial_z69.nii")
    if wm_path is None:
        for alt in ("wm_mask.nii", "wm_mask_axial.nii"):
            wm_path = _find(gt_search, alt)
            if wm_path:
                break
    if wm_path is None:
        return None
    wm = nib.load(wm_path).get_fdata()
    wm = wm[:, :, 0] if wm.ndim == 3 else wm
    if wm.shape != shape:
        return None
    return wm > 0


def optimal_threshold(est_vglia, true_lesion_mask, wm_mask):
    """Youden's-J optimal v_glia threshold over WM voxels."""
    y_true = true_lesion_mask[wm_mask].astype(int)
    y_scores = est_vglia[wm_mask]
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, thr = roc_curve(y_true, y_scores)
    return thr[np.argmax(tpr - fpr)]


def make_comparison(pathology, gt_search, map_paths, out_dir):
    """map_paths: dict {proto: path}. Builds the 3-panel figure."""
    roi_path = _find(gt_search, f"phantom_{pathology}_roi_labels.nii")
    if roi_path is None:
        print(f"  [skip] no ROI labels for {pathology}")
        return
    true_lesion_mask = nib.load(roi_path).get_fdata()[:, :, 0] > 0

    # Reference shape/WM from whichever protocol map we have.
    ref_proto = next(iter(map_paths))
    ref_med = load_map(map_paths[ref_proto])
    wm_mask = _load_wm(gt_search, ref_med.shape[:2])
    if wm_mask is None:
        wm_mask = np.any(ref_med != 0, axis=-1)

    panels = [("True ROI (ground truth)", None, None)]
    for proto in ("hcp", "novel"):
        if proto not in map_paths:
            continue
        med = load_map(map_paths[proto])
        est = med[:, :, PARAM_INDEX]
        thr = optimal_threshold(est, true_lesion_mask, wm_mask)
        if thr is None:
            print(f"  [warn] {pathology}/{proto}: no binary classes, skipping panel")
            continue
        pred = (est >= thr) & wm_mask
        disp = "NODDI" if proto == "hcp" else "Novel"
        panels.append((f"{disp} predicted\n(v_glia >= {thr:.3f})", pred, disp))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4))
    if n == 1:
        axes = [axes]

    for ax, (title, pred, _disp) in zip(axes, panels):
        # WM backdrop (grey, context only) on every panel.
        ax.imshow(np.where(wm_mask > 0, 0.85, np.nan),
                  cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        if pred is None:
            # True-ROI panel: blue fill of the actual lesion.
            ax.imshow(np.where(true_lesion_mask, 1.0, np.nan),
                      cmap='Blues', alpha=0.7, vmin=0, vmax=1,
                      interpolation='nearest')
        else:
            # Predicted panel: red fill + blue true outline for reference.
            ax.imshow(np.where(pred, 1.0, np.nan),
                      cmap='Reds', alpha=0.6, vmin=0, vmax=1,
                      interpolation='nearest')
            ax.contour(np.nan_to_num(true_lesion_mask), levels=[0.5],
                       colors='blue', linewidths=1.0)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f'{pathology.capitalize()}: true ROI versus predicted lesion '
                 f'(full white-matter slice)', fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'roi_comparison_{pathology}.jpg')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [saved] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--gt-root", default=None, dest="gt_root")
    ap.add_argument("--out", default=".")
    ap.add_argument("--roi", action="store_true",
                    help="Use _roi maps instead of full-WM (no-suffix) maps.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    suffix = "_mcmc_median_roi.nii" if args.roi else "_mcmc_median.nii"
    maps = sorted(glob.glob(
        os.path.join(args.root, "**", f"phantom_*{suffix}"), recursive=True))
    if not maps:
        print(f"No '{suffix}' maps under {args.root}. "
              f"(Full-WM run finished? Or pass --roi for ROI maps.)")
        return

    # Group maps by pathology so both protocols share one figure.
    by_path = {}
    for m in maps:
        base = os.path.basename(m)
        core = base.replace("phantom_", "").replace(suffix, "")
        proto = core.split("_")[-1]
        pathology = core[: -(len(proto) + 1)]
        by_path.setdefault(pathology, {})[proto] = m

    for pathology, map_paths in by_path.items():
        folder = os.path.dirname(next(iter(map_paths.values())))
        gt_search = [folder, args.gt_root,
                     os.path.join(args.gt_root, pathology) if args.gt_root else None]
        print(f"{pathology}: protocols found = {sorted(map_paths)}")
        make_comparison(pathology, gt_search, map_paths, args.out)


if __name__ == "__main__":
    main()
