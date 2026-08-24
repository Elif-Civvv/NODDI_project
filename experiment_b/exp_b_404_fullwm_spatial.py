"""
Spatial discrimination maps for the glial-fraction detector.

Reads the fitted posterior-median maps (NIfTI, 6 parameter volumes) and, for each
pathology, produces ONE figure with three spatial panels over the white-matter slice:

  - TRUE ROI (ground truth, from roi_labels)
  - PREDICTED lesion under the gaNODDI (HCP) protocol  (median v_glia >= threshold)
  - PREDICTED lesion under the novel protocol          (median v_glia >= threshold)

The per-protocol Youden-optimal threshold is computed internally (lesion vs all
healthy WM voxels) so the predicted maps are comparable to the previous ROC version,
but no ROC curve is plotted.

v_glia is volume index 2 in the median map (healthy ~0.03, astro-ROI elevated).
NOTE on edema: true v_glia is 0 everywhere, so any predicted "lesion" here is the model
leaking free-water signal into the glial channel -- a DETECTABILITY result, not evidence
of a real glial population. The edema panels are labelled to make this explicit.
"""

import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_curve

WM_COLOR = "#ebebeb"      # white-matter backdrop
LESION_COLOR = "#0000d9"  # true ROI + predicted lesions

BASE = "results/glia"
VGLIA_IDX = 2                      # v_glia volume in the median map
PATHOLOGIES = ["edema", "astrogliosis"]
PROTOCOLS = ["hcp", "novel"]       # hcp == gaNODDI; novel == novel protocol
PROTO_LABELS = {"hcp": "gaNODDI (HCP)", "novel": "Novel protocol"}
OUTDIR = "results/glia/fullwm_spatial"
os.makedirs(OUTDIR, exist_ok=True)


def load_slice(path):
    """Load a NIfTI and squeeze the singleton z-axis -> 2D (or 3D if param axis)."""
    arr = nib.load(path).get_fdata()
    return np.squeeze(arr, axis=2) if arr.shape[2] == 1 else arr


def predicted_lesion(median_path, labels, wm):
    """Return predicted-lesion mask at the Youden-optimal v_glia threshold."""
    median = load_slice(median_path)          # (X, Y, 6)
    vglia = median[..., VGLIA_IDX]
    lesion = (labels > 0) & wm

    scores = vglia[wm]
    truth = lesion[wm].astype(int)            # 1 = lesion, 0 = healthy WM
    fpr, tpr, thr = roc_curve(truth, scores)
    opt_thr = thr[np.argmax(tpr - fpr)]

    return (vglia >= opt_thr) & wm, opt_thr


def run(pathology):
    d = os.path.join(BASE, pathology)
    label_path = os.path.join(d, f"phantom_{pathology}_roi_labels.nii")
    labels = load_slice(label_path)           # (X, Y), values {0,1,2,3}

    # white matter / true ROI defined from the first available median map's v_ic
    medians = {p: os.path.join(d, f"phantom_{pathology}_{p}_mcmc_median.nii")
               for p in PROTOCOLS}
    avail = {p: m for p, m in medians.items() if os.path.exists(m)}
    if not avail:
        print(f"[skip] {pathology:13s}: no median maps found")
        return

    ref = load_slice(next(iter(avail.values())))
    wm = ref[..., 0] != 0                      # v_ic nonzero == white matter
    lesion = (labels > 0) & wm

    # ---- figure: True ROI + predicted lesion per protocol ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    wm_cmap = ListedColormap([WM_COLOR])
    lesion_cmap = ListedColormap([LESION_COLOR])

    def show(axis, mask, title):
        bg = np.where(wm.T, 0.0, np.nan)       # white-matter backdrop, anterior up
        axis.imshow(bg, cmap=wm_cmap, vmin=0, vmax=1, origin="lower")
        axis.imshow(np.where(mask.T, 0.0, np.nan), cmap=lesion_cmap,
                    vmin=0, vmax=1, origin="lower")
        axis.set_title(title)
        axis.axis("off")

    show(ax[0], lesion, "True ROI (ground truth)")

    for axis, proto in zip(ax[1:], PROTOCOLS):
        label = PROTO_LABELS[proto]
        if proto not in avail:
            axis.text(0.5, 0.5, f"{label}\n(map missing)", ha="center",
                      va="center", transform=axis.transAxes)
            axis.axis("off")
            print(f"[skip] {pathology:13s} {proto:5s}: no median map")
            continue
        pred, opt_thr = predicted_lesion(avail[proto], labels, wm)
        show(axis, pred, f"Predicted: {label}\n(v_glia >= {opt_thr:.3f})")
        print(f"[ok]   {pathology:13s} {proto:5s}: thr={opt_thr:.3f}")

    note = ""
    if pathology == "edema":
        note = ("\nNOTE: true v_glia = 0 everywhere; predicted 'lesion' is "
                "free-water leakage into the glial channel, not real glia.")
    fig.suptitle(f"{pathology.capitalize()} - spatial discrimination{note}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUTDIR, f"fullwm_spatial_{pathology}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"       -> {out}")


if __name__ == "__main__":
    for path in PATHOLOGIES:
        run(path)
    print("\nDone. Figures in", OUTDIR)

