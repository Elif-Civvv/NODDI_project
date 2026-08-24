"""
Full-white-matter ROC and spatial discrimination for the glial-fraction detector.

Reads the fitted posterior-median maps (NIfTI, 6 parameter volumes) and scores
every lesion voxel against ALL healthy white-matter voxels in the slice -- a more
stringent test than the per-ROI version that sampled only ~300 healthy negatives.

For each pathology x protocol it produces:
  - ROC over the full WM slice (lesion vs all healthy WM), AUC, Youden-optimal threshold
  - false-positive rate at that threshold, with healthy-voxel counts
  - a two-panel spatial figure: TRUE ROI (from roi_labels) vs PREDICTED lesion
    (median v_glia >= threshold), i.e. where the model thinks the ROI is vs where it is.

v_glia is volume index 2 in the median map (confirmed: healthy ~0.03, astro-ROI elevated).
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
from sklearn.metrics import roc_curve, auc

BASE = "results/glia"
VGLIA_IDX = 2                      # v_glia volume in the median map
PATHOLOGIES = ["edema", "astrogliosis"]
PROTOCOLS = ["hcp", "novel"]       # novel skipped automatically if map missing
OUTDIR = "results/glia/fullwm_roc"
os.makedirs(OUTDIR, exist_ok=True)


def load_slice(path):
    """Load a NIfTI and squeeze the singleton z-axis -> 2D (or 3D if param axis)."""
    arr = nib.load(path).get_fdata()
    return np.squeeze(arr, axis=2) if arr.shape[2] == 1 else arr


def run(pathology, protocol):
    d = os.path.join(BASE, pathology)
    median_path = os.path.join(d, f"phantom_{pathology}_{protocol}_mcmc_median.nii")
    label_path = os.path.join(d, f"phantom_{pathology}_roi_labels.nii")

    if not os.path.exists(median_path):
        print(f"[skip] {pathology:13s} {protocol:5s}: no full median map "
              f"(job likely still running -- only _roi exists)")
        return None

    median = load_slice(median_path)          # (X, Y, 6)
    labels = load_slice(label_path)           # (X, Y), values {0,1,2,3}

    vglia = median[..., VGLIA_IDX]
    wm = median[..., 0] != 0                  # v_ic nonzero == white matter
    lesion = (labels > 0) & wm
    healthy = (labels == 0) & wm

    scores = vglia[wm]
    truth = lesion[wm].astype(int)            # 1 = lesion, 0 = healthy WM

    fpr, tpr, thr = roc_curve(truth, scores)
    roc_auc = auc(fpr, tpr)
    youden = np.argmax(tpr - fpr)
    opt_thr = thr[youden]

    pred = (vglia >= opt_thr) & wm
    n_healthy = int(healthy.sum())
    n_fp = int((pred & healthy).sum())
    fp_rate = n_fp / n_healthy if n_healthy else float("nan")
    med_lesion = float(np.median(vglia[lesion]))
    med_healthy = float(np.median(vglia[healthy]))

    print(f"[ok]   {pathology:13s} {protocol:5s}: "
          f"AUC={roc_auc:.3f}  thr={opt_thr:.3f}  "
          f"FPR={fp_rate:.3f} ({n_fp}/{n_healthy})  "
          f"median vglia lesion={med_lesion:.3f} healthy={med_healthy:.3f}")

    # ---- figure: ROC + true ROI + predicted lesion ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].plot(fpr, tpr, color="C3", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax[0].plot([0, 1], [0, 1], "--", color="navy", lw=1)
    ax[0].scatter([fpr[youden]], [tpr[youden]], color="red", zorder=5,
                  label=f"thr = {opt_thr:.3f}")
    ax[0].set_xlabel("False positive rate")
    ax[0].set_ylabel("True positive rate")
    ax[0].set_title(f"Full-WM ROC: {pathology} ({protocol.upper()})")
    ax[0].legend(loc="lower right")
    ax[0].set_aspect("equal")

    # orient for display: transpose so anterior is up, consistent with thesis figs
    def show(a, title, cmap, vmax=None):
        ax_ = ax[1] if "True" in title else ax[2]
        bg = np.where(wm.T, 0.15, np.nan)
        ax_.imshow(bg, cmap="Greys", vmin=0, vmax=1, origin="lower")
        ax_.imshow(np.where(a.T, 1.0, np.nan), cmap=cmap, vmin=0, vmax=1,
                   origin="lower")
        ax_.set_title(title)
        ax_.axis("off")

    show(lesion, "True ROI (ground truth)", "Blues")
    pred_title = "Predicted lesion (v_glia >= thr)"
    show(pred, pred_title, "Reds")

    note = ""
    if pathology == "edema":
        note = ("\nNOTE: true v_glia = 0 everywhere; predicted 'lesion' is "
                "free-water leakage into the glial channel, not real glia.")
    fig.suptitle(f"{pathology.capitalize()} - {protocol.upper()} protocol{note}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUTDIR, f"fullwm_roc_{pathology}_{protocol}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"       -> {out}")

    return dict(pathology=pathology, protocol=protocol, auc=roc_auc,
                thr=opt_thr, fpr=fp_rate, n_fp=n_fp, n_healthy=n_healthy,
                med_lesion=med_lesion, med_healthy=med_healthy)


if __name__ == "__main__":
    results = []
    for path in PATHOLOGIES:
        for proto in PROTOCOLS:
            r = run(path, proto)
            if r:
                results.append(r)
    print("\nDone. Figures + this summary in", OUTDIR)
