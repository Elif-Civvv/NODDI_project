#!/usr/bin/env python
"""
Full-white-matter ROC, spatial discrimination, and confusion matrix for the
glial-fraction detector.

For each pathology x protocol it produces:
  - ROC over the full WM slice (lesion vs all healthy WM), AUC, Youden threshold
  - confusion matrix at that threshold: TP / FP / FN / TN, plus sensitivity,
    specificity, precision, FPR  (the FP count is what the Results prose needs)
  - a 4-panel figure: ROC | True ROI | Predicted lesion | confusion matrix

v_glia is volume index 2 in the median map.
NOTE on edema: true v_glia = 0 everywhere, so a predicted "lesion" is free-water
leakage into the glial channel, not real glia -- labelled on the figure.

Freshness guard: if the median map is OLDER than its checkpoint (.npy), it is
likely stale (an earlier run that a still-running job has not yet overwritten);
the script warns and skips it. Override with --allow-stale.
"""

import os
import sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

BASE = "results/glia"
VGLIA_IDX = 2
PATHOLOGIES = ["edema", "astrogliosis"]
PROTOCOLS = ["hcp", "novel"]
OUTDIR = "results/glia/fullwm_roc"
ALLOW_STALE = "--allow-stale" in sys.argv
os.makedirs(OUTDIR, exist_ok=True)


def load_slice(path):
    arr = nib.load(path).get_fdata()
    return np.squeeze(arr, axis=2) if arr.shape[2] == 1 else arr


def run(pathology, protocol):
    d = os.path.join(BASE, pathology)
    median_path = os.path.join(d, f"phantom_{pathology}_{protocol}_mcmc_median.nii")
    label_path = os.path.join(d, f"phantom_{pathology}_roi_labels.nii")
    ckpt_path = os.path.join(d, f"phantom_{pathology}_{protocol}_ckpt.npy")

    if not os.path.exists(median_path):
        print(f"[skip] {pathology:13s} {protocol:5s}: no full median map")
        return None

    # freshness guard
    if os.path.exists(ckpt_path) and not ALLOW_STALE:
        if os.path.getmtime(median_path) < os.path.getmtime(ckpt_path):
            print(f"[STALE] {pathology:13s} {protocol:5s}: median map predates "
                  f"its checkpoint -- likely an old run. Skipping "
                  f"(use --allow-stale to force).")
            return None

    median = load_slice(median_path)
    labels = load_slice(label_path)

    vglia = median[..., VGLIA_IDX]
    wm = median[..., 0] != 0
    lesion = (labels > 0) & wm
    healthy = (labels == 0) & wm

    scores = vglia[wm]
    truth = lesion[wm].astype(int)

    fpr, tpr, thr = roc_curve(truth, scores)
    roc_auc = auc(fpr, tpr)
    youden = np.argmax(tpr - fpr)
    opt_thr = thr[youden]

    pred = (vglia >= opt_thr) & wm
    TP = int((pred & lesion).sum())
    FP = int((pred & healthy).sum())
    FN = int((~pred & lesion).sum())
    TN = int((~pred & healthy).sum())

    sens = TP / (TP + FN) if (TP + FN) else float("nan")   # recall / TPR
    spec = TN / (TN + FP) if (TN + FP) else float("nan")
    prec = TP / (TP + FP) if (TP + FP) else float("nan")
    fp_rate = FP / (FP + TN) if (FP + TN) else float("nan")

    print(f"\n=== {pathology} ({protocol.upper()}) ===")
    print(f"  AUC={roc_auc:.3f}  threshold={opt_thr:.3f}")
    print(f"  TP={TP}  FP={FP}  FN={FN}  TN={TN}")
    print(f"  sensitivity={sens:.3f}  specificity={spec:.3f}  "
          f"precision={prec:.3f}  FPR={fp_rate:.3f}")

    # ---- figure ----
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    ax[0].plot(fpr, tpr, color="C3", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax[0].plot([0, 1], [0, 1], "--", color="navy", lw=1)
    ax[0].scatter([fpr[youden]], [tpr[youden]], color="red", zorder=5,
                  label=f"thr = {opt_thr:.3f}")
    ax[0].set_xlabel("False positive rate")
    ax[0].set_ylabel("True positive rate")
    ax[0].set_title(f"Full-WM ROC: {pathology}")
    ax[0].legend(loc="lower right")
    ax[0].set_aspect("equal")

    def show(panel, a, title, cmap):
        panel.imshow(np.where(wm.T, 0.15, np.nan), cmap="Greys",
                     vmin=0, vmax=1, origin="lower")
        panel.imshow(np.where(a.T, 1.0, np.nan), cmap=cmap,
                     vmin=0, vmax=1, origin="lower")
        panel.set_title(title)
        panel.axis("off")

    show(ax[1], lesion, "True ROI (ground truth)", "Blues")
    show(ax[2], pred, "Predicted lesion (v_glia >= thr)", "Reds")

    # confusion matrix panel
    cm = np.array([[TP, FN], [FP, TN]])
    ax[3].imshow(cm, cmap="Blues")
    ax[3].set_xticks([0, 1]); ax[3].set_yticks([0, 1])
    ax[3].set_xticklabels(["pred +", "pred -"])
    ax[3].set_yticklabels(["true +", "true -"])
    labels_cm = [["TP", "FN"], ["FP", "TN"]]
    for i in range(2):
        for j in range(2):
            ax[3].text(j, i, f"{labels_cm[i][j]}\n{cm[i, j]}",
                       ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black",
                       fontsize=11)
    ax[3].set_title("Confusion matrix")

    note = ""
    if pathology == "edema":
        note = ("  NOTE: true v_glia = 0 everywhere; predicted 'lesion' is "
                "free-water leakage into the glial channel, not real glia.")
    fig.suptitle(f"{pathology.capitalize()} - {protocol.upper()} protocol{note}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUTDIR, f"fullwm_roc_{pathology}_{protocol}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")

    return dict(pathology=pathology, protocol=protocol, auc=roc_auc, thr=opt_thr,
                TP=TP, FP=FP, FN=FN, TN=TN, sens=sens, spec=spec, prec=prec)


if __name__ == "__main__":
    rows = []
    for path in PATHOLOGIES:
        for proto in PROTOCOLS:
            r = run(path, proto)
            if r:
                rows.append(r)
    if rows:
        print("\nSummary (cite these):")
        print(f"{'pathology':13s}{'proto':6s}{'AUC':>7}{'thr':>7}"
              f"{'TP':>6}{'FP':>6}{'FN':>6}{'TN':>6}{'sens':>7}{'spec':>7}")
        for r in rows:
            print(f"{r['pathology']:13s}{r['protocol']:6s}"
                  f"{r['auc']:>7.3f}{r['thr']:>7.3f}"
                  f"{r['TP']:>6}{r['FP']:>6}{r['FN']:>6}{r['TN']:>6}"
                  f"{r['sens']:>7.3f}{r['spec']:>7.3f}")
