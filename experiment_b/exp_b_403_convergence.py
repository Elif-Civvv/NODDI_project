#!/usr/bin/env python
"""
Experiment B convergence table (split-Rhat), the spatial analogue of Table 4.

Experiment B fits ~5400 voxels independently (no sweep), so the per-condition
"n>1.1 out of 4" column of Table 4 does not apply. Instead, for each fitted
parameter we summarise the per-voxel Rhat distribution across white-matter voxels:
median, max, and the percentage of voxels exceeding 1.1.

Reads the per-voxel Rhat maps (..._mcmc_rhat.nii), same (X,Y,1,6) layout and
volume order as the median maps: [v_ic, ODI, v_glia, v_iso, ...].

Fixed-T2 configuration => T2,t and T2,s are held at truth, not fitted, so they
are not reported. Emits a LaTeX table matching the thesis style.
Freshness guard: skips an Rhat map older than its checkpoint (likely stale).
"""

import os
import sys
import numpy as np
import nibabel as nib

BASE = "results/glia"
PROTOCOLS = ["hcp", "novel"]
PATHOLOGIES = ["astrogliosis", "edema"]
ALLOW_STALE = "--allow-stale" in sys.argv

# fitted parameters in Experiment B (fixed-T2): name -> volume index, LaTeX label
PARAMS = [
    ("v_ic",   0, r"$v_{ic}$"),
    ("ODI",    1, r"ODI"),
    ("v_glia", 2, r"$v_{glia}$"),
    ("v_iso",  3, r"$v_{iso}$"),
]
THRESH = 1.1


def load_slice(path):
    arr = nib.load(path).get_fdata()
    return np.squeeze(arr, axis=2) if arr.shape[2] == 1 else arr


def summarise(pathology, protocol):
    d = os.path.join(BASE, pathology)
    rhat_path = os.path.join(d, f"phantom_{pathology}_{protocol}_mcmc_rhat.nii")
    med_path = os.path.join(d, f"phantom_{pathology}_{protocol}_mcmc_median.nii")
    ckpt_path = os.path.join(d, f"phantom_{pathology}_{protocol}_ckpt.npy")

    if not os.path.exists(rhat_path):
        print(f"[skip] {pathology} {protocol}: no rhat map")
        return None
    if os.path.exists(ckpt_path) and not ALLOW_STALE:
        if os.path.getmtime(rhat_path) < os.path.getmtime(ckpt_path):
            print(f"[STALE] {pathology} {protocol}: rhat map predates checkpoint, "
                  f"skipping (use --allow-stale to force)")
            return None

    rhat = load_slice(rhat_path)               # (X, Y, 6)
    wm = load_slice(med_path)[..., 0] != 0     # WM from v_ic of the median map

    rows = {}
    for name, idx, _ in PARAMS:
        vals = rhat[..., idx][wm]
        vals = vals[np.isfinite(vals) & (vals > 0)]   # drop unfitted/zero voxels
        med = float(np.median(vals))
        mx = float(np.max(vals))
        pct = 100.0 * float(np.mean(vals > THRESH))
        rows[name] = (med, mx, pct)
    return rows


def main():
    data = {}   # (pathology, protocol) -> rows
    for path in PATHOLOGIES:
        for proto in PROTOCOLS:
            r = summarise(path, proto)
            if r:
                data[(path, proto)] = r

    # console summary
    for (path, proto), rows in data.items():
        print(f"\n=== {path} ({proto.upper()}) ===")
        print(f"{'Param':<8}{'med':>7}{'max':>7}{'%>1.1':>8}")
        for name, _, _ in PARAMS:
            med, mx, pct = rows[name]
            print(f"{name:<8}{med:>7.2f}{mx:>7.2f}{pct:>7.1f}%")

    # ---- LaTeX (one table per pathology, NODDI vs Novel side by side) ----
    print("\n\n% ---------- LaTeX ----------")
    for path in PATHOLOGIES:
        if (path, "hcp") not in data:
            continue
        print(r"\begin{table}[H]\centering\footnotesize")
        print(r"\begin{tabular}{lccc ccc}")
        print(r"\toprule")
        print(r"& \multicolumn{3}{c}{NODDI} & \multicolumn{3}{c}{Novel} \\")
        print(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}")
        print(r"Parameter & med & max & $\%{>}1.1$ & med & max & $\%{>}1.1$ \\")
        print(r"\midrule")
        for name, _, label in PARAMS:
            h = data.get((path, "hcp"), {}).get(name)
            n = data.get((path, "novel"), {}).get(name)
            hs = (f"{h[0]:.2f} & {h[1]:.2f} & {h[2]:.1f}" if h else "-- & -- & --")
            ns = (f"{n[0]:.2f} & {n[1]:.2f} & {n[2]:.1f}" if n else "-- & -- & --")
            print(f"{label} & {hs} & {ns} \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(rf"\caption{{Per-voxel convergence (split-$\hat{{R}}$) for the {path} "
              rf"spatial fits, summarised across white-matter voxels. "
              rf"`$\%{{>}}1.1$' is the percentage of voxels exceeding the 1.1 "
              rf"threshold. $T_2$ parameters are fixed in this configuration and "
              rf"not fitted.}}")
        print(rf"\label{{tab:conv_{path}}}")
        print(r"\end{table}")
        print()


if __name__ == "__main__":
    main()
