"""
check_speed.py
==============
Times the MCMC fit on a small batch of voxels so you can see vox/s and
extrapolate whether the remaining voxels finish inside the wall, BEFORE
committing another 36h job.

Run on the cluster (login node is fine for a tiny batch, or in an interactive
job):
  python check_speed.py --pathology edema --protocol novel --n 8

It imports the real engine, so it uses the exact same physics, MCMC depth,
and signal model as the production run.
"""

import argparse
import os
import time
import numpy as np
import nibabel as nib

# Import the production engine so timing reflects the real fit.
import exp_b_301_fit_mcmc_engine as eng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pathology", default="edema")
    ap.add_argument("--protocol", default="novel", choices=["hcp", "novel"])
    ap.add_argument("--n", type=int, default=8, help="voxels to time")
    args = ap.parse_args()

    HERE = os.path.dirname(os.path.abspath(__file__))
    PHANTOM_DIR = os.path.join(HERE, "results", "glia", args.pathology)
    prefix = f"phantom_{args.pathology}"

    # Load data + geometry exactly as the engine does.
    data = nib.load(os.path.join(
        PHANTOM_DIR, f"{prefix}_data_{args.protocol}.nii")).get_fdata()
    bvals = np.loadtxt(os.path.join(PHANTOM_DIR, f"bvals_{args.protocol}.txt"))
    bvecs = np.loadtxt(os.path.join(PHANTOM_DIR, f"bvecs_{args.protocol}.txt"))
    TEs = np.loadtxt(os.path.join(PHANTOM_DIR, f"TEs_{args.protocol}.txt"))

    # Use ROI voxels (the ones the production job fits).
    roi = nib.load(os.path.join(
        PHANTOM_DIR, f"{prefix}_roi_labels.nii")).get_fdata()[:, :, 0] > 0
    vox = data[roi, 0, :]
    n = min(args.n, len(vox))
    batch = vox[:n]

    watson = eng.fibonacci_sphere(samples=500)
    sphere_cache = eng.precompute_spheres(bvals, bvecs)

    print(f"Timing {n} {args.pathology}/{args.protocol} voxels "
          f"({batch.shape[1]} measurements each, "
          f"{eng.N_WALKERS}w x {eng.N_STEPS}s)...")

    t0 = time.time()
    for v in batch:
        eng.fit_voxel_mcmc(v, bvals, bvecs, TEs, watson, sphere_cache)
    dt = time.time() - t0

    per_vox = dt / n
    vps = n / dt
    print(f"\n  {n} voxels in {dt:.1f}s  ->  {per_vox:.1f}s/voxel  "
          f"({vps:.3f} vox/s, single core)")

    # Extrapolate. Production uses all NCPUS cores in parallel.
    cores = int(os.environ.get("NCPUS", 8))
    eff = vps * cores * 0.8  # 80% parallel efficiency, rough
    for remaining in (570, 770):
        hrs = remaining / eff / 3600
        print(f"  ~{remaining} voxels on {cores} cores (~80% eff): "
              f"{hrs:.1f} h  -> {'OK within 36h wall' if hrs < 34 else 'WILL NOT FINISH'}")


if __name__ == "__main__":
    main()
