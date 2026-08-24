import nibabel as nib
import numpy as np
import os

# ---------------------------------------------------------------------------
# Portable project paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "100307")
RESULTS_DIR = os.path.join(HERE, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


def extract_priors():
    print("Loading whole-brain NODDI maps and WM mask...")

    # Load WM mask and brain mask
    wm_mask = nib.load(os.path.join(DATA_DIR, "wm_mask.nii")).get_fdata()
    brain_mask = nib.load(os.path.join(DATA_DIR, "nodif_brain_mask.nii")).get_fdata()
    mask = (wm_mask > 0) & (brain_mask > 0)
    mask_flat = mask.flatten()

    print(f"Total WM voxels: {mask_flat.sum()}")

    # Define map sources
    sources = {
        "elif": {
            "odi": os.path.join(DATA_DIR, "odi_v2.nii"),
            "vic": os.path.join(DATA_DIR, "vic_v2.nii"),
            "viso": os.path.join(DATA_DIR, "viso_v2.nii"),
        },
        "prof": {
            "odi": os.path.join(DATA_DIR, "odi_whole_brain_numba.nii"),
            "vic": os.path.join(DATA_DIR, "vic_whole_brain_numba.nii"),
            "viso": os.path.join(DATA_DIR, "viso_whole_brain_numba.nii"),
        },
    }

    all_results = []

    for author, maps in sources.items():
        for param, path in maps.items():
            data = nib.load(path).get_fdata().flatten()
            vals = data[mask_flat]

            p5 = np.percentile(vals, 5)
            p95 = np.percentile(vals, 95)
            mean = np.mean(vals)
            median = np.median(vals)
            std = np.std(vals)

            row = f"{param}, {author}, {p5:.4f}, {p95:.4f}, {mean:.4f}, {median:.4f}, {std:.4f}"
            all_results.append(row)

            print(row)

    # Save to CSV
    header = "parameter, source, p5, p95, mean, median, std"
    out_path = os.path.join(RESULTS_DIR, "prior_ranges.csv")
    with open(out_path, "w") as f:
        f.write(header + "\n")
        for r in all_results:
            f.write(r + "\n")

    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    extract_priors()
