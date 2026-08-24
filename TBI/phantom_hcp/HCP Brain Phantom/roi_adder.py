import nibabel as nib
import numpy as np
import os

from hcp_cube_parallel import calculate_noddi_total, odi_to_kappa, D_par, D_iso

# Portable project paths
HERE = os.path.dirname(os.path.abspath(__file__))
SLICES_DIR = os.path.join(HERE, "results", "slices_100206")
RESULTS_DIR = os.path.join(HERE, "results", "phantoms")
DATA_DIR = os.path.join(HERE, "100206")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Empirical healthy WM values (from prior_ranges.csv, elif medians)
HEALTHY_VIC = 0.73
HEALTHY_ODI = 0.25
HEALTHY_VISO = 0.16

# Pathology definitions
PATHOLOGIES = {
    "astrogliosis": {
        "description": "Elevated ODI from glial scarring and fibre disorganisation",
        "vic": 0.65,      # slight decrease (axonal loss)
        "odi": 0.55,      # significant increase
        "viso": 0.16,     # unchanged
    },
    "edema": {
        "description": "Elevated free water from vasogenic edema",
        "vic": 0.65,      # slight decrease
        "odi": 0.25,      # unchanged
        "viso": 0.45,     # significant increase
    },
    "chronic_tbi": {
        "description": "Combined chronic TBI: axonal loss + gliosis + edema",
        "vic": 0.55,      # significant decrease (axonal degeneration)
        "odi": 0.50,      # increased (fibre disorganisation)
        "viso": 0.35,     # increased (residual edema)
    },
}

def define_rois(nx, ny, wm_slice):
    """Define 3 ROIs in different WM regions.
    Returns a list of (name, mask_2d) tuples.
    Each mask is True inside the ROI and within WM.
    """
    rois = []
    roi_size = 15

    # ROI 1: upper-left quadrant
    cx1, cy1 = nx // 4, ny // 4
    # ROI 2: centre
    cx2, cy2 = nx // 2, ny // 2
    # ROI 3: lower-right quadrant
    cx3, cy3 = 3 * nx // 4, 3 * ny // 4

    for name, cx, cy in [("roi_upper_left", cx1, cy1),
                          ("roi_centre", cx2, cy2),
                          ("roi_lower_right", cx3, cy3)]:
        roi_mask = np.zeros((nx, ny), dtype=bool)
        for i in range(max(0, cx - roi_size), min(nx, cx + roi_size)):
            for j in range(max(0, cy - roi_size), min(ny, cy + roi_size)):
                if wm_slice[i, j] > 0:
                    roi_mask[i, j] = True

        # Only keep ROI if it has enough WM voxels
        if roi_mask.sum() > 10:
            rois.append((name, roi_mask))
            print(f"  {name}: {roi_mask.sum()} WM voxels at ({cx}, {cy})")
        else:
            print(f"  {name}: skipped (only {roi_mask.sum()} WM voxels)")

    return rois


def assign_orientations(nx, ny, wm_slice, seed=42):
    """Assign varying fibre orientations across the slice.

    Different quadrants get different primary orientations,
    with per-voxel jitter to simulate realistic variation.
    """
    rng = np.random.RandomState(seed)

    # Base orientations for different regions (theta, phi)
    # These represent different WM tract directions
    theta_map = np.zeros((nx, ny))
    phi_map = np.zeros((nx, ny))

    for i in range(nx):
        for j in range(ny):
            if wm_slice[i, j] > 0:
                # Assign base orientation by quadrant
                if i < nx // 2 and j < ny // 2:
                    # Upper-left: anterior-posterior (along y)
                    base_theta, base_phi = np.pi / 2, np.pi / 2
                elif i < nx // 2 and j >= ny // 2:
                    # Upper-right: left-right (along x)
                    base_theta, base_phi = np.pi / 2, 0.0
                elif i >= nx // 2 and j < ny // 2:
                    # Lower-left: diagonal
                    base_theta, base_phi = np.pi / 2, np.pi / 4
                else:
                    # Lower-right: superior-inferior (along z)
                    base_theta, base_phi = np.pi / 4, 0.0

                # Add per-voxel jitter (±15 degrees)
                jitter = np.deg2rad(15)
                theta_map[i, j] = base_theta + rng.uniform(-jitter, jitter)
                phi_map[i, j] = base_phi + rng.uniform(-jitter, jitter)

    return theta_map, phi_map


def generate_phantom(pathology_name, seed=42):
    """Generate a phantom for a given pathology type."""

    if pathology_name not in PATHOLOGIES:
        raise ValueError(f"Unknown pathology: {pathology_name}. "
                         f"Choose from {list(PATHOLOGIES.keys())}")

    pathology = PATHOLOGIES[pathology_name]
    print(f"\n{'=' * 60}")
    print(f"Generating phantom: {pathology_name}")
    print(f"  {pathology['description']}")
    print(f"  Pathology values: vic={pathology['vic']}, "
          f"odi={pathology['odi']}, viso={pathology['viso']}")
    print(f"{'=' * 60}")

    np.random.seed(seed)

    # Load sliced WM mask (already brain-masked)
    mask_img = nib.load(os.path.join(SLICES_DIR, "wm_mask_axial_z69.nii"))
    wm_mask = mask_img.get_fdata()
    affine = mask_img.affine

    # Load acquisition parameters
    bvals = np.loadtxt(os.path.join(DATA_DIR, "bvals")) / 1000.0
    bvecs = np.loadtxt(os.path.join(DATA_DIR, "bvecs")).T

    # Get dimensions (nx, ny, 1 from sliced data)
    nx, ny, nz = wm_mask.shape
    wm_slice = wm_mask[:, :, 0]
    n_bvals = len(bvals)

    print(f"Slice dimensions: {nx} x {ny}, WM voxels: {int(wm_slice.sum())}")

    # Define ROIs
    print("Defining ROIs...")
    rois = define_rois(nx, ny, wm_slice)

    # Build combined ROI mask (union of all ROIs)
    combined_roi = np.zeros((nx, ny), dtype=bool)
    for name, roi_mask in rois:
        combined_roi |= roi_mask

    # Assign varying orientations
    print("Assigning fibre orientations...")
    theta_map, phi_map = assign_orientations(nx, ny, wm_slice, seed=seed)

    # Initialise ground truth and signal arrays (with singleton z)
    vic_gt = np.zeros((nx, ny, 1))
    odi_gt = np.zeros((nx, ny, 1))
    viso_gt = np.zeros((nx, ny, 1))
    roi_label = np.zeros((nx, ny, 1))  # label map: 0=healthy, 1/2/3=ROI
    S_phantom = np.zeros((nx, ny, 1, n_bvals))

    # Generate signal
    print("Generating synthetic diffusion signal...")
    voxel_count = 0

    for i in range(nx):
        for j in range(ny):
            if wm_slice[i, j] > 0:
                # Determine parameters
                if combined_roi[i, j]:
                    vic = pathology["vic"]
                    odi = pathology["odi"]
                    viso = pathology["viso"]

                    # Label which ROI
                    for k, (name, roi_mask) in enumerate(rois):
                        if roi_mask[i, j]:
                            roi_label[i, j, 0] = k + 1
                            break
                else:
                    vic = HEALTHY_VIC
                    odi = HEALTHY_ODI
                    viso = HEALTHY_VISO

                vic_gt[i, j, 0] = vic
                odi_gt[i, j, 0] = odi
                viso_gt[i, j, 0] = viso

                # Per-voxel orientation
                theta = theta_map[i, j]
                phi = phi_map[i, j]
                mu = np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                ])

                kappa = odi_to_kappa(odi)
                S_clean = calculate_noddi_total(
                    bvals, bvecs, vic, viso, kappa, mu, D_par, D_iso
                )

                # Add Rician noise (SNR = 30)
                sigma = 1.0 / 30.0
                N1 = np.random.normal(0, sigma, n_bvals)
                N2 = np.random.normal(0, sigma, n_bvals)
                S_noisy = np.sqrt((S_clean + N1) ** 2 + N2 ** 2)

                S_phantom[i, j, 0, :] = S_noisy
                voxel_count += 1

    print(f"Generated signal for {voxel_count} WM voxels")

    # Save outputs
    out_prefix = f"phantom_{pathology_name}"
    out_dir = os.path.join(RESULTS_DIR, pathology_name)
    os.makedirs(out_dir, exist_ok=True)

    nib.save(nib.Nifti1Image(S_phantom, affine),
             os.path.join(out_dir, f"{out_prefix}_data.nii"))
    nib.save(nib.Nifti1Image(vic_gt, affine),
             os.path.join(out_dir, f"{out_prefix}_vic_gt.nii"))
    nib.save(nib.Nifti1Image(odi_gt, affine),
             os.path.join(out_dir, f"{out_prefix}_odi_gt.nii"))
    nib.save(nib.Nifti1Image(viso_gt, affine),
             os.path.join(out_dir, f"{out_prefix}_viso_gt.nii"))
    nib.save(nib.Nifti1Image(roi_label, affine),
             os.path.join(out_dir, f"{out_prefix}_roi_labels.nii"))

    print(f"Saved to: {out_dir}")
    print(f"  {out_prefix}_data.nii       (4D synthetic signal)")
    print(f"  {out_prefix}_vic_gt.nii     (ground truth vic)")
    print(f"  {out_prefix}_odi_gt.nii     (ground truth odi)")
    print(f"  {out_prefix}_viso_gt.nii    (ground truth viso)")
    print(f"  {out_prefix}_roi_labels.nii (ROI label map)")

    return out_dir


if __name__ == "__main__":
    for pathology in ["astrogliosis", "edema", "chronic_tbi"]:
        generate_phantom(pathology, seed=42)

    print("\n" + "=" * 60)
    print("All phantoms generated successfully.")
    print("=" * 60)
