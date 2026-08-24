import nibabel as nib
import numpy as np
import os

from hcp_compartments import (
    calculate_ball_signal,
    calculate_watson_stick_signal,
    calculate_noddi_extra_signal
)
from bbdb_compartments import calculate_sphere_signal
from bbdb_sh_utils import fibonacci_sphere

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
SLICES_DIR = os.path.join(HERE, "results", "slices_100206")
RESULTS_DIR = os.path.join(HERE, "results", "glia")
DATA_DIR = os.path.join(HERE, "100206")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
D_par = 1.7
D_iso = 3.0
D_glia = 3.0
R_glia = 5.0

T2_tissue = 100.0
T2_sphere = 30.0
T2_CSF = 2000.0

BIG_DELTA = 43.1
LITTLE_DELTA = 10.6

HEALTHY_VIC = 0.73
HEALTHY_ODI = 0.25
HEALTHY_VISO = 0.16
HEALTHY_VGLIA = 0.0

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------
def make_protocol(name):
    n_dirs = 64
    directions = fibonacci_sphere(samples=n_dirs)
    n_b0 = 6

    bvals, bvecs, te_list = [], [], []

    if name == "hcp":
        b_shells = [1.0, 2.0, 3.0]
        TEs = [62.0]
        for te in TEs:
            for _ in range(n_b0):
                bvals.append(0.0)
                bvecs.append([1.0, 0.0, 0.0])
                te_list.append(te)
            for b in b_shells:
                for d in range(n_dirs):
                    bvals.append(b)
                    bvecs.append(directions[d])
                    te_list.append(te)
        desc = f"HCP: {n_b0} b0 + 3 shells x 64 dirs x 1 TE = {len(bvals)}"

    elif name == "novel":
        b_shells = [1.0, 2.0]
        TEs = [62.0, 92.0, 130.0]
        for te in TEs:
            for _ in range(n_b0):
                bvals.append(0.0)
                bvecs.append([1.0, 0.0, 0.0])
                te_list.append(te)
            for b in b_shells:
                for d in range(n_dirs):
                    bvals.append(b)
                    bvecs.append(directions[d])
                    te_list.append(te)
        desc = f"Novel: {n_b0*3} b0 + 2 shells x 64 dirs x 3 TEs = {len(bvals)}"

    return {
        "name": name,
        "bvals": np.array(bvals),
        "bvecs": np.array(bvecs),
        "TEs": np.array(te_list),
        "n_measurements": len(bvals),
        "description": desc,
    }


def b_to_G(b, delta, Delta):
    if b < 1e-10:
        return 0.0
    gamma = 2.6752218744e8
    delta_s = delta * 1e-3
    Delta_s = Delta * 1e-3
    b_si = b * 1e9
    G_T_m = np.sqrt(b_si / (gamma**2 * delta_s**2 * (Delta_s - delta_s / 3.0)))
    return G_T_m * 1e3


def odi_to_kappa(odi):
    odi = np.clip(odi, 0.01, 0.99)
    return 1.0 / np.tan(odi * np.pi / 2.0)


PATHOLOGIES = {
    "astrogliosis": {
        "description": "Glial scarring: elevated ODI + glia fraction",
        "vic": 0.65, "odi": 0.55, "viso": 0.16, "vglia": 0.20,
    },
    "edema": {
        "description": "Vasogenic edema: elevated free water",
        "vic": 0.65, "odi": 0.25, "viso": 0.45, "vglia": 0.0,
    },
    "chronic_tbi": {
        "description": "Combined chronic TBI: axonal loss + gliosis + edema",
        "vic": 0.55, "odi": 0.50, "viso": 0.35, "vglia": 0.15,
    },
}


def define_rois(nx, ny, wm_slice):
    rois = []
    roi_size = 6
    for name, cx, cy in [("roi_upper_left", nx // 4, ny // 4),
                          ("roi_centre", nx // 2, ny // 2),
                          ("roi_lower_right", 3 * nx // 4, 3 * ny // 4)]:
        roi_mask = np.zeros((nx, ny), dtype=bool)
        for i in range(max(0, cx - roi_size), min(nx, cx + roi_size)):
            for j in range(max(0, cy - roi_size), min(ny, cy + roi_size)):
                if wm_slice[i, j] > 0:
                    roi_mask[i, j] = True
        if roi_mask.sum() > 10:
            rois.append((name, roi_mask))
            print(f"  {name}: {roi_mask.sum()} WM voxels at ({cx}, {cy})")
    return rois


def assign_orientations(nx, ny, wm_slice, seed=42):
    rng = np.random.RandomState(seed)
    theta_map = np.zeros((nx, ny))
    phi_map = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            if wm_slice[i, j] > 0:
                if i < nx // 2 and j < ny // 2:
                    bt, bp = np.pi / 2, np.pi / 2
                elif i < nx // 2:
                    bt, bp = np.pi / 2, 0.0
                elif j < ny // 2:
                    bt, bp = np.pi / 2, np.pi / 4
                else:
                    bt, bp = np.pi / 4, 0.0
                jit = np.deg2rad(15)
                theta_map[i, j] = bt + rng.uniform(-jit, jit)
                phi_map[i, j] = bp + rng.uniform(-jit, jit)
    return theta_map, phi_map


def generate_voxel_signal(protocol, vic, odi, viso, vglia, theta, phi,
                          watson_samples, snr=20):
    """Generate signal for one voxel using vectorized math."""
    bvals = protocol["bvals"]
    bvecs = protocol["bvecs"]
    TEs = protocol["TEs"]
    n_meas = protocol["n_measurements"]

    kappa = odi_to_kappa(odi)
    mu = np.array([np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)])

    # VECTORIZED: Pass the entire bvals array to the signal calculators
    A_ic = calculate_watson_stick_signal(bvals, D_par, mu, kappa, bvecs, watson_samples)
    A_ec = calculate_noddi_extra_signal(bvals, D_par, vic, kappa, mu, bvecs)
    A_ball = np.exp(-bvals * D_iso)
    
    # Precompute sphere signal for all b-values at once
    A_sphere = np.array([calculate_sphere_signal(b, b_to_G(b, LITTLE_DELTA, BIG_DELTA), 
                         LITTLE_DELTA, BIG_DELTA, R_glia, D_glia, bvecs[:1])[0] 
                         if b > 1e-10 else 1.0 for b in bvals])

    # Vectorized T2 weights
    w_t = np.exp(-TEs / T2_tissue)
    w_s = np.exp(-TEs / T2_sphere)
    w_c = np.exp(-TEs / T2_CSF)

    # Full truth signal
    tissue_signal = (1.0 - vglia) * (vic * A_ic + (1.0 - vic) * A_ec) * w_t
    glia_signal = vglia * A_sphere * w_s
    csf_signal = viso * A_ball * w_c

    S_out = (1.0 - viso) * (tissue_signal + glia_signal) + csf_signal

    # Add Rician noise
    sigma = np.mean(S_out[bvals < 1e-10]) / snr if np.any(bvals < 1e-10) else np.mean(S_out) / snr
    N1 = np.random.normal(0, sigma, n_meas)
    N2 = np.random.normal(0, sigma, n_meas)
    return np.sqrt((S_out + N1) ** 2 + N2 ** 2)

def generate_phantom(pathology_name, seed=42):
    pathology = PATHOLOGIES[pathology_name]
    print(f"\n{'=' * 60}")
    print(f"Generating: {pathology_name}")
    print(f"  {pathology['description']}")
    print(f"  vic={pathology['vic']}, odi={pathology['odi']}, "
          f"viso={pathology['viso']}, vglia={pathology['vglia']}")
    print(f"{'=' * 60}")

    np.random.seed(seed)

    mask_img = nib.load(os.path.join(SLICES_DIR, "wm_mask_axial_z69.nii"))
    wm_mask = mask_img.get_fdata()
    affine = mask_img.affine
    nx, ny, nz = wm_mask.shape
    wm_slice = wm_mask[:, :, 0]
    total_wm = int(wm_slice.sum())

    print(f"Slice: {nx} x {ny}, WM voxels: {total_wm}")
    print("Defining ROIs...")
    rois = define_rois(nx, ny, wm_slice)
    combined_roi = np.zeros((nx, ny), dtype=bool)
    for _, rm in rois:
        combined_roi |= rm

    print("Assigning orientations...")
    theta_map, phi_map = assign_orientations(nx, ny, wm_slice, seed)
    watson_samples = fibonacci_sphere(samples=500)
    protocols = {"hcp": make_protocol("hcp"), "novel": make_protocol("novel")}

    vic_gt = np.zeros((nx, ny, 1))
    odi_gt = np.zeros((nx, ny, 1))
    viso_gt = np.zeros((nx, ny, 1))
    vglia_gt = np.zeros((nx, ny, 1))
    roi_label = np.zeros((nx, ny, 1))

    signal_arrays = {}
    for pn, pr in protocols.items():
        signal_arrays[pn] = np.zeros((nx, ny, 1, pr["n_measurements"]))

    voxel_count = 0
    for i in range(nx):
        for j in range(ny):
            if wm_slice[i, j] > 0:
                if combined_roi[i, j]:
                    vic = pathology["vic"]
                    odi = pathology["odi"]
                    viso = pathology["viso"]
                    vglia = pathology["vglia"]
                    for k, (_, rm) in enumerate(rois):
                        if rm[i, j]:
                            roi_label[i, j, 0] = k + 1
                            break
                else:
                    vic, odi, viso, vglia = HEALTHY_VIC, HEALTHY_ODI, HEALTHY_VISO, HEALTHY_VGLIA

                vic_gt[i, j, 0] = vic
                odi_gt[i, j, 0] = odi
                viso_gt[i, j, 0] = viso
                vglia_gt[i, j, 0] = vglia

                for pn, pr in protocols.items():
                    signal_arrays[pn][i, j, 0, :] = generate_voxel_signal(
                        pr, vic, odi, viso, vglia, theta_map[i, j], phi_map[i, j],
                        watson_samples, snr=20
                    )
                voxel_count += 1
                if voxel_count % 500 == 0:
                    print(f"  {voxel_count}/{total_wm} voxels...")

    print(f"Generated {voxel_count} voxels")

    out_dir = os.path.join(RESULTS_DIR, pathology_name)
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"phantom_{pathology_name}"

    nib.save(nib.Nifti1Image(vic_gt, affine), os.path.join(out_dir, f"{prefix}_vic_gt.nii"))
    nib.save(nib.Nifti1Image(odi_gt, affine), os.path.join(out_dir, f"{prefix}_odi_gt.nii"))
    nib.save(nib.Nifti1Image(viso_gt, affine), os.path.join(out_dir, f"{prefix}_viso_gt.nii"))
    nib.save(nib.Nifti1Image(vglia_gt, affine), os.path.join(out_dir, f"{prefix}_vglia_gt.nii"))
    nib.save(nib.Nifti1Image(roi_label, affine), os.path.join(out_dir, f"{prefix}_roi_labels.nii"))

    for pn, pr in protocols.items():
        nib.save(nib.Nifti1Image(signal_arrays[pn], affine),
                 os.path.join(out_dir, f"{prefix}_data_{pn}.nii"))
        np.savetxt(os.path.join(out_dir, f"bvals_{pn}.txt"), pr["bvals"])
        np.savetxt(os.path.join(out_dir, f"bvecs_{pn}.txt"), pr["bvecs"])
        np.savetxt(os.path.join(out_dir, f"TEs_{pn}.txt"), pr["TEs"])

    print(f"Saved to: {out_dir}")
    for pn in protocols:
        print(f"  {prefix}_data_{pn}.nii  ({protocols[pn]['description']})")


if __name__ == "__main__":
    for p in ["astrogliosis", "edema", "chronic_tbi"]:
        generate_phantom(p, seed=42)
    print("\nAll phantoms generated.")
