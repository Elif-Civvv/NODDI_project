import nibabel as nib
import numpy as np
import os
import multiprocessing
from functools import partial
from scipy.optimize import differential_evolution

# 1. IMPORT UTILITIES
try:
    from hcp_compartments import (
        calculate_ball_signal,
        calculate_watson_stick_signal,
        calculate_noddi_extra_signal
    )
    from bbdb_sh_utils import fibonacci_sphere
except ImportError:
    raise ImportError("Ensure hcp_compartments.py and bbdb_sh_utils.py are in this folder.")

# 2. CONSTANTS
D_par = 1.7
D_iso = 3.0
WATSON_SAMPLES = fibonacci_sphere(samples=1000)

def odi_to_kappa(odi):
    odi = np.clip(odi, 0.01, 0.99)
    return 1.0 / np.tan(odi * np.pi / 2.0)

def calculate_noddi_total(b, g_vectors, v_ic, v_iso, kappa, mu, D_par, D_iso):
    # Standard NODDI signal equation
    A_ic = calculate_watson_stick_signal(b, D_par, mu, kappa, g_vectors, WATSON_SAMPLES)
    A_ec = calculate_noddi_extra_signal(b, D_par, v_ic, kappa, mu, g_vectors)
    A_iso = calculate_ball_signal(b, D_iso, g_vectors)
    
    S = (1.0 - v_iso) * (v_ic * A_ic + (1.0 - v_ic) * A_ec) + v_iso * A_iso
    return S

def fit_voxel_worker(S_raw, bvals, bvecs):
    """Worker function to fit a single voxel"""
    # Normalize the data first
    b0_mask = bvals < 0.01
    S0_est = np.mean(S_raw[b0_mask]) if np.any(b0_mask) else 1.0
    S_norm = S_raw / S0_est 

    def cost_function(x):
        v_ic_curr, odi_curr, v_iso_curr, theta, phi = x
        kappa_curr = odi_to_kappa(odi_curr)
        
        mu_curr = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

        S_model = calculate_noddi_total(
            bvals, bvecs, 
            v_ic_curr, v_iso_curr, kappa_curr, 
            mu_curr, D_par, D_iso
        )
        return np.sum((S_model - S_norm) ** 2)

    # Search space
    bounds = [(0.01, 0.99), (0.01, 0.99), (0.0, 1.0), (0, np.pi), (0, 2*np.pi)]
    result = differential_evolution(cost_function, bounds, popsize=30, tol=0.01)
    return result.x

if __name__ == "__main__":
    data_subfolder = "100307"
    cube_img = nib.load(os.path.join(data_subfolder, "wm_cube.nii"))
    cube_data = cube_img.get_fdata()
    affine = cube_img.affine

    # Scale b-values to ms/um^2
    bvals = np.loadtxt(os.path.join(data_subfolder, "bvals")) / 1000
    bvecs = np.loadtxt(os.path.join(data_subfolder, "bvecs")).T 

    nx, ny, nz, nt = cube_data.shape
    flat_data = cube_data.reshape(-1, nt)
    
    # Prepare results storage
    total_voxels = len(flat_data)
    all_results = np.zeros((total_voxels, 5))

    print(f"Parallel fitting with {multiprocessing.cpu_count()} cores...")
    print(f"{'Voxel #':<10} | {'v_ic':<8} | {'ODI':<8} | {'v_iso':<8}")
    print("-" * 50)

    # Partial allows passing constant bvals/bvecs to the worker
    worker_func = partial(fit_voxel_worker, bvals=bvals, bvecs=bvecs)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # imap returns results as they finish, allowing us to print updates
        for i, res in enumerate(pool.imap(worker_func, flat_data)):
            all_results[i, :] = res
            
            # Print every 10th voxel result
            if i % 10 == 0:
                print(f"Voxel {i:<5} | {res[0]:.4f} | {res[1]:.4f} | {res[2]:.4f}")

    # Reshape and save
    vic_map = all_results[:, 0].reshape((nx, ny, nz))
    odi_map = all_results[:, 1].reshape((nx, ny, nz))
    
    nib.save(nib.Nifti1Image(vic_map, affine), os.path.join(data_subfolder, "vic_result.nii"))
    nib.save(nib.Nifti1Image(odi_map, affine), os.path.join(data_subfolder, "odi_result.nii"))
    
    print("\nProcessing complete. Check your results now!")