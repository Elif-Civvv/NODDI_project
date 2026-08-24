import nibabel as nib
import numpy as np
import os
from hcp_cube_parallel import calculate_noddi_total, odi_to_kappa, D_par, D_iso

def create_astrogliosis_phantom(base_path, data_subfolder="100307"):
    folder_path = os.path.join(base_path, data_subfolder)
    
    # 1. Load Data
    bvals = np.loadtxt(os.path.join(folder_path, "bvals")) / 1000.0
    bvecs = np.loadtxt(os.path.join(folder_path, "bvecs")).T
    
    mask_img = nib.load(os.path.join(folder_path, "wm_mask.nii"))
    wm_mask = mask_img.get_fdata()
    affine = mask_img.affine
    
    # 2. Extract 2D Slice (Middle Axial Slice)
    nx, ny, nz = wm_mask.shape
    z_slice = nz // 2
    wm_slice = wm_mask[:, :, z_slice]
    
    # 3. Initialize Ground Truth Arrays
    vic_gt = np.zeros((nx, ny))
    odi_gt = np.zeros((nx, ny))
    viso_gt = np.zeros((nx, ny))
    S_phantom = np.zeros((nx, ny, len(bvals)))
    
    # 4. Define Healthy and Glial Parameters
    # Healthy WM
    healthy_vic = 0.60
    healthy_odi = 0.15
    healthy_viso = 0.05
    
    # Astrogliosis (High dispersion)
    glia_vic = 0.60
    glia_odi = 0.65  # Drastically increased
    glia_viso = 0.05
    
    # Define ROI bounding box (e.g., a 10x10 square in the center)
    cx, cy = nx // 2, ny // 2
    roi_size = 10
    
    print("Generating Astrogliosis Phantom...")
    # 5. Populate Arrays and Generate Signal
    # Assume a primary fiber orientation along the x-axis for the phantom
    mu = np.array([1.0, 0.0, 0.0])
    
    for i in range(nx):
        for j in range(ny):
            if wm_slice[i, j] > 0:
                # Check if voxel is inside our artificial Glia ROI
                is_glia = (cx - roi_size < i < cx + roi_size) and (cy - roi_size < j < cy + roi_size)
                
                vic = glia_vic if is_glia else healthy_vic
                odi = glia_odi if is_glia else healthy_odi
                viso = glia_viso if is_glia else healthy_viso
                
                vic_gt[i, j] = vic
                odi_gt[i, j] = odi
                viso_gt[i, j] = viso
                
                kappa = odi_to_kappa(odi)
                S_clean = calculate_noddi_total(bvals, bvecs, vic, viso, kappa, mu, D_par, D_iso)
                
                # Add Rician Noise (SNR = 30)
                sigma = 1.0 / 30.0 
                N1 = np.random.normal(0, sigma, len(bvals))
                N2 = np.random.normal(0, sigma, len(bvals))
                S_noisy = np.sqrt((S_clean + N1)**2 + N2**2)
                
                S_phantom[i, j, :] = S_noisy

    # 6. Save Ground Truth and Phantom Signal
    nib.save(nib.Nifti1Image(S_phantom, affine), os.path.join(folder_path, "phantom_astrogliosis_data.nii"))
    nib.save(nib.Nifti1Image(vic_gt, affine), os.path.join(folder_path, "phantom_astrogliosis_vic_gt.nii"))
    nib.save(nib.Nifti1Image(odi_gt, affine), os.path.join(folder_path, "phantom_astrogliosis_odi_gt.nii"))
    nib.save(nib.Nifti1Image(viso_gt, affine), os.path.join(folder_path, "phantom_astrogliosis_viso_gt.nii"))
    
    print("Astrogliosis Phantom saved successfully.")

if __name__ == "__main__":
    BASE_DIR = r'/Users/elifcivelekoglu/Library/Mobile Documents/com~apple~CloudDocs/Imperial/year 4/Thesis/Base compartments'
    create_astrogliosis_phantom(BASE_DIR)