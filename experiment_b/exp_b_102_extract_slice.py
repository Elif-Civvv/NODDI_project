import nibabel as nib
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "100206")
RESULTS_DIR = os.path.join(HERE, "results", "slices_100206")

os.makedirs(RESULTS_DIR, exist_ok=True)

def extract_axial_slice(data_dir=DATA_DIR, z_slice=69):
    print("Loading NIfTI files...")

    # Load the 4D diffusion data
    img_data = nib.load(os.path.join(data_dir, "data.nii"))
    data = img_data.get_fdata()
    affine = img_data.affine

    # Load and clean the WM mask
    mask = nib.load(os.path.join(data_dir, "wm_mask.nii")).get_fdata()
    brain_mask = nib.load(os.path.join(data_dir, "nodif_brain_mask.nii")).get_fdata()
    mask = mask * (brain_mask > 0)

    # Load amy's NODDI maps (numba)
    odi_numba = nib.load(os.path.join(data_dir, "odi_whole_brain_numba.nii")).get_fdata()
    vic_numba = nib.load(os.path.join(data_dir, "vic_whole_brain_numba.nii")).get_fdata()
    viso_numba = nib.load(os.path.join(data_dir, "viso_whole_brain_numba.nii")).get_fdata()

    # Load my NODDI maps (v2)
    odi_v2 = nib.load(os.path.join(data_dir, "odi_v2.nii")).get_fdata()
    vic_v2 = nib.load(os.path.join(data_dir, "vic_v2.nii")).get_fdata()
    viso_v2 = nib.load(os.path.join(data_dir, "viso_v2.nii")).get_fdata()

    nx, ny, nz, nt = data.shape

    print(f"Original data shape: {data.shape}")
    print(f"Extracting Axial Slice: Z = {z_slice}")

    # Keep dimensionality intact for NIfTI compatibility
    sliced_data = data[:, :, z_slice:z_slice+1, :]
    sliced_mask = mask[:, :, z_slice:z_slice+1]

    sliced_odi_numba = odi_numba[:, :, z_slice:z_slice+1]
    sliced_vic_numba = vic_numba[:, :, z_slice:z_slice+1]
    sliced_viso_numba = viso_numba[:, :, z_slice:z_slice+1]

    sliced_odi_v2 = odi_v2[:, :, z_slice:z_slice+1]
    sliced_vic_v2 = vic_v2[:, :, z_slice:z_slice+1]
    sliced_viso_v2 = viso_v2[:, :, z_slice:z_slice+1]

    print(f"New data shape: {sliced_data.shape}")

    # Save all slices
    nib.save(nib.Nifti1Image(sliced_data, affine), os.path.join(RESULTS_DIR, f"data_axial_z{z_slice}.nii"))
    nib.save(nib.Nifti1Image(sliced_mask, affine), os.path.join(RESULTS_DIR, f"wm_mask_axial_z{z_slice}.nii"))

    nib.save(nib.Nifti1Image(sliced_odi_numba, affine), os.path.join(RESULTS_DIR, f"odi_numba_z{z_slice}.nii"))
    nib.save(nib.Nifti1Image(sliced_vic_numba, affine), os.path.join(RESULTS_DIR, f"vic_numba_z{z_slice}.nii"))
    nib.save(nib.Nifti1Image(sliced_viso_numba, affine), os.path.join(RESULTS_DIR, f"viso_numba_z{z_slice}.nii"))

    nib.save(nib.Nifti1Image(sliced_odi_v2, affine), os.path.join(RESULTS_DIR, f"odi_v2_z{z_slice}.nii"))
    nib.save(nib.Nifti1Image(sliced_vic_v2, affine), os.path.join(RESULTS_DIR, f"vic_v2_z{z_slice}.nii"))
    nib.save(nib.Nifti1Image(sliced_viso_v2, affine), os.path.join(RESULTS_DIR, f"viso_v2_z{z_slice}.nii"))

    print("Extraction complete!")
    print(f"All slices saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    extract_axial_slice(z_slice=69)
