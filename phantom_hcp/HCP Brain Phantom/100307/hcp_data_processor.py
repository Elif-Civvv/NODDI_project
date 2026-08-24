import nibabel as nib
import numpy as np
import os

def run_hcp_pipeline(base_path, files_dict):
    data_path = os.path.join(base_path, files_dict['data_path'])
    output_path = os.path.join(base_path, files_dict['output_cube_name'])

    img = nib.load(data_path)
    data = img.get_fdata()
    
    # Center of white matter tract
    x, y, z = 70, 90, 70 
    cube_data = data[x:x+10, y:y+10, z:z+10, :]
    
    nib.save(nib.Nifti1Image(cube_data, img.affine), output_path)
    print(f"Cube successfully extracted to: {output_path}")

if __name__ == "__main__":
    DATA_DIR = r'/Users/elifcivelekoglu/Library/Mobile Documents/com~apple~CloudDocs/Imperial/year 4/Thesis/Base compartments/100307'
    file_configs = {'data_path': 'data.nii', 'output_cube_name': 'wm_cube.nii'}
    run_hcp_pipeline(DATA_DIR, file_configs)