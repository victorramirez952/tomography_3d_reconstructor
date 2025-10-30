#!/usr/bin/env python3
import os
import shutil
from ellipsoid_slice_generator import EllipsoidSliceGenerator

def generate_slices_from_mask(mask_path, n_slices, output_directory, num_start, increase):
    """Generates half-ellipsoid slices from mask for 3D reconstruction."""
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory, exist_ok=True)
    
    if not os.path.exists(mask_path):
        print(f"Error: Image '{mask_path}' not found.")
        return
    
    try:
        generator = EllipsoidSliceGenerator(mask_path)
        slice_files = generator.generate_slices_half_ellipsoid(n_slices, output_directory, num_start, increase)
        print(f"Generated {len(slice_files)} slices in '{output_directory}'")
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    mask_path = "/home/vector64/Documents/UDEM/9NO_SEMESTRE/PEF/Analisis_tomografia_reconstruccion/Patient_B/Cortes_tomografías/Section_1/Mask_Bellaney_117.png"
    n_slices = 25
    output_directory = "/home/vector64/Documents/UDEM/9NO_SEMESTRE/PEF/Analisis_tomografia_reconstruccion/Patient_B/Cortes_tomografías/Section_02"
    num_start = 120
    increase = True
    
    generate_slices_from_mask(mask_path, n_slices, output_directory, num_start, increase)

if __name__ == "__main__":
    main()