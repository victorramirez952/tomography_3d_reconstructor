#!/usr/bin/env python3
"""
Simple Ellipsoid Slice Generator
================================

This script generates ellipsoid slices from a middle slice image
following the requirements in Pseudocode.md
"""

import os
import shutil
from ellipsoid_slice_generator import EllipsoidSliceGenerator

def generate_slices_from_mask(mask_path, n_slices, output_directory):
    """
    Function to generate n slices from the mask, considering these masks will be used for 3D reconstruction.
    Ensure the new n slices will make a half-ellipsoid 3d shape when stacked together, keeping the shape of the original mask in mind.
    The original shape of the mask will be the base of the half-ellipsoid.
    """
    # Clean output_directory if it exists, else create it
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
        print(f"Cleaned existing directory: {output_directory}")
    os.makedirs(output_directory, exist_ok=True)
    print(f"Created output directory: {output_directory}")
    
    # Check if image exists
    if not os.path.exists(mask_path):
        print(f"Error: Image '{mask_path}' not found.")
        return
    
    try:
        print(f"Processing image: {mask_path}")
        print(f"Generating {n_slices} slices...")
        
        # Create generator and generate slices with sequential naming for half-ellipsoid
        generator = EllipsoidSliceGenerator(mask_path)
        
        # Hardcoded values as per pseudocode requirements
        num_start = 79  # Some number (integer and can be negative)
        increase = False  # true/false
        
        slice_files = generator.generate_slices_half_ellipsoid(n_slices, output_directory, num_start, increase)
        
        print(f"\n✓ Successfully generated {len(slice_files)} slices")
        print(f"✓ Slices saved in '{output_directory}' directory")
        print("✓ Sequential naming following pseudocode requirements")
        
        # Show some statistics
        params = generator.ellipse_params
        print(f"\nEllipse parameters:")
        print(f"  Semi-major axis: {params['semi_major_axis']:.1f} pixels")
        print(f"  Semi-minor axis: {params['semi_minor_axis']:.1f} pixels")
        print(f"  Center: ({params['center'][0]:.1f}, {params['center'][1]:.1f})")
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    # Hardcoded variables as required by Pseudocode.md
    mask_path = "/home/vector64/Documents/UDEM/9NO_SEMESTRE/PEF/Analisis_tomografia_reconstruccion/Patient_M/Cortes_tomografias/Section_1/Mask_Maria_79.png"
    n_slices = 25  # Number of slices to generate
    output_directory = "/home/vector64/Documents/UDEM/9NO_SEMESTRE/PEF/Analisis_tomografia_reconstruccion/Patient_M/Cortes_tomografia/Section_0"
    
    # Generate slices following the pseudocode requirements
    generate_slices_from_mask(mask_path, n_slices, output_directory)

if __name__ == "__main__":
    main()