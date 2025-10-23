#!/usr/bin/env python3
"""
Volume calculator module for 3D reconstruction.
Handles volume calculations and object analysis.
"""

import numpy as np


class VolumeCalculator:
    """Handles volume calculations and object property analysis."""
    
    def __init__(self):
        pass
    
    def calculate_voxel_volume(self, voxel_data: np.ndarray, mm_per_pixel_x: float, 
                              mm_per_pixel_y: float, mm_per_slice: float) -> float:
        """Calculate volume from voxel data in mm³."""
        voxel_volume = mm_per_pixel_x * mm_per_pixel_y * mm_per_slice
        total_volume = np.sum(voxel_data) * voxel_volume
        return total_volume
    
    def calculate_bounding_box(self, voxel_data: np.ndarray, mm_per_pixel_x: float, 
                              mm_per_pixel_y: float, mm_per_slice: float) -> dict:
        """Calculate bounding box in mm."""
        z_coords, y_coords, x_coords = np.where(voxel_data)
        
        bbox_x = (x_coords.min() * mm_per_pixel_x, x_coords.max() * mm_per_pixel_x)
        bbox_y = (y_coords.min() * mm_per_pixel_y, y_coords.max() * mm_per_pixel_y)
        bbox_z = (z_coords.min() * mm_per_slice, z_coords.max() * mm_per_slice)
        
        bbox_dimensions = (
            bbox_x[1] - bbox_x[0],
            bbox_y[1] - bbox_y[0], 
            bbox_z[1] - bbox_z[0]
        )
        
        return {
            'x': bbox_x,
            'y': bbox_y, 
            'z': bbox_z,
            'dimensions': bbox_dimensions
        }
    
    def calculate_density(self, volume: float, x_length_mm: float, 
                         y_length_mm: float, total_depth_mm: float) -> float:
        """Calculate object density as percentage of total space."""
        total_possible_volume = x_length_mm * y_length_mm * total_depth_mm
        return volume / total_possible_volume
    
    def analyze_object_properties(self, voxel_data: np.ndarray, processed_volume: float,
                                 mesh_volume: float, surface_area: float,
                                 mm_per_pixel_x: float, mm_per_pixel_y: float, 
                                 mm_per_slice: float, x_length_mm: float,
                                 y_length_mm: float, total_depth_mm: float) -> dict:
        """Analyze comprehensive object properties."""
        voxel_volume = self.calculate_voxel_volume(voxel_data, mm_per_pixel_x, mm_per_pixel_y, mm_per_slice)
        bbox_info = self.calculate_bounding_box(voxel_data, mm_per_pixel_x, mm_per_pixel_y, mm_per_slice)
        
        primary_volume = mesh_volume if mesh_volume is not None else processed_volume
        density = self.calculate_density(primary_volume, x_length_mm, y_length_mm, total_depth_mm)
        
        print(f"Volume: {primary_volume:.4f} mm³")
        print(f"Dimensions: {bbox_info['dimensions'][0]:.2f} x {bbox_info['dimensions'][1]:.2f} x {bbox_info['dimensions'][2]:.2f} mm")
        if surface_area:
            print(f"Surface Area: {surface_area:.4f} mm²")
        print(f"Density: {100*density:.1f}% of total space")
        
        return {
            'volume_mm3': primary_volume,
            'voxel_volume_mm3': voxel_volume,
            'processed_voxel_volume_mm3': processed_volume,
            'mesh_volume_mm3': mesh_volume,
            'bounding_box': {'x': bbox_info['x'], 'y': bbox_info['y'], 'z': bbox_info['z']},
            'dimensions': bbox_info['dimensions'],
            'surface_area_mm2': surface_area,
            'density': density
        }