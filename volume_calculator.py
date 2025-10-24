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
    
    def calculate_voxel_volume_variable_depth(self, voxel_data: np.ndarray, mm_per_pixel_x: float, 
                                            mm_per_pixel_y: float, slice_depths: np.ndarray) -> float:
        """Calculate volume from voxel data with variable slice depths in mm³."""
        if len(slice_depths) == 0:
            return 0.0
            
        total_volume = 0.0
        
        for z in range(min(voxel_data.shape[0], len(slice_depths))):
            slice_volume = mm_per_pixel_x * mm_per_pixel_y * slice_depths[z]
            total_volume += np.sum(voxel_data[z]) * slice_volume
        
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
    
    def calculate_bounding_box_variable_depth(self, voxel_data: np.ndarray, mm_per_pixel_x: float, 
                                            mm_per_pixel_y: float, slice_depths: np.ndarray) -> dict:
        """Calculate bounding box in mm with variable slice depths."""
        z_coords, y_coords, x_coords = np.where(voxel_data)
        
        if len(z_coords) == 0 or len(slice_depths) == 0:
            return {
                'x': (0, 0),
                'y': (0, 0), 
                'z': (0, 0),
                'dimensions': (0, 0, 0)
            }
        
        bbox_x = (x_coords.min() * mm_per_pixel_x, x_coords.max() * mm_per_pixel_x)
        bbox_y = (y_coords.min() * mm_per_pixel_y, y_coords.max() * mm_per_pixel_y)
        
        # Calculate z bounds using cumulative depths
        cumulative_depths = np.cumsum(np.concatenate([[0], slice_depths]))
        
        z_min = cumulative_depths[z_coords.min()]
        z_max = cumulative_depths[min(z_coords.max() + 1, len(cumulative_depths) - 1)]
        
        bbox_z = (z_min, z_max)
        
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
                                 slice_depths: np.ndarray, x_length_mm: float,
                                 y_length_mm: float, total_depth_mm: float) -> dict:
        """Analyze comprehensive object properties with variable slice depths."""
        voxel_volume = self.calculate_voxel_volume_variable_depth(voxel_data, mm_per_pixel_x, mm_per_pixel_y, slice_depths)
        bbox_info = self.calculate_bounding_box_variable_depth(voxel_data, mm_per_pixel_x, mm_per_pixel_y, slice_depths)
        
        primary_volume = mesh_volume if mesh_volume is not None else processed_volume
        
        # Calculate total actual depth including Side_0 and Side_2
        total_actual_depth = np.sum(slice_depths)
        density = self.calculate_density(primary_volume, x_length_mm, y_length_mm, total_actual_depth)
        
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