#!/usr/bin/env python3
"""
Voxel processing module for 3D reconstruction.
Handles voxel data creation and morphological operations.
"""

import numpy as np
import warnings

warnings.filterwarnings('ignore')

try:
    from skimage.morphology import binary_closing, binary_opening, ball
    from scipy import ndimage
    SKIMAGE_AVAILABLE = True
    SCIPY_AVAILABLE = True
except ImportError:
    try:
        from skimage.morphology import binary_closing, binary_opening
        SKIMAGE_AVAILABLE = True
        SCIPY_AVAILABLE = False
    except ImportError:
        SKIMAGE_AVAILABLE = False
        SCIPY_AVAILABLE = False


class VoxelProcessor:
    """Handles voxel data creation and processing operations."""
    
    def __init__(self):
        self.voxel_data = None
    
    def create_voxel_data(self, mask_images: list, close_ends: bool = True) -> np.ndarray:
        """Create 3D voxel data from masks."""
        if not mask_images:
            raise ValueError("Load masks first, hmm.")
        
        self.voxel_data = np.stack(mask_images, axis=0)
        
        if close_ends:
            self.voxel_data = self._close_volume_ends(self.voxel_data)
        
        active = np.sum(self.voxel_data)
        print(f"Voxels: {self.voxel_data.shape}, active: {active:,}")
        
        return self.voxel_data
    
    def _close_volume_ends(self, voxel_data: np.ndarray) -> np.ndarray:
        """Close volume ends for watertight model."""
        closed_data = voxel_data.copy()
        
        if np.any(closed_data[0]):
            if SCIPY_AVAILABLE:
                closed_data[0] = ndimage.binary_fill_holes(closed_data[0])
            elif SKIMAGE_AVAILABLE:
                closed_data[0] = binary_closing(closed_data[0])
        
        if np.any(closed_data[-1]):
            if SCIPY_AVAILABLE:
                closed_data[-1] = ndimage.binary_fill_holes(closed_data[-1])
            elif SKIMAGE_AVAILABLE:
                closed_data[-1] = binary_closing(closed_data[-1])
        
        for z in range(1, closed_data.shape[0] - 1):
            if np.any(closed_data[z-1]) and np.any(closed_data[z+1]):
                intersection = np.logical_and(closed_data[z-1], closed_data[z+1])
                closed_data[z] = np.logical_or(closed_data[z], intersection)
        
        return closed_data
    
    def smooth_voxel_data(self, voxel_data: np.ndarray, iterations: int = 3, create_manifold: bool = True) -> np.ndarray:
        """Smooth voxel data with morphological operations."""
        if not SKIMAGE_AVAILABLE:
            return voxel_data
        
        smoothed_data = voxel_data.copy()
        
        try:
            if create_manifold:
                smoothed_data = binary_opening(smoothed_data)
            
            for i in range(iterations):
                smoothed_data = binary_closing(smoothed_data)
                        
        except Exception:
            for i in range(iterations):
                smoothed_data = binary_closing(smoothed_data)
        
        return smoothed_data
    
    def generate_point_cloud(self, voxel_data: np.ndarray, mm_per_pixel_x: float, 
                           mm_per_pixel_y: float, mm_per_slice: float, subsample_factor: int = 1) -> np.ndarray:
        """Generate point cloud from voxels."""
        z_coords, y_coords, x_coords = np.where(voxel_data)
        
        if subsample_factor > 1:
            indices = np.arange(0, len(z_coords), subsample_factor)
            z_coords = z_coords[indices]
            y_coords = y_coords[indices]
            x_coords = x_coords[indices]
        
        points_mm = np.column_stack([
            x_coords * mm_per_pixel_x,
            y_coords * mm_per_pixel_y,
            z_coords * mm_per_slice
        ])
        
        return points_mm