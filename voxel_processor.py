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
        self.side_0_count = 0
        self.side_1_count = 0
        self.side_2_count = 0
    
    def create_voxel_data(self, mask_images: list, close_ends: bool = True, 
                         side_0_count: int = 0, side_1_count: int = 0, side_2_count: int = 0) -> np.ndarray:
        """Create 3D voxel data from masks with side information."""
        if not mask_images:
            raise ValueError("Load masks first, hmm.")
        
        self.side_0_count = side_0_count
        self.side_1_count = side_1_count
        self.side_2_count = side_2_count
        
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
                           mm_per_pixel_y: float, slice_depths: np.ndarray, subsample_factor: int = 1) -> np.ndarray:
        """Generate point cloud from voxels with variable slice depths."""
        z_coords, y_coords, x_coords = np.where(voxel_data)
        
        if subsample_factor > 1:
            indices = np.arange(0, len(z_coords), subsample_factor)
            z_coords = z_coords[indices]
            y_coords = y_coords[indices]
            x_coords = x_coords[indices]
        
        # Calculate cumulative depths for z-coordinate mapping
        cumulative_depths = np.cumsum(np.concatenate([[0], slice_depths]))
        
        # Map z coordinates to actual depths
        z_coords_mm = []
        for z in z_coords:
            if z < len(slice_depths):
                z_coords_mm.append(cumulative_depths[z] + slice_depths[z] / 2)  # Center of slice
            else:
                z_coords_mm.append(cumulative_depths[-1])
        
        points_mm = np.column_stack([
            np.array(z_coords_mm),
            y_coords * mm_per_pixel_y,
            x_coords * mm_per_pixel_x
        ])
        
        return points_mm
    
    def calculate_slice_depths(self, total_depth_mm: float) -> np.ndarray:
        """Calculate depth per slice based on side structure."""
        total_slices = self.side_0_count + self.side_1_count + self.side_2_count
        
        if self.side_1_count == 0 or total_slices == 0:
            # Fallback to uniform distribution if no Side_1 or no slices at all
            if total_slices == 0:
                return np.array([])
            return np.full(total_slices, total_depth_mm / total_slices)
        
        # Side_1 represents the main body with TOTAL_DEPTH_MM
        side_1_depth_per_slice = total_depth_mm / self.side_1_count
        
        # Side_0 and Side_2 each have total depth = 2 * side_1_depth_per_slice
        side_0_2_total_depth = 2 * side_1_depth_per_slice
        side_0_depth_per_slice = side_0_2_total_depth / self.side_0_count if self.side_0_count > 0 else 0
        side_2_depth_per_slice = side_0_2_total_depth / self.side_2_count if self.side_2_count > 0 else 0
        
        depths = []
        
        # Add Side_0 depths (first slices in sequence)
        for i in range(self.side_0_count):
            depths.append(side_0_depth_per_slice)
        
        # Add Side_1 depths (middle slices in sequence)  
        for i in range(self.side_1_count):
            depths.append(side_1_depth_per_slice)
        
        # Add Side_2 depths (last slices in sequence)
        for i in range(self.side_2_count):
            depths.append(side_2_depth_per_slice)
        
        # Verify sequential order matches loaded images
        print(f"Slice depth sequence: Side_0[0-{self.side_0_count-1}], Side_1[{self.side_0_count}-{self.side_0_count+self.side_1_count-1}], Side_2[{self.side_0_count+self.side_1_count}-{len(depths)-1}]")
        
        return np.array(depths)