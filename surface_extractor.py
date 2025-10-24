#!/usr/bin/env python3
"""
Surface extraction module for 3D reconstruction.
Handles marching cubes and mesh processing.
"""

import numpy as np
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

try:
    from skimage import measure
    from scipy import ndimage
    SKIMAGE_AVAILABLE = True
    SCIPY_AVAILABLE = True
except ImportError:
    try:
        from skimage import measure
        SKIMAGE_AVAILABLE = True
        SCIPY_AVAILABLE = False
    except ImportError:
        SKIMAGE_AVAILABLE = False
        SCIPY_AVAILABLE = False


class SurfaceExtractor:
    """Handles surface extraction using marching cubes algorithm."""
    
    def __init__(self):
        pass
    
    def extract_manifold_surface(self, volume_data: np.ndarray, slice_depths: np.ndarray, 
                                mm_per_pixel_y: float, mm_per_pixel_x: float,
                                smooth: bool = True, manifold: bool = True, 
                                add_padding: bool = True) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Extract surface using marching cubes."""
        if not SKIMAGE_AVAILABLE:
            return None
        
        try:
            if manifold and add_padding:
                volume_data = self._add_volume_padding(volume_data)
            
            volume = volume_data.astype(float)
            
            if manifold and SCIPY_AVAILABLE:
                try:
                    from scipy.ndimage import gaussian_filter
                    volume = gaussian_filter(volume, sigma=0.5)
                except ImportError:
                    pass
            
            vertices, faces, normals, values = measure.marching_cubes(volume, level=0.5)
            
            if manifold:
                vertices[:, 0] -= 1
                vertices[:, 1] -= 1  
                vertices[:, 2] -= 1
            
            # Apply variable slice depths
            self._apply_variable_slice_depths(vertices, slice_depths, add_padding)
            vertices[:, 1] *= mm_per_pixel_y
            vertices[:, 2] *= mm_per_pixel_x
            
            if manifold:
                vertices, faces = self._ensure_manifold_mesh(vertices, faces)
            
            print(f"Surface: {len(vertices)} vertices, {len(faces)} faces")
            
            return vertices, faces
            
        except Exception:
            return None
    
    def _add_volume_padding(self, volume_data: np.ndarray, pad_size: int = 1) -> np.ndarray:
        """Add padding around volume."""
        padded_volume = np.pad(volume_data, pad_size, mode='constant', constant_values=False)
        return padded_volume
    
    def _apply_variable_slice_depths(self, vertices: np.ndarray, slice_depths: np.ndarray, add_padding: bool = True) -> None:
        """Apply variable slice depths to vertex z-coordinates."""
        if len(slice_depths) == 0:
            # No slice depths available, keep original coordinates
            return
            
        if add_padding:
            # Account for padding adjustment
            adjusted_slice_depths = np.concatenate([[slice_depths[0]], slice_depths, [slice_depths[-1]]])
        else:
            adjusted_slice_depths = slice_depths
        
        # Calculate cumulative depths for z-coordinate mapping
        cumulative_depths = np.cumsum(np.concatenate([[0], adjusted_slice_depths]))
        
        # Map vertex z-coordinates using linear interpolation
        for i in range(len(vertices)):
            z_idx = vertices[i, 0]
            if z_idx < 0:
                vertices[i, 0] = 0
            elif z_idx >= len(cumulative_depths) - 1:
                vertices[i, 0] = cumulative_depths[-1]
            else:
                # Linear interpolation between slices
                lower_idx = int(np.floor(z_idx))
                upper_idx = lower_idx + 1
                fraction = z_idx - lower_idx
                
                if upper_idx < len(cumulative_depths):
                    vertices[i, 0] = cumulative_depths[lower_idx] + fraction * adjusted_slice_depths[min(lower_idx, len(adjusted_slice_depths)-1)]
                else:
                    vertices[i, 0] = cumulative_depths[lower_idx]
    
    def _ensure_manifold_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Clean mesh for manifold properties."""
        unique_vertices, inverse_indices = np.unique(vertices, axis=0, return_inverse=True)
        new_faces = inverse_indices[faces]
        
        valid_faces = []
        for face in new_faces:
            if len(np.unique(face)) == 3:
                valid_faces.append(face)
        
        new_faces = np.array(valid_faces)
        return unique_vertices, new_faces
    
    def calculate_mesh_volume(self, vertices: np.ndarray, faces: np.ndarray) -> float:
        """Calculate mesh volume using divergence theorem."""
        volume = 0.0
        for face in faces:
            v0 = vertices[face[0]]
            v1 = vertices[face[1]] 
            v2 = vertices[face[2]]
            
            tetrahedron_volume = np.dot(v0, np.cross(v1, v2)) / 6.0
            volume += tetrahedron_volume
        
        return abs(volume)
    
    def calculate_surface_area(self, vertices: np.ndarray, faces: np.ndarray) -> float:
        """Calculate surface area from triangular faces."""
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        cross_product = np.cross(v1 - v0, v2 - v0)
        triangle_areas = 0.5 * np.linalg.norm(cross_product, axis=1)
        return np.sum(triangle_areas)