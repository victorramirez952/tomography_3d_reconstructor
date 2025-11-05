#!/usr/bin/env python3
"""
GLB file exporter module for 3D reconstruction.
Handles exporting 3D models to GLB (binary glTF) format.
"""

import numpy as np
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False


class GLBExporter:
    """Handles exporting 3D models to GLB file format."""
    
    def __init__(self):
        pass
    
    def export_to_glb(self, vertices: np.ndarray, faces: np.ndarray, 
                     filename: str = "tomography_model.glb",
                     vertex_colors: Optional[np.ndarray] = None) -> bool:
        """Export 3D model to GLB format with optional vertex colors."""
        if not TRIMESH_AVAILABLE:
            print("Trimesh not available, install required")
            return False
        
        try:
            # Create trimesh mesh object
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, 
                                  vertex_colors=vertex_colors)
            
            # Fix normals and ensure manifold
            mesh.fix_normals()
            
            # Export to GLB format
            mesh.export(filename, file_type='glb')
            
            print(f"Model exported: {filename}")
            return True
            
        except Exception as e:
            print(f"Export failed: {e}")
            return False
    
    def create_layer_colors(self, vertices: np.ndarray, slice_depths: np.ndarray,
                           first_section1_slice: int, last_section1_slice: int,
                           highlight_thickness_mm: float = 1.0) -> np.ndarray:
        """Create vertex colors with highlighted layers for Section_1 boundaries.
        
        Args:
            vertices: Vertex positions (N, 3) with z-axis as depth
            slice_depths: Cumulative depths for each slice
            first_section1_slice: Index of first Section_1 slice
            last_section1_slice: Index of last Section_1 slice
            highlight_thickness_mm: Thickness of highlight in mm
        
        Returns:
            Vertex colors (N, 4) in RGBA format (0-255)
        """
        num_vertices = len(vertices)
        colors = np.full((num_vertices, 4), [200, 200, 200, 255], dtype=np.uint8)  # Default gray
        
        # Calculate cumulative depths for z-coordinate mapping
        cumulative_depths = np.cumsum(np.concatenate([[0], slice_depths]))
        
        # Get depth range for first Section_1 slice (red highlight)
        if first_section1_slice < len(cumulative_depths) - 1:
            first_depth_start = cumulative_depths[first_section1_slice]
            first_depth_end = first_depth_start + highlight_thickness_mm
            
            # Mark vertices in red highlight zone
            in_first_zone = (vertices[:, 0] >= first_depth_start) & (vertices[:, 0] <= first_depth_end)
            colors[in_first_zone] = [255, 0, 0, 255]  # Red
        
        # Get depth range for last Section_1 slice (blue highlight)
        if last_section1_slice < len(cumulative_depths) - 1:
            last_depth_start = cumulative_depths[last_section1_slice]
            last_depth_end = last_depth_start + highlight_thickness_mm
            
            # Mark vertices in blue highlight zone
            in_last_zone = (vertices[:, 0] >= last_depth_start) & (vertices[:, 0] <= last_depth_end)
            colors[in_last_zone] = [0, 0, 255, 255]  # Blue
        
        return colors
