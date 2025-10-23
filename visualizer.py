#!/usr/bin/env python3
"""
Visualization module for 3D reconstruction.
Handles matplotlib and plotly visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import warnings

warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class Visualizer:
    """Handles 3D visualizations for reconstruction data."""
    
    def __init__(self):
        pass
    
    def visualize_3d_solid_matplotlib(self, vertices: np.ndarray, faces: np.ndarray):
        """Create 3D solid visualization."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                       triangles=faces, alpha=0.8, shade=True, 
                       cmap='viridis', linewidth=0.1)
        
        ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
        ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
        ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('3D Solid Reconstruction')
        
        max_range = np.array([vertices[:, 0].max() - vertices[:, 0].min(),
                             vertices[:, 1].max() - vertices[:, 1].min(),
                             vertices[:, 2].max() - vertices[:, 2].min()]).max() / 2.0
        
        mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
        mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
        mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.show()

    def visualize_3d_voxels_matplotlib(self, voxel_data: np.ndarray):
        """Create voxel visualization."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.voxels(voxel_data, facecolors='lightblue', edgecolors='darkblue', alpha=0.7)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('3D Voxel Reconstruction')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_3d_interactive_mesh(self, vertices: Optional[np.ndarray] = None, 
                                    faces: Optional[np.ndarray] = None,
                                    points: Optional[np.ndarray] = None,
                                    save_path: str = "3d_reconstruction_interactive.html"):
        """Create interactive 3D visualization."""
        if not PLOTLY_AVAILABLE:
            return
        
        if points is not None:
            # Point cloud visualization
            fig = go.Figure(data=[go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1], 
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=points[:, 2],
                    colorscale='Viridis',
                    colorbar=dict(title="Depth (mm)"),
                    opacity=0.8
                ),
                name='Point Cloud'
            )])
        elif vertices is not None and faces is not None:
            # Mesh visualization
            fig = go.Figure(data=[go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                colorscale='Viridis',
                intensity=vertices[:, 2],
                colorbar=dict(title="Depth (mm)"),
                opacity=0.8,
                name='3D Mesh'
            )])
        else:
            return
        
        fig.update_layout(
            title='Interactive 3D Reconstruction',
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='cube'
            ),
            width=900,
            height=700
        )
        
        fig.write_html(save_path)