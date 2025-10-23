#!/usr/bin/env python3
"""
Tomography 3D Reconstruction Script

This script loads all mask images (files starting with "Mask_") from the current directory
and performs 3D reconstruction of an irregular object from tomographic slices.

The script provides multiple 3D reconstruction techniques:
1. Voxel-based reconstruction
2. Marching cubes surface extraction
3. Point cloud generation
4. Volume rendering visualization

Requirements:
- All mask images must be the same size
- Known x, y dimensions in mm (hardcoded)
- Known depth covered by all images (hardcoded)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import glob
import os
from typing import List, Tuple, Optional
import warnings

# Import configuration
import config

warnings.filterwarnings('ignore')

try:
    from skimage import measure
    from skimage.morphology import binary_closing, binary_opening, ball
    from scipy import ndimage
    SKIMAGE_AVAILABLE = True
    SCIPY_AVAILABLE = True
except ImportError:
    try:
        from skimage import measure  
        from skimage.morphology import binary_closing, binary_opening
        SKIMAGE_AVAILABLE = True
        SCIPY_AVAILABLE = False
        print("Warning: scipy not available. Some manifold features will be disabled.")
    except ImportError:
        SKIMAGE_AVAILABLE = False
        SCIPY_AVAILABLE = False
        print("Warning: scikit-image not available. Some advanced features will be disabled.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Interactive 3D visualization will be disabled.")




class Tomography3DReconstruction:
    def __init__(self, x_length_mm: float, y_length_mm: float, total_depth_mm: float):
        """
        Initialize the 3D reconstruction system.
        
        Args:
            x_length_mm (float): Width of images in mm
            y_length_mm (float): Height of images in mm  
            total_depth_mm (float): Total depth covered by all slices in mm
        """
        # Hardcoded physical dimensions
        self.x_length_mm = x_length_mm
        self.y_length_mm = y_length_mm
        self.total_depth_mm = total_depth_mm
        
        # Data storage
        self.mask_files = []
        self.mask_images = []
        self.voxel_data = None
        
        # Image properties (will be set after loading)
        self.image_width = None
        self.image_height = None
        self.num_slices = 0
        
        # Resolution calculations (will be calculated after loading images)
        self.mm_per_pixel_x = None
        self.mm_per_pixel_y = None
        self.mm_per_slice = None
        
        print(f"3D Reconstruction System Initialized")
        print(f"Physical dimensions: {x_length_mm:.2f} x {y_length_mm:.2f} x {total_depth_mm:.2f} mm")
        
    def load_mask_images(self, directory: str = ".", threshold: int = 200) -> bool:
        """
        Load all mask images starting with 'Mask_' from the specified directory.
        
        Args:
            directory (str): Directory to search for mask images
            threshold (int): Threshold for binarizing mask images (0-255)
            
        Returns:
            bool: True if images loaded successfully, False otherwise
        """
        try:
            # Find all mask files
            mask_pattern = os.path.join(directory, "Mask_*.png")
            self.mask_files = sorted(glob.glob(mask_pattern))
            
            if not self.mask_files:
                print(f"No mask files found matching pattern: {mask_pattern}")
                return False
                
            print(f"Found {len(self.mask_files)} mask files:")
            for i, file in enumerate(self.mask_files):
                print(f"  {i+1}: {os.path.basename(file)}")
            
            # Load images
            self.mask_images = []
            first_image = None
            
            for i, file_path in enumerate(self.mask_files):
                # Load image in grayscale
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Error loading image: {file_path}")
                    continue
                
                # Validate dimensions consistency
                if first_image is None:
                    first_image = img
                    self.image_height, self.image_width = img.shape
                    print(f"Image dimensions: {self.image_width}x{self.image_height} pixels")
                else:
                    if img.shape != first_image.shape:
                        print(f"Warning: Image {file_path} has different dimensions!")
                        continue
                
                # Binarize the mask (convert to boolean)
                binary_mask = img >= threshold
                self.mask_images.append(binary_mask)
            
            self.num_slices = len(self.mask_images)
            
            if self.num_slices == 0:
                print("No valid images loaded!")
                return False
            
            # Calculate resolution
            self.mm_per_pixel_x = self.x_length_mm / self.image_width
            self.mm_per_pixel_y = self.y_length_mm / self.image_height
            self.mm_per_slice = self.total_depth_mm / self.num_slices
            
            print(f"\nLoaded {self.num_slices} images successfully")
            print(f"Resolution:")
            print(f"  X: {self.mm_per_pixel_x:.4f} mm/pixel")
            print(f"  Y: {self.mm_per_pixel_y:.4f} mm/pixel") 
            print(f"  Z: {self.mm_per_slice:.4f} mm/slice")
            
            return True
            
        except Exception as e:
            print(f"Error loading mask images: {e}")
            return False
    
    def create_voxel_data(self, close_ends: bool = True) -> np.ndarray:
        """
        Create 3D voxel data from the loaded 2D mask images.
        
        Args:
            close_ends (bool): Whether to add end caps to create a closed volume
        
        Returns:
            np.ndarray: 3D boolean array representing the voxel data
        """
        if not self.mask_images:
            raise ValueError("No mask images loaded. Call load_mask_images() first.")
        
        # Stack all 2D masks into a 3D volume
        self.voxel_data = np.stack(self.mask_images, axis=0)
        
        if close_ends:
            self.voxel_data = self._close_volume_ends(self.voxel_data)
        
        print(f"Voxel data created: {self.voxel_data.shape}")
        print(f"Total voxels: {self.voxel_data.size:,}")
        print(f"Active voxels: {np.sum(self.voxel_data):,} ({100*np.sum(self.voxel_data)/self.voxel_data.size:.2f}%)")
        
        return self.voxel_data
    
    def _close_volume_ends(self, voxel_data: np.ndarray) -> np.ndarray:
        """
        Close the ends of the voxel volume to create a solid, watertight model.
        
        Args:
            voxel_data (np.ndarray): Original voxel data
            
        Returns:
            np.ndarray: Voxel data with closed ends
        """
        closed_data = voxel_data.copy()
        
        # For first and last slices, if there are any active voxels,
        # fill the internal regions to create solid caps
        
        # Process first slice (z=0)
        if np.any(closed_data[0]):
            # Fill holes in the first slice to create a solid cap
            if SCIPY_AVAILABLE:
                closed_data[0] = ndimage.binary_fill_holes(closed_data[0])
            else:
                # Simple fill by expanding the boundary slightly
                if SKIMAGE_AVAILABLE:
                    closed_data[0] = binary_closing(closed_data[0])
        
        # Process last slice (z=-1)  
        if np.any(closed_data[-1]):
            # Fill holes in the last slice to create a solid cap
            if SCIPY_AVAILABLE:
                closed_data[-1] = ndimage.binary_fill_holes(closed_data[-1])
            else:
                # Simple fill by expanding the boundary slightly
                if SKIMAGE_AVAILABLE:
                    closed_data[-1] = binary_closing(closed_data[-1])
        
        # Additionally, ensure continuity by filling gaps between slices
        # This creates a more solid internal structure
        for z in range(1, closed_data.shape[0] - 1):
            if np.any(closed_data[z-1]) and np.any(closed_data[z+1]):
                # If neighboring slices have content, ensure current slice
                # has at least the intersection of the neighbors
                intersection = np.logical_and(closed_data[z-1], closed_data[z+1])
                closed_data[z] = np.logical_or(closed_data[z], intersection)
        
        print("Applied end closure for watertight volume")
        return closed_data
    
    def calculate_volume(self, use_processed_data: bool = False) -> float:
        """
        Calculate the total volume of the reconstructed object in mm³.
        
        Args:
            use_processed_data (bool): If True, use the same processed data as the OBJ export
        
        Returns:
            float: Volume in mm³
        """
        if self.voxel_data is None:
            self.create_voxel_data(close_ends=True)
        
        if use_processed_data:
            # Use the same processing pipeline as the OBJ export for consistency
            volume_data = self.smooth_voxel_data(iterations=config.SMOOTHING_ITERATIONS, create_manifold=config.CREATE_MANIFOLD)
            calculation_note = "processed (matching OBJ model)"
        else:
            # Use original closed voxel data
            volume_data = self.voxel_data
            calculation_note = "original voxel data"
        
        # Volume per voxel in mm³
        voxel_volume = self.mm_per_pixel_x * self.mm_per_pixel_y * self.mm_per_slice
        
        # Total volume
        total_volume = np.sum(volume_data) * voxel_volume
        
        print(f"\nVolume Calculation ({calculation_note}):")
        print(f"  Voxel volume: {voxel_volume:.6f} mm³")
        print(f"  Active voxels: {np.sum(volume_data):,}")
        print(f"  Total volume: {total_volume:.4f} mm³")
        
        return total_volume
    
    def calculate_mesh_volume_from_obj(self) -> Optional[float]:
        """
        Calculate volume using the same mesh data that gets exported to OBJ.
        This provides the most accurate volume for the actual 3D model.
        
        Returns:
            float: Volume in mm³ or None if surface extraction fails
        """
        surface_result = self.extract_manifold_surface(smooth=True, manifold=True)
        if surface_result is None:
            print("Cannot calculate mesh volume - surface extraction failed")
            return None
        
        vertices, faces = surface_result
        
        # Calculate volume using divergence theorem (sum of tetrahedra volumes)
        volume = 0.0
        
        for face in faces:
            # Get triangle vertices
            v0 = vertices[face[0]]
            v1 = vertices[face[1]] 
            v2 = vertices[face[2]]
            
            # Calculate signed volume of tetrahedron formed by origin and triangle
            # V = (1/6) * dot(v0, cross(v1, v2))
            tetrahedron_volume = np.dot(v0, np.cross(v1, v2)) / 6.0
            volume += tetrahedron_volume
        
        # Take absolute value to handle orientation
        volume = abs(volume)
        
        print(f"\nMesh Volume Calculation (OBJ-accurate):")
        print(f"  Triangular faces: {len(faces):,}")
        print(f"  Mesh volume: {volume:.4f} mm³")
        
        return volume
    
    def smooth_voxel_data(self, iterations: int = 3, create_manifold: bool = True) -> np.ndarray:
        """
        Smooth the voxel data using morphological operations to create a manifold solid 3D model.
        
        Args:
            iterations (int): Number of smoothing iterations
            create_manifold (bool): Whether to apply additional operations for manifold creation
            
        Returns:
            np.ndarray: Smoothed and manifold voxel data
        """
        if not SKIMAGE_AVAILABLE:
            print("Smoothing requires scikit-image. Using original voxel data.")
            return self.voxel_data
        
        if self.voxel_data is None:
            self.create_voxel_data()
        
        smoothed_data = self.voxel_data.copy()
        
        try:
            if create_manifold:
                # First apply opening to remove noise and thin connections
                smoothed_data = binary_opening(smoothed_data)
                print("Applied morphological opening for manifold preparation")
            
            # Apply morphological closing to fill holes and smooth surfaces
            for i in range(iterations):
                smoothed_data = binary_closing(smoothed_data)
                print(f"Smoothing iteration {i+1}/{iterations} completed")
            
            if create_manifold:
                # Additional closing with larger structure for manifold solidity
                try:
                    struct_elem = ball(2)  # Create spherical structuring element
                    smoothed_data = binary_closing(smoothed_data, structure=struct_elem)
                    print("Applied manifold closing with spherical structuring element")
                except:
                    # Fallback with basic structuring element
                    smoothed_data = binary_closing(smoothed_data)
                    print("Applied basic manifold closing (ball structure not available)")
                
                # Fill any remaining holes in individual slices
                if SCIPY_AVAILABLE:
                    for z in range(smoothed_data.shape[0]):
                        smoothed_data[z] = ndimage.binary_fill_holes(smoothed_data[z])
                    print("Filled holes in individual slices for manifold integrity")
                else:
                    print("Scipy not available - skipping hole filling")
            
            print(f"Manifold voxel processing complete. Active voxels: {np.sum(smoothed_data):,}")
            
        except Exception as e:
            print(f"Error in manifold processing: {e}. Using basic smoothing.")
            # Fallback to basic smoothing
            for i in range(iterations):
                smoothed_data = binary_closing(smoothed_data)
                print(f"Fallback smoothing iteration {i+1}/{iterations} completed")
        
        return smoothed_data

    def extract_manifold_surface(self, smooth: bool = True, manifold: bool = True) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Extract a manifold surface using marching cubes algorithm with additional processing.
        
        Args:
            smooth (bool): Whether to apply smoothing before surface extraction
            manifold (bool): Whether to ensure manifold properties and closed surface
        
        Returns:
            Tuple of (vertices, faces) or None if skimage not available
        """
        if not SKIMAGE_AVAILABLE:
            print("Marching cubes requires scikit-image. Please install: pip install scikit-image")
            return None
        
        if self.voxel_data is None:
            self.create_voxel_data(close_ends=True)
        
        try:
            # Use smoothed or original data with manifold creation
            if smooth:
                volume_data = self.smooth_voxel_data(iterations=config.SMOOTHING_ITERATIONS, create_manifold=manifold and config.CREATE_MANIFOLD)
            else:
                volume_data = self.voxel_data
            
            # Add padding around the volume to ensure closed surface
            if manifold and config.ADD_VOLUME_PADDING:
                volume_data = self._add_volume_padding(volume_data)
            
            # Convert boolean to float for marching cubes
            volume = volume_data.astype(float)
            
            # Apply gaussian smoothing for better manifold surface
            if manifold and SCIPY_AVAILABLE:
                try:
                    from scipy.ndimage import gaussian_filter
                    volume = gaussian_filter(volume, sigma=0.5)
                    print("Applied Gaussian smoothing for manifold surface")
                except ImportError:
                    print("Gaussian smoothing not available")
            
            # Apply marching cubes
            vertices, faces, normals, values = measure.marching_cubes(volume, level=0.5)
            
            # Adjust coordinates if padding was added
            if manifold:
                # Account for the padding offset (1 voxel on each side)
                vertices[:, 0] -= 1  # Z offset
                vertices[:, 1] -= 1  # Y offset  
                vertices[:, 2] -= 1  # X offset
            
            # Scale vertices to real-world coordinates
            vertices[:, 0] *= self.mm_per_slice  # Z dimension
            vertices[:, 1] *= self.mm_per_pixel_y  # Y dimension  
            vertices[:, 2] *= self.mm_per_pixel_x  # X dimension
            
            if manifold:
                vertices, faces = self._ensure_manifold_mesh(vertices, faces)
            
            print(f"Manifold surface extracted: {len(vertices)} vertices, {len(faces)} faces")
            
            return vertices, faces
            
        except Exception as e:
            print(f"Error in manifold surface extraction: {e}")
            return None
    
    def _add_volume_padding(self, volume_data: np.ndarray, pad_size: int = 1) -> np.ndarray:
        """
        Add padding around the volume to ensure closed surface extraction.
        
        Args:
            volume_data (np.ndarray): Original volume data
            pad_size (int): Number of voxels to pad on each side
            
        Returns:
            np.ndarray: Padded volume data
        """
        # Add padding of False (empty) voxels around the volume
        padded_volume = np.pad(volume_data, pad_size, mode='constant', constant_values=False)
        print(f"Added padding: {volume_data.shape} -> {padded_volume.shape}")
        return padded_volume
    
    def _ensure_manifold_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensure the mesh is a proper manifold by removing duplicate vertices and degenerate faces.
        
        Args:
            vertices (np.ndarray): Mesh vertices
            faces (np.ndarray): Mesh faces
            
        Returns:
            Tuple of cleaned (vertices, faces)
        """
        # Remove duplicate vertices
        unique_vertices, inverse_indices = np.unique(vertices, axis=0, return_inverse=True)
        
        # Update face indices to use unique vertices
        new_faces = inverse_indices[faces]
        
        # Remove degenerate faces (where all vertices are the same)
        valid_faces = []
        for face in new_faces:
            if len(np.unique(face)) == 3:  # Triangle should have 3 unique vertices
                valid_faces.append(face)
        
        new_faces = np.array(valid_faces)
        
        print(f"Manifold cleanup: {len(vertices)} -> {len(unique_vertices)} vertices, "
              f"{len(faces)} -> {len(new_faces)} faces")
        
        return unique_vertices, new_faces

    def extract_surface_marching_cubes(self, smooth: bool = True) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Extract the surface using marching cubes algorithm (legacy method).
        Use extract_manifold_surface for better manifold results.
        """
        return self.extract_manifold_surface(smooth=smooth, manifold=False)
    
    def generate_point_cloud(self, subsample_factor: int = 1) -> np.ndarray:
        """
        Generate a point cloud from the voxel data.
        
        Args:
            subsample_factor (int): Factor to subsample points (1 = all points)
            
        Returns:
            np.ndarray: Point cloud coordinates in mm
        """
        if self.voxel_data is None:
            self.create_voxel_data()
        
        # Find all active voxel coordinates
        z_coords, y_coords, x_coords = np.where(self.voxel_data)
        
        # Subsample if requested
        if subsample_factor > 1:
            indices = np.arange(0, len(z_coords), subsample_factor)
            z_coords = z_coords[indices]
            y_coords = y_coords[indices]
            x_coords = x_coords[indices]
        
        # Convert to real-world coordinates (in mm)
        points_mm = np.column_stack([
            x_coords * self.mm_per_pixel_x,
            y_coords * self.mm_per_pixel_y,
            z_coords * self.mm_per_slice
        ])
        
        print(f"Point cloud generated: {len(points_mm):,} points")
        
        return points_mm
    
    def visualize_slices(self, num_slices_to_show: int = 9, save_path: str = "slice_visualization.png"):
        """
        Visualize a selection of 2D slices.
        
        Args:
            num_slices_to_show (int): Number of slices to display
            save_path (str): Path to save the visualization
        """
        if not self.mask_images:
            print("No mask images to visualize")
            return
        
        # Select slices evenly distributed
        slice_indices = np.linspace(0, len(self.mask_images)-1, 
                                  min(num_slices_to_show, len(self.mask_images)), 
                                  dtype=int)
        
        # Calculate subplot grid
        cols = min(3, len(slice_indices))
        rows = (len(slice_indices) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, slice_idx in enumerate(slice_indices):
            ax = axes[i] if len(slice_indices) > 1 else axes
            
            ax.imshow(self.mask_images[slice_idx], cmap='gray')
            ax.set_title(f'Slice {slice_idx+1}/{len(self.mask_images)}\n'
                        f'Z = {slice_idx * self.mm_per_slice:.2f} mm')
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(len(slice_indices), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Slice visualization saved to: {save_path}")
    
    def visualize_3d_solid_matplotlib(self, smooth: bool = True, save_path: str = "3d_reconstruction.png"):
        """
        Create a solid 3D visualization using matplotlib and marching cubes.
        
        Args:
            smooth (bool): Whether to apply smoothing before visualization
            save_path (str): Path to save the visualization
        """
        if not SKIMAGE_AVAILABLE:
            print("Solid visualization requires scikit-image for marching cubes.")
            print("Falling back to voxel visualization...")
            self.visualize_3d_voxels_matplotlib(save_path=save_path)
            return
        
        # Extract manifold surface using marching cubes
        surface_result = self.extract_manifold_surface(smooth=smooth, manifold=True)
        if surface_result is None:
            print("Failed to extract manifold surface. Cannot create solid visualization.")
            return
        
        vertices, faces = surface_result
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot triangular mesh surface
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                       triangles=faces, alpha=0.8, shade=True, 
                       cmap='viridis', linewidth=0.1)
        
        # Set the axes limits
        ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
        ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
        ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('3D Solid Reconstruction from Tomography Slices')
        
        # Set equal aspect ratio
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
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"3D solid visualization saved to: {save_path}")

    def visualize_3d_voxels_matplotlib(self, smooth: bool = True, save_path: str = "3d_voxel_reconstruction.png"):
        """
        Create a 3D voxel visualization using matplotlib.
        
        Args:
            smooth (bool): Whether to apply smoothing before visualization
            save_path (str): Path to save the visualization
        """
        if self.voxel_data is None:
            self.create_voxel_data()
        
        # Use smoothed or original data
        if smooth and SKIMAGE_AVAILABLE:
            voxel_data = self.smooth_voxel_data(iterations=config.SMOOTHING_ITERATIONS)
        else:
            voxel_data = self.voxel_data
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create coordinate arrays for voxel positions
        z, y, x = np.meshgrid(
            np.arange(voxel_data.shape[0]) * self.mm_per_slice,
            np.arange(voxel_data.shape[1]) * self.mm_per_pixel_y,
            np.arange(voxel_data.shape[2]) * self.mm_per_pixel_x,
            indexing='ij'
        )
        
        # Plot voxels as 3D blocks
        ax.voxels(voxel_data, facecolors='lightblue', edgecolors='darkblue', alpha=0.7)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('3D Voxel Reconstruction from Tomography Slices')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"3D voxel visualization saved to: {save_path}")
    
    def visualize_3d_interactive_mesh(self, smooth: bool = True, save_path: str = "3d_reconstruction_interactive.html"):
        """
        Create an interactive 3D mesh visualization using plotly.
        
        Args:
            smooth (bool): Whether to apply smoothing before visualization
            save_path (str): Path to save the HTML file
        """
        if not PLOTLY_AVAILABLE:
            print("Interactive visualization requires plotly. Please install: pip install plotly")
            return
        
        if not SKIMAGE_AVAILABLE:
            print("Interactive mesh visualization requires scikit-image. Creating point cloud instead...")
            points = self.generate_point_cloud(subsample_factor=config.SUBSAMPLE_FACTOR)
            
            # Create 3D scatter plot as fallback
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
        else:
            # Extract manifold surface for mesh visualization
            surface_result = self.extract_manifold_surface(smooth=smooth, manifold=True)
            if surface_result is None:
                print("Failed to extract manifold surface for interactive visualization.")
                return
            
            vertices, faces = surface_result
            
            # Create 3D mesh plot
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
        
        fig.update_layout(
            title='Interactive 3D Solid Reconstruction from Tomography Slices',
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
        print(f"Interactive 3D mesh visualization saved to: {save_path}")
        print("Open the HTML file in a web browser to view the interactive plot.")
    
    def analyze_object_properties(self):
        """
        Analyze various properties of the reconstructed object.
        """
        if self.voxel_data is None:
            self.create_voxel_data()
        
        print("\n" + "="*60)
        print("OBJECT ANALYSIS")
        print("="*60)
        
        # Basic properties - calculate both voxel and mesh volumes
        voxel_volume = self.calculate_volume(use_processed_data=False)
        processed_voxel_volume = self.calculate_volume(use_processed_data=True) 
        mesh_volume = self.calculate_mesh_volume_from_obj()
        
        # Use mesh volume as the primary volume if available (most accurate for OBJ model)
        primary_volume = mesh_volume if mesh_volume is not None else processed_voxel_volume
        
        # Bounding box
        z_coords, y_coords, x_coords = np.where(self.voxel_data)
        
        bbox_x = (x_coords.min() * self.mm_per_pixel_x, x_coords.max() * self.mm_per_pixel_x)
        bbox_y = (y_coords.min() * self.mm_per_pixel_y, y_coords.max() * self.mm_per_pixel_y)
        bbox_z = (z_coords.min() * self.mm_per_slice, z_coords.max() * self.mm_per_slice)
        
        bbox_dimensions = (
            bbox_x[1] - bbox_x[0],
            bbox_y[1] - bbox_y[0], 
            bbox_z[1] - bbox_z[0]
        )
        
        # Surface area (calculate using manifold surface if available)
        surface_area = None
        if SKIMAGE_AVAILABLE:
            try:
                surface_result = self.extract_manifold_surface(smooth=True, manifold=True)
                if surface_result is not None:
                    vertices, faces = surface_result
                    # Calculate surface area from triangular faces
                    v0 = vertices[faces[:, 0]]
                    v1 = vertices[faces[:, 1]]
                    v2 = vertices[faces[:, 2]]
                    
                    # Cross product to get triangle areas
                    cross_product = np.cross(v1 - v0, v2 - v0)
                    triangle_areas = 0.5 * np.linalg.norm(cross_product, axis=1)
                    surface_area = np.sum(triangle_areas)
                    
            except Exception as e:
                print(f"Note: Could not calculate surface area: {e}")
        
        # Print results
        print(f"Volume Analysis:")
        print(f"  Original voxel volume: {voxel_volume:.4f} mm³")
        print(f"  Processed voxel volume: {processed_voxel_volume:.4f} mm³")
        if mesh_volume is not None:
            print(f"  Mesh volume (OBJ-accurate): {mesh_volume:.4f} mm³")
            print(f"  Primary volume (used for calculations): {primary_volume:.4f} mm³")
        else:
            print(f"  Primary volume (processed voxels): {primary_volume:.4f} mm³")
            
        print(f"Bounding Box:")
        print(f"  X: {bbox_x[0]:.2f} to {bbox_x[1]:.2f} mm (width: {bbox_dimensions[0]:.2f} mm)")
        print(f"  Y: {bbox_y[0]:.2f} to {bbox_y[1]:.2f} mm (height: {bbox_dimensions[1]:.2f} mm)")
        print(f"  Z: {bbox_z[0]:.2f} to {bbox_z[1]:.2f} mm (depth: {bbox_dimensions[2]:.2f} mm)")
        
        if surface_area:
            print(f"Surface Area: {surface_area:.4f} mm²")
            print(f"Surface Area to Volume Ratio: {surface_area/primary_volume:.4f} mm⁻¹")
        
        # Density analysis
        total_possible_volume = self.x_length_mm * self.y_length_mm * self.total_depth_mm
        density = primary_volume / total_possible_volume
        print(f"Object Density: {density:.4f} ({100*density:.2f}% of total volume)")
        
        return {
            'volume_mm3': primary_volume,
            'voxel_volume_mm3': voxel_volume,
            'processed_voxel_volume_mm3': processed_voxel_volume,
            'mesh_volume_mm3': mesh_volume,
            'bounding_box': {'x': bbox_x, 'y': bbox_y, 'z': bbox_z},
            'dimensions': bbox_dimensions,
            'surface_area_mm2': surface_area,
            'density': density
        }
    
    def export_to_obj(self, filename: str = "tomography_model.obj", smooth: bool = True) -> bool:
        """
        Export the 3D model to OBJ file format.
        
        Args:
            filename (str): Output filename for the OBJ file
            smooth (bool): Whether to apply smoothing before export
            
        Returns:
            bool: True if export successful, False otherwise
        """
        if not SKIMAGE_AVAILABLE:
            print("OBJ export requires scikit-image for surface extraction.")
            return False
        
        # Extract manifold surface mesh
        surface_result = self.extract_manifold_surface(smooth=smooth, manifold=True)
        if surface_result is None:
            print("Failed to extract manifold surface for OBJ export.")
            return False
        
        vertices, faces = surface_result
        
        try:
            with open(filename, 'w') as f:
                # Write header
                f.write("# OBJ file generated from tomography reconstruction\n")
                f.write("# Manifold solid model with closed surfaces\n")
                f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n")
                f.write(f"# Physical dimensions: {self.x_length_mm:.2f} x {self.y_length_mm:.2f} x {self.total_depth_mm:.2f} mm\n\n")
                
                # Write vertices
                for vertex in vertices:
                    f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                
                f.write("\n")
                
                # Write faces (OBJ uses 1-based indexing)
                for face in faces:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            
            print(f"Closed manifold 3D model exported to: {filename}")
            print(f"  Vertices: {len(vertices):,}")
            print(f"  Faces: {len(faces):,}")
            print(f"  Model should be watertight and ready for 3D printing")
            return True
            
        except Exception as e:
            print(f"Error exporting to OBJ: {e}")
            return False


def main():
    """Main function using configuration from config.py."""
    
    # Use configuration values from config.py
    X_LENGTH_MM = config.X_LENGTH_MM
    Y_LENGTH_MM = config.Y_LENGTH_MM
    TOTAL_DEPTH_MM = config.TOTAL_DEPTH_MM
    
    # Processing parameters from config
    THRESHOLD = config.THRESHOLD
    SUBSAMPLE_FACTOR = config.SUBSAMPLE_FACTOR
    
    # Visualization control from config
    SHOW_3D_VISUALIZATION = config.SHOW_3D_VISUALIZATION
    EXPORT_OBJ_MODEL = config.EXPORT_OBJ_MODEL
    
    print("Tomography 3D Reconstruction")
    print("="*60)
    print(f"Physical Dimensions: {X_LENGTH_MM} x {Y_LENGTH_MM} x {TOTAL_DEPTH_MM} mm")
    print(f"Mask Threshold: {THRESHOLD}")
    print(f"Show 3D Visualization: {SHOW_3D_VISUALIZATION}")
    print(f"Export OBJ Model: {EXPORT_OBJ_MODEL}")
    print("="*60)
    
    try:
        # Initialize the reconstruction system
        reconstructor = Tomography3DReconstruction(
            x_length_mm=X_LENGTH_MM,
            y_length_mm=Y_LENGTH_MM,
            total_depth_mm=TOTAL_DEPTH_MM
        )
        
        # Load mask images
        print("\n1. Loading mask images...")
        if not reconstructor.load_mask_images(directory=config.DATA_PATH, threshold=THRESHOLD):
            print("Failed to load mask images. Exiting.")
            return 1
        
        # Show slice preview if enabled
        if config.SHOW_SLICE_PREVIEW:
            print("\n1.5. Generating slice preview...")
            reconstructor.visualize_slices(
                num_slices_to_show=config.NUM_PREVIEW_SLICES,
                save_path=config.SLICE_PREVIEW_PNG
            )
        
        # Create 3D voxel data with closed ends
        print("\n2. Creating closed 3D voxel data...")
        reconstructor.create_voxel_data(close_ends=config.CLOSE_VOLUME_ENDS)
        
        # Analyze object properties
        print("\n3. Analyzing object properties...")
        properties = reconstructor.analyze_object_properties()
        
        # Generate visualizations (if enabled)
        if SHOW_3D_VISUALIZATION:
            print("\n4. Generating 3D solid reconstruction...")
            
            # 3D solid visualization
            print("   Creating smoothed 3D solid visualization...")
            reconstructor.visualize_3d_solid_matplotlib(
                smooth=config.APPLY_SMOOTHING,
                save_path=config.VISUALIZATION_PNG
            )
            
            # Interactive 3D visualization (if plotly available)
            if PLOTLY_AVAILABLE:
                print("   Creating interactive 3D mesh visualization...")
                reconstructor.visualize_3d_interactive_mesh(
                    smooth=config.APPLY_SMOOTHING,
                    save_path=config.INTERACTIVE_HTML
                )
        else:
            print("\n4. Skipping 3D visualization (SHOW_3D_VISUALIZATION = False)")
        
        # Export 3D model to OBJ file (if enabled)
        if EXPORT_OBJ_MODEL:
            print("\n5. Exporting 3D model to OBJ file...")
            success = reconstructor.export_to_obj(
                filename=config.OBJ_FILENAME,
                smooth=config.APPLY_SMOOTHING
            )
            if not success:
                print("   Failed to export OBJ file.")
        else:
            print("\n5. Skipping OBJ export (EXPORT_OBJ_MODEL = False)")
        
        print("\n" + "="*60)
        print("RECONSTRUCTION COMPLETE!")
        print("="*60)
        print("Generated files:")
        if config.SHOW_SLICE_PREVIEW:
            print(f"  - {config.SLICE_PREVIEW_PNG} (slice preview)")
        if SHOW_3D_VISUALIZATION:
            print(f"  - {config.VISUALIZATION_PNG} (3D solid model)")
            if PLOTLY_AVAILABLE:
                print(f"  - {config.INTERACTIVE_HTML} (Interactive 3D mesh)")
        if EXPORT_OBJ_MODEL:
            print(f"  - {config.OBJ_FILENAME} (3D model file)")
        if not config.SHOW_SLICE_PREVIEW and not SHOW_3D_VISUALIZATION and not EXPORT_OBJ_MODEL:
            print("  - No output files generated (all features disabled)")
        
        print(f"\nFinal Results:")
        if properties.get('mesh_volume_mm3') is not None:
            print(f"  Object Volume (OBJ-accurate): {properties['volume_mm3']:.4f} mm³")
            print(f"  Volume difference (mesh vs original): {((properties['mesh_volume_mm3'] - properties['voxel_volume_mm3']) / properties['voxel_volume_mm3'] * 100):+.1f}%")
        else:
            print(f"  Object Volume: {properties['volume_mm3']:.4f} mm³")
        print(f"  Object Dimensions: {properties['dimensions'][0]:.2f} x {properties['dimensions'][1]:.2f} x {properties['dimensions'][2]:.2f} mm")
        print(f"  Object Density: {100*properties['density']:.2f}% of total volume")
        
        return 0
        
    except Exception as e:
        print(f"Error during reconstruction: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())