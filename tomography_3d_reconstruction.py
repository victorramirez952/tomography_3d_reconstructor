#!/usr/bin/env python3
"""
3D reconstruction from tomography slices.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from typing import Tuple, Optional
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
    except ImportError:
        SKIMAGE_AVAILABLE = False
        SCIPY_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False




class Tomography3DReconstruction:
    def __init__(self, x_length_mm: float, y_length_mm: float, total_depth_mm: float):
        """Initialize reconstruction system."""
        self.x_length_mm = x_length_mm
        self.y_length_mm = y_length_mm
        self.total_depth_mm = total_depth_mm
        
        self.mask_files = []
        self.mask_images = []
        self.voxel_data = None
        
        self.image_width = None
        self.image_height = None
        self.num_slices = 0
        
        self.mm_per_pixel_x = None
        self.mm_per_pixel_y = None
        self.mm_per_slice = None
        
    def load_mask_images(self, directory: str = ".", threshold: int = 200) -> bool:
        """Load mask images from directory."""
        try:
            mask_pattern = os.path.join(directory, "Mask_*.png")
            self.mask_files = sorted(glob.glob(mask_pattern))
            
            if not self.mask_files:
                return False
                
            print(f"Found {len(self.mask_files)} masks")
            
            self.mask_images = []
            first_image = None
            
            for file_path in self.mask_files:
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                if first_image is None:
                    first_image = img
                    self.image_height, self.image_width = img.shape
                elif img.shape != first_image.shape:
                    continue
                
                binary_mask = img >= threshold
                self.mask_images.append(binary_mask)
            
            self.num_slices = len(self.mask_images)
            
            if self.num_slices == 0:
                return False
            
            self.mm_per_pixel_x = self.x_length_mm / self.image_width
            self.mm_per_pixel_y = self.y_length_mm / self.image_height
            self.mm_per_slice = self.total_depth_mm / self.num_slices
            
            print(f"Loaded {self.num_slices} images - {self.mm_per_slice:.3f} mm/slice")
            return True
            
        except Exception as e:
            print(f"Loading failed: {e}")
            return False
    
    def create_voxel_data(self, close_ends: bool = True) -> np.ndarray:
        """Create 3D voxel data from masks."""
        if not self.mask_images:
            raise ValueError("Load masks first, hmm.")
        
        self.voxel_data = np.stack(self.mask_images, axis=0)
        
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
    
    def calculate_volume(self, use_processed_data: bool = False) -> float:
        """Calculate volume in mm³."""
        if self.voxel_data is None:
            self.create_voxel_data(close_ends=True)
        
        if use_processed_data:
            volume_data = self.smooth_voxel_data(iterations=config.SMOOTHING_ITERATIONS, create_manifold=config.CREATE_MANIFOLD)
        else:
            volume_data = self.voxel_data
        
        voxel_volume = self.mm_per_pixel_x * self.mm_per_pixel_y * self.mm_per_slice
        total_volume = np.sum(volume_data) * voxel_volume
        
        return total_volume
    
    def calculate_mesh_volume_from_obj(self) -> Optional[float]:
        """Calculate mesh volume using divergence theorem."""
        surface_result = self.extract_manifold_surface(smooth=True, manifold=True)
        if surface_result is None:
            return None
        
        vertices, faces = surface_result
        
        volume = 0.0
        for face in faces:
            v0 = vertices[face[0]]
            v1 = vertices[face[1]] 
            v2 = vertices[face[2]]
            
            tetrahedron_volume = np.dot(v0, np.cross(v1, v2)) / 6.0
            volume += tetrahedron_volume
        
        return abs(volume)
    
    def smooth_voxel_data(self, iterations: int = 3, create_manifold: bool = True) -> np.ndarray:
        """Smooth voxel data with morphological operations."""
        if not SKIMAGE_AVAILABLE:
            return self.voxel_data
        
        if self.voxel_data is None:
            self.create_voxel_data()
        
        smoothed_data = self.voxel_data.copy()
        
        try:
            if create_manifold:
                smoothed_data = binary_opening(smoothed_data)
            
            for i in range(iterations):
                smoothed_data = binary_closing(smoothed_data)
            
            if create_manifold:
                try:
                    struct_elem = ball(2)
                    smoothed_data = binary_closing(smoothed_data, structure=struct_elem)
                except:
                    smoothed_data = binary_closing(smoothed_data)
                
                if SCIPY_AVAILABLE:
                    for z in range(smoothed_data.shape[0]):
                        smoothed_data[z] = ndimage.binary_fill_holes(smoothed_data[z])
            
        except Exception as e:
            for i in range(iterations):
                smoothed_data = binary_closing(smoothed_data)
        
        return smoothed_data

    def extract_manifold_surface(self, smooth: bool = True, manifold: bool = True) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Extract surface using marching cubes."""
        if not SKIMAGE_AVAILABLE:
            return None
        
        if self.voxel_data is None:
            self.create_voxel_data(close_ends=True)
        
        try:
            if smooth:
                volume_data = self.smooth_voxel_data(iterations=config.SMOOTHING_ITERATIONS, create_manifold=manifold and config.CREATE_MANIFOLD)
            else:
                volume_data = self.voxel_data
            
            if manifold and config.ADD_VOLUME_PADDING:
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
            
            vertices[:, 0] *= self.mm_per_slice
            vertices[:, 1] *= self.mm_per_pixel_y
            vertices[:, 2] *= self.mm_per_pixel_x
            
            if manifold:
                vertices, faces = self._ensure_manifold_mesh(vertices, faces)
            
            print(f"Surface: {len(vertices)} vertices, {len(faces)} faces")
            
            return vertices, faces
            
        except Exception as e:
            return None
    
    def _add_volume_padding(self, volume_data: np.ndarray, pad_size: int = 1) -> np.ndarray:
        """Add padding around volume."""
        padded_volume = np.pad(volume_data, pad_size, mode='constant', constant_values=False)
        return padded_volume
    
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


    
    def generate_point_cloud(self, subsample_factor: int = 1) -> np.ndarray:
        """Generate point cloud from voxels."""
        if self.voxel_data is None:
            self.create_voxel_data()
        
        z_coords, y_coords, x_coords = np.where(self.voxel_data)
        
        if subsample_factor > 1:
            indices = np.arange(0, len(z_coords), subsample_factor)
            z_coords = z_coords[indices]
            y_coords = y_coords[indices]
            x_coords = x_coords[indices]
        
        points_mm = np.column_stack([
            x_coords * self.mm_per_pixel_x,
            y_coords * self.mm_per_pixel_y,
            z_coords * self.mm_per_slice
        ])
        
        return points_mm
    
    def visualize_slices(self, num_slices_to_show: int = 9, save_path: str = "slice_visualization.png"):
        """Visualize 2D slices."""
        if not self.mask_images:
            return
        
        slice_indices = np.linspace(0, len(self.mask_images)-1, 
                                  min(num_slices_to_show, len(self.mask_images)), 
                                  dtype=int)
        
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
            ax.set_title(f'Slice {slice_idx+1}\nZ={slice_idx * self.mm_per_slice:.2f}mm')
            ax.axis('off')
        
        for i in range(len(slice_indices), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_3d_solid_matplotlib(self, smooth: bool = True, save_path: str = "3d_reconstruction.png"):
        """Create 3D solid visualization."""
        if not SKIMAGE_AVAILABLE:
            self.visualize_3d_voxels_matplotlib(save_path=save_path)
            return
        
        surface_result = self.extract_manifold_surface(smooth=smooth, manifold=True)
        if surface_result is None:
            return
        
        vertices, faces = surface_result
        
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
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def visualize_3d_voxels_matplotlib(self, smooth: bool = True, save_path: str = "3d_voxel_reconstruction.png"):
        """Create voxel visualization."""
        if self.voxel_data is None:
            self.create_voxel_data()
        
        if smooth and SKIMAGE_AVAILABLE:
            voxel_data = self.smooth_voxel_data(iterations=config.SMOOTHING_ITERATIONS)
        else:
            voxel_data = self.voxel_data
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.voxels(voxel_data, facecolors='lightblue', edgecolors='darkblue', alpha=0.7)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('3D Voxel Reconstruction')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_3d_interactive_mesh(self, smooth: bool = True, save_path: str = "3d_reconstruction_interactive.html"):
        """Create interactive 3D visualization."""
        if not PLOTLY_AVAILABLE:
            return
        
        if not SKIMAGE_AVAILABLE:
            points = self.generate_point_cloud(subsample_factor=config.SUBSAMPLE_FACTOR)
            
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
            surface_result = self.extract_manifold_surface(smooth=smooth, manifold=True)
            if surface_result is None:
                return
            
            vertices, faces = surface_result
            
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
    
    def analyze_object_properties(self):
        """Analyze object properties."""
        if self.voxel_data is None:
            self.create_voxel_data()
        
        print("\nAnalyzing object properties...")
        
        voxel_volume = self.calculate_volume(use_processed_data=False)
        processed_voxel_volume = self.calculate_volume(use_processed_data=True) 
        mesh_volume = self.calculate_mesh_volume_from_obj()
        
        primary_volume = mesh_volume if mesh_volume is not None else processed_voxel_volume
        
        z_coords, y_coords, x_coords = np.where(self.voxel_data)
        
        bbox_x = (x_coords.min() * self.mm_per_pixel_x, x_coords.max() * self.mm_per_pixel_x)
        bbox_y = (y_coords.min() * self.mm_per_pixel_y, y_coords.max() * self.mm_per_pixel_y)
        bbox_z = (z_coords.min() * self.mm_per_slice, z_coords.max() * self.mm_per_slice)
        
        bbox_dimensions = (
            bbox_x[1] - bbox_x[0],
            bbox_y[1] - bbox_y[0], 
            bbox_z[1] - bbox_z[0]
        )
        
        surface_area = None
        if SKIMAGE_AVAILABLE:
            try:
                surface_result = self.extract_manifold_surface(smooth=True, manifold=True)
                if surface_result is not None:
                    vertices, faces = surface_result
                    v0 = vertices[faces[:, 0]]
                    v1 = vertices[faces[:, 1]]
                    v2 = vertices[faces[:, 2]]
                    
                    cross_product = np.cross(v1 - v0, v2 - v0)
                    triangle_areas = 0.5 * np.linalg.norm(cross_product, axis=1)
                    surface_area = np.sum(triangle_areas)
                    
            except Exception:
                pass
        
        print(f"Volume: {primary_volume:.4f} mm³")
        print(f"Dimensions: {bbox_dimensions[0]:.2f} x {bbox_dimensions[1]:.2f} x {bbox_dimensions[2]:.2f} mm")
        if surface_area:
            print(f"Surface Area: {surface_area:.4f} mm²")
        
        density = primary_volume / (self.x_length_mm * self.y_length_mm * self.total_depth_mm)
        print(f"Density: {100*density:.1f}% of total space")
        
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
        """Export 3D model to OBJ format."""
        if not SKIMAGE_AVAILABLE:
            return False
        
        surface_result = self.extract_manifold_surface(smooth=smooth, manifold=True)
        if surface_result is None:
            return False
        
        vertices, faces = surface_result
        
        try:
            with open(filename, 'w') as f:
                f.write("# Tomography reconstruction model\n")
                f.write(f"# {len(vertices)} vertices, {len(faces)} faces\n\n")
                
                for vertex in vertices:
                    f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                
                f.write("\n")
                
                for face in faces:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            
            print(f"Model exported: {filename}")
            return True
            
        except Exception as e:
            print(f"Export failed: {e}")
            return False


def main():
    """Main function."""
    print("Tomography 3D Reconstruction")
    
    try:
        reconstructor = Tomography3DReconstruction(
            x_length_mm=config.X_LENGTH_MM,
            y_length_mm=config.Y_LENGTH_MM,
            total_depth_mm=config.TOTAL_DEPTH_MM
        )
        
        print("Loading masks...")
        if not reconstructor.load_mask_images(directory=config.DATA_PATH, threshold=config.THRESHOLD):
            print("Failed to load masks")
            return 1
        
        if config.SHOW_SLICE_PREVIEW:
            reconstructor.visualize_slices(
                num_slices_to_show=config.NUM_PREVIEW_SLICES,
                save_path=config.SLICE_PREVIEW_PNG
            )
        
        print("Creating voxel data...")
        reconstructor.create_voxel_data(close_ends=config.CLOSE_VOLUME_ENDS)
        
        properties = reconstructor.analyze_object_properties()
        
        if config.SHOW_3D_VISUALIZATION:
            print("Creating visualizations...")
            
            reconstructor.visualize_3d_solid_matplotlib(
                smooth=config.APPLY_SMOOTHING,
                save_path=config.VISUALIZATION_PNG
            )
            
            if PLOTLY_AVAILABLE:
                reconstructor.visualize_3d_interactive_mesh(
                    smooth=config.APPLY_SMOOTHING,
                    save_path=config.INTERACTIVE_HTML
                )
        
        if config.EXPORT_OBJ_MODEL:
            print("Exporting OBJ...")
            success = reconstructor.export_to_obj(
                filename=config.OBJ_FILENAME,
                smooth=config.APPLY_SMOOTHING
            )
        
        print("\nReconstruction complete!")
        print(f"Volume: {properties['volume_mm3']:.4f} mm³")
        print(f"Dimensions: {properties['dimensions'][0]:.2f} x {properties['dimensions'][1]:.2f} x {properties['dimensions'][2]:.2f} mm")
        print(f"Density: {100*properties['density']:.1f}%")
        
        return 0
        
    except Exception as e:
        print(f"Reconstruction failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())