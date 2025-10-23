#!/usr/bin/env python3
"""
3D reconstruction from tomography slices - Main orchestrator.
"""

import numpy as np
from typing import Optional

# Import configuration
import config

# Import modular components
from image_loader import ImageLoader
from voxel_processor import VoxelProcessor
from surface_extractor import SurfaceExtractor
from visualizer import Visualizer
from volume_calculator import VolumeCalculator
from obj_exporter import OBJExporter


class Tomography3DReconstruction:
    """Main orchestrator for 3D reconstruction from tomography slices."""
    
    def __init__(self, x_length_mm: float, y_length_mm: float, total_depth_mm: float):
        """Initialize reconstruction system."""
        self.x_length_mm = x_length_mm
        self.y_length_mm = y_length_mm
        self.total_depth_mm = total_depth_mm
        
        # Initialize modular components
        self.image_loader = ImageLoader()
        self.voxel_processor = VoxelProcessor()
        self.surface_extractor = SurfaceExtractor()
        self.visualizer = Visualizer()
        self.volume_calculator = VolumeCalculator()
        self.obj_exporter = OBJExporter()
        
        # Resolution calculations
        self.mm_per_pixel_x = None
        self.mm_per_pixel_y = None
        self.mm_per_slice = None
        
    def load_mask_images(self, directory: str = ".", threshold: int = 200) -> bool:
        """Load mask images using ImageLoader."""
        success = self.image_loader.load_mask_images(directory, threshold)
        
        if success:
            # Calculate resolution
            width, height = self.image_loader.get_image_dimensions()
            num_slices = self.image_loader.get_num_slices()
            
            self.mm_per_pixel_x = self.x_length_mm / width
            self.mm_per_pixel_y = self.y_length_mm / height
            self.mm_per_slice = self.total_depth_mm / num_slices
            
            print(f"Loaded {num_slices} images - {self.mm_per_slice:.3f} mm/slice")
        
        return success
    
    def create_voxel_data(self, close_ends: bool = True) -> np.ndarray:
        """Create 3D voxel data using VoxelProcessor."""
        mask_images = self.image_loader.get_mask_images()
        return self.voxel_processor.create_voxel_data(mask_images, close_ends)
    
    def calculate_volume(self, use_processed_data: bool = False) -> float:
        """Calculate volume in mm³."""
        if self.voxel_processor.voxel_data is None:
            self.create_voxel_data(close_ends=True)
        
        if use_processed_data:
            volume_data = self.voxel_processor.smooth_voxel_data(
                self.voxel_processor.voxel_data, 
                iterations=config.SMOOTHING_ITERATIONS, 
                create_manifold=config.CREATE_MANIFOLD
            )
        else:
            volume_data = self.voxel_processor.voxel_data
        
        return self.volume_calculator.calculate_voxel_volume(
            volume_data, self.mm_per_pixel_x, self.mm_per_pixel_y, self.mm_per_slice
        )
    
    def calculate_mesh_volume_from_obj(self) -> Optional[float]:
        """Calculate mesh volume using SurfaceExtractor."""
        if self.voxel_processor.voxel_data is None:
            return None
            
        volume_data = self.voxel_processor.smooth_voxel_data(
            self.voxel_processor.voxel_data,
            iterations=config.SMOOTHING_ITERATIONS,
            create_manifold=config.CREATE_MANIFOLD
        )
        
        surface_result = self.surface_extractor.extract_manifold_surface(
            volume_data, self.mm_per_slice, self.mm_per_pixel_y, self.mm_per_pixel_x,
            smooth=True, manifold=True, add_padding=config.ADD_VOLUME_PADDING
        )
        
        if surface_result is None:
            return None
        
        vertices, faces = surface_result
        return self.surface_extractor.calculate_mesh_volume(vertices, faces)
    
    def visualize_3d_solid_matplotlib(self, smooth: bool = True):
        """Create 3D solid visualization using Visualizer."""
        if self.voxel_processor.voxel_data is None:
            return
            
        volume_data = self.voxel_processor.voxel_data
        if smooth:
            volume_data = self.voxel_processor.smooth_voxel_data(
                volume_data, iterations=config.SMOOTHING_ITERATIONS, create_manifold=config.CREATE_MANIFOLD
            )
        
        surface_result = self.surface_extractor.extract_manifold_surface(
            volume_data, self.mm_per_slice, self.mm_per_pixel_y, self.mm_per_pixel_x,
            smooth=smooth, manifold=True, add_padding=config.ADD_VOLUME_PADDING
        )
        
        if surface_result is None:
            # Fallback to voxel visualization
            self.visualizer.visualize_3d_voxels_matplotlib(volume_data)
            return
        
        vertices, faces = surface_result
        self.visualizer.visualize_3d_solid_matplotlib(vertices, faces)
    
    def visualize_3d_interactive_mesh(self, smooth: bool = True, save_path: str = "3d_reconstruction_interactive.html"):
        """Create interactive 3D visualization using Visualizer."""
        if self.voxel_processor.voxel_data is None:
            return
            
        volume_data = self.voxel_processor.voxel_data
        if smooth:
            volume_data = self.voxel_processor.smooth_voxel_data(
                volume_data, iterations=config.SMOOTHING_ITERATIONS, create_manifold=config.CREATE_MANIFOLD
            )
        
        surface_result = self.surface_extractor.extract_manifold_surface(
            volume_data, self.mm_per_slice, self.mm_per_pixel_y, self.mm_per_pixel_x,
            smooth=smooth, manifold=True, add_padding=config.ADD_VOLUME_PADDING
        )
        
        if surface_result is None:
            # Fallback to point cloud
            points = self.voxel_processor.generate_point_cloud(
                volume_data, self.mm_per_pixel_x, self.mm_per_pixel_y, 
                self.mm_per_slice, config.SUBSAMPLE_FACTOR
            )
            self.visualizer.visualize_3d_interactive_mesh(points=points, save_path=save_path)
            return
        
        vertices, faces = surface_result
        self.visualizer.visualize_3d_interactive_mesh(vertices=vertices, faces=faces, save_path=save_path)
    
    def analyze_object_properties(self):
        """Analyze object properties using VolumeCalculator."""
        if self.voxel_processor.voxel_data is None:
            self.create_voxel_data()
        
        print("\nAnalyzing object properties...")
        
        voxel_volume = self.calculate_volume(use_processed_data=False)
        processed_voxel_volume = self.calculate_volume(use_processed_data=True) 
        mesh_volume = self.calculate_mesh_volume_from_obj()
        
        # Calculate surface area
        surface_area = None
        volume_data = self.voxel_processor.smooth_voxel_data(
            self.voxel_processor.voxel_data,
            iterations=config.SMOOTHING_ITERATIONS,
            create_manifold=config.CREATE_MANIFOLD
        )
        
        surface_result = self.surface_extractor.extract_manifold_surface(
            volume_data, self.mm_per_slice, self.mm_per_pixel_y, self.mm_per_pixel_x,
            smooth=True, manifold=True, add_padding=config.ADD_VOLUME_PADDING
        )
        
        if surface_result is not None:
            vertices, faces = surface_result
            try:
                surface_area = self.surface_extractor.calculate_surface_area(vertices, faces)
            except Exception:
                pass
        
        return self.volume_calculator.analyze_object_properties(
            self.voxel_processor.voxel_data, processed_voxel_volume, mesh_volume, surface_area,
            self.mm_per_pixel_x, self.mm_per_pixel_y, self.mm_per_slice,
            self.x_length_mm, self.y_length_mm, self.total_depth_mm
        )
    
    def export_to_obj(self, filename: str = "tomography_model.obj", smooth: bool = True) -> bool:
        """Export 3D model to OBJ format using OBJExporter."""
        if self.voxel_processor.voxel_data is None:
            return False
            
        volume_data = self.voxel_processor.voxel_data
        if smooth:
            volume_data = self.voxel_processor.smooth_voxel_data(
                volume_data, iterations=config.SMOOTHING_ITERATIONS, create_manifold=config.CREATE_MANIFOLD
            )
        
        surface_result = self.surface_extractor.extract_manifold_surface(
            volume_data, self.mm_per_slice, self.mm_per_pixel_y, self.mm_per_pixel_x,
            smooth=smooth, manifold=True, add_padding=config.ADD_VOLUME_PADDING
        )
        
        if surface_result is None:
            return False
        
        vertices, faces = surface_result
        return self.obj_exporter.export_to_obj(vertices, faces, filename)


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
        
        print("Creating voxel data...")
        reconstructor.create_voxel_data(close_ends=config.CLOSE_VOLUME_ENDS)
        
        properties = reconstructor.analyze_object_properties()
        
        if config.SHOW_3D_VISUALIZATION:
            print("Creating visualizations...")
            
            reconstructor.visualize_3d_solid_matplotlib(
                smooth=config.APPLY_SMOOTHING
            )
            
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