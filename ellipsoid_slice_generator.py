import numpy as np
import cv2
import os
from typing import Tuple, List
import matplotlib.pyplot as plt
from scipy import ndimage

class EllipsoidSliceGenerator:
    def __init__(self, image_path: str):
        """Initialize ellipsoid slice generator with middle slice image."""
        self.image_path = image_path
        self.middle_slice = self._load_and_preprocess_image()
        self.ellipse_params = self._extract_ellipse_parameters()
        
    def _load_and_preprocess_image(self) -> np.ndarray:
        """Load and preprocess image to binary format."""
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image from {self.image_path}")
        
        _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return binary_img
    
    def _extract_ellipse_parameters(self) -> dict:
        """Extract ellipse parameters from middle slice."""
        contours, _ = cv2.findContours(self.middle_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("No contours found in the image")
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            center, axes, angle = ellipse
            
            a = max(axes) / 2
            b = min(axes) / 2
            
            return {
                'center': center,
                'semi_major_axis': a,
                'semi_minor_axis': b,
                'angle': angle,
                'area': cv2.contourArea(largest_contour)
            }
        else:
            raise ValueError("Could not fit ellipse to the contour")
    
    def _calculate_ellipse_area_at_height(self, z: float, c: float) -> float:
        """Calculate ellipse cross-section area at height z using ellipsoid formula."""
        if abs(z) > c:
            return 0.0
        
        factor = np.sqrt(1 - (z / c) ** 2)
        a_z = self.ellipse_params['semi_major_axis'] * factor
        b_z = self.ellipse_params['semi_minor_axis'] * factor
        
        return np.pi * a_z * b_z
    
    def _generate_slice_at_height(self, z: float, c: float) -> np.ndarray:
        """Generate slice at height z (z=0: original mask, z=c: smallest slice)."""
        if z < 0 or z > c:
            return np.zeros_like(self.middle_slice)
        
        factor = np.sqrt(1 - (z / c) ** 2) if c > 0 else 0
        
        if factor <= 0:
            return np.zeros_like(self.middle_slice)
        
        center = self.ellipse_params['center']
        height, width = self.middle_slice.shape
        
        M = cv2.getRotationMatrix2D(center, 0, factor)
        scaled_slice = cv2.warpAffine(self.middle_slice, M, (width, height))
        
        return scaled_slice
    
    def generate_slices(self, num_slices: int, output_dir: str = "slices") -> List[str]:
        """Generate n slices sorted by area (smallest to largest)."""
        os.makedirs(output_dir, exist_ok=True)
        
        c = min(self.ellipse_params['semi_major_axis'], self.ellipse_params['semi_minor_axis'])
        z_positions = np.linspace(-c, c, num_slices)
        middle_index = len(z_positions) // 2
        
        slice_data = []
        
        for i, z in enumerate(z_positions):
            slice_img = self._generate_slice_at_height(z, c)
            area = np.sum(slice_img > 0)
            slice_data.append((i, z, slice_img, area))
        
        slice_data.sort(key=lambda x: x[3])
        
        saved_files = []
        
        for mask_number, (original_index, z, slice_img, area) in enumerate(slice_data, 1):
            filename = f"Mask_{mask_number:03d}.png"
            filepath = os.path.join(output_dir, filename)
            
            cv2.imwrite(filepath, slice_img)
            saved_files.append(filepath)
        
        return saved_files
    
    def generate_slices_half_ellipsoid(self, num_slices: int, output_dir: str = "slices", 
                                      num_start: int = 28, increase: bool = True) -> List[str]:
        """Generate half-ellipsoid slices with sequential naming (original mask as base)."""
        c = min(self.ellipse_params['semi_major_axis'], self.ellipse_params['semi_minor_axis'])
        z_positions = np.linspace(0, c, num_slices + 2)
        
        if increase:
            num_end = num_start + 1 + num_slices
        else:
            num_end = num_start - num_slices - 1
            aux = num_start
            num_start = num_end
            num_end = aux
        
        saved_files = []
        number_range = list(range(num_start, num_end + 1))
        
        for i, number in enumerate(number_range):
            z_index = i if increase else len(number_range) - 1 - i
            
            if z_index < len(z_positions):
                z = z_positions[z_index]
            else:
                z = c
            
            slice_img = self._generate_slice_at_height(z, c)
            
            filename = f"Mask_Patient_{number}.png"
            filepath = os.path.join(output_dir, filename)
            
            cv2.imwrite(filepath, slice_img)
            saved_files.append(filepath)
        
        # Delete the extreme masks (first and last one)
        os.remove(saved_files[0])
        os.remove(saved_files[-1])
        return saved_files
    
    def visualize_slices(self, slice_files: List[str], max_display: int = 10):
        """Visualize subset of generated slices."""
        num_display = min(len(slice_files), max_display)
        step = len(slice_files) // num_display if num_display > 1 else 1
        
        fig, axes = plt.subplots(2, (num_display + 1) // 2, figsize=(15, 6))
        if num_display == 1:
            axes = [axes]
        axes = axes.flatten()
        
        for i in range(num_display):
            idx = i * step
            if idx < len(slice_files):
                img = cv2.imread(slice_files[idx], cv2.IMREAD_GRAYSCALE)
                axes[i].imshow(img, cmap='gray')
                axes[i].set_title(f"{os.path.basename(slice_files[idx])}")
                axes[i].axis('off')
        
        for i in range(num_display, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    """Demonstrate ellipsoid slice generator."""
    image_path = "Temporal.png"
    
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found. Please provide a valid image path.")
        image_path = input("Enter the path to your middle slice image: ")
    
    try:
        generator = EllipsoidSliceGenerator(image_path)
        
        params = generator.ellipse_params
        print("Detected ellipse parameters:")
        print(f"  Center: ({params['center'][0]:.1f}, {params['center'][1]:.1f})")
        print(f"  Semi-major axis: {params['semi_major_axis']:.1f}")
        print(f"  Semi-minor axis: {params['semi_minor_axis']:.1f}")
        print(f"  Angle: {params['angle']:.1f}°")
        print(f"  Area: {params['area']:.0f} pixels")
        
        num_slices = int(input("\nEnter the number of slices to generate (default 20): ") or "20")
        
        print(f"\nGenerating {num_slices} slices...")
        slice_files = generator.generate_slices(num_slices)
        
        print(f"\nSuccessfully generated {len(slice_files)} slices in 'slices' directory")
        print("Slices are named from Mask_001 (smallest area) to Mask_XXX (largest area)")
        
        visualize = input("\nWould you like to visualize some slices? (y/n): ").lower().startswith('y')
        if visualize:
            generator.visualize_slices(slice_files)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()