import numpy as np
import cv2
import os
from typing import Tuple, List
import matplotlib.pyplot as plt
from scipy import ndimage

class EllipsoidSliceGenerator:
    def __init__(self, image_path: str):
        """
        Initialize the ellipsoid slice generator with the middle slice image.
        
        Args:
            image_path (str): Path to the middle slice image
        """
        self.image_path = image_path
        self.middle_slice = self._load_and_preprocess_image()
        self.ellipse_params = self._extract_ellipse_parameters()
        
    def _load_and_preprocess_image(self) -> np.ndarray:
        """Load and preprocess the image to binary format."""
        # Read image
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image from {self.image_path}")
        
        # Convert to binary (assuming white oval on black background)
        _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        return binary_img
    
    def _extract_ellipse_parameters(self) -> dict:
        """Extract ellipse parameters from the middle slice."""
        # Find contours
        contours, _ = cv2.findContours(self.middle_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("No contours found in the image")
        
        # Get the largest contour (assuming it's the ellipse)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fit ellipse to the largest contour
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            center, axes, angle = ellipse
            
            # Get semi-major and semi-minor axes
            a = max(axes) / 2  # semi-major axis
            b = min(axes) / 2  # semi-minor axis
            
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
        """
        Calculate the area of an ellipse cross-section at height z.
        
        Args:
            z (float): Height from center (normalized from -1 to 1)
            c (float): Semi-axis in z direction
            
        Returns:
            float: Area of the ellipse at height z
        """
        if abs(z) > c:
            return 0.0
        
        # For an ellipsoid with semi-axes a, b, c, the cross-section at height z
        # has semi-axes a*sqrt(1 - (z/c)^2) and b*sqrt(1 - (z/c)^2)
        factor = np.sqrt(1 - (z / c) ** 2)
        a_z = self.ellipse_params['semi_major_axis'] * factor
        b_z = self.ellipse_params['semi_minor_axis'] * factor
        
        return np.pi * a_z * b_z
    
    def _generate_slice_at_height(self, z: float, c: float) -> np.ndarray:
        """
        Generate a slice at height z.
        For half-ellipsoid: z=0 gives original mask, z=c gives smallest slice
        
        Args:
            z (float): Height from base (0 to c for half-ellipsoid)
            c (float): Semi-axis in z direction
            
        Returns:
            np.ndarray: Generated slice image
        """
        if z < 0 or z > c:
            # Return empty slice (all black)
            return np.zeros_like(self.middle_slice)
        
        # For half-ellipsoid: at z=0 factor=1 (full size), at z=c factor=0 (empty)
        # Calculate scaling factor for this height
        factor = np.sqrt(1 - (z / c) ** 2) if c > 0 else 0
        
        if factor <= 0:
            return np.zeros_like(self.middle_slice)
        
        # Scale the middle slice
        center = self.ellipse_params['center']
        height, width = self.middle_slice.shape
        
        # Create transformation matrix for scaling
        M = cv2.getRotationMatrix2D(center, 0, factor)
        
        # Apply scaling transformation
        scaled_slice = cv2.warpAffine(self.middle_slice, M, (width, height))
        
        return scaled_slice
    
    def generate_slices(self, num_slices: int, output_dir: str = "slices") -> List[str]:
        """
        Generate n slices of the ellipsoid.
        
        Args:
            num_slices (int): Number of slices to generate
            output_dir (str): Output directory for the slices
            
        Returns:
            List[str]: List of generated slice file paths
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Estimate the semi-axis in z direction based on the aspect ratio
        # Assuming the ellipsoid is reasonably proportioned
        c = min(self.ellipse_params['semi_major_axis'], self.ellipse_params['semi_minor_axis'])
        
        # Calculate z positions for slices
        z_positions = np.linspace(-c, c, num_slices)
        
        # Find the middle slice index
        middle_index = len(z_positions) // 2
        
        # Generate slices and calculate their areas for sorting
        slice_data = []
        
        for i, z in enumerate(z_positions):
            slice_img = self._generate_slice_at_height(z, c)
            area = np.sum(slice_img > 0)  # Count white pixels
            slice_data.append((i, z, slice_img, area))
        
        # Sort slices by area (smallest to largest)
        slice_data.sort(key=lambda x: x[3])
        
        # Save slices with naming convention Mask_{number}
        saved_files = []
        
        for mask_number, (original_index, z, slice_img, area) in enumerate(slice_data, 1):
            filename = f"Mask_{mask_number:03d}.png"
            filepath = os.path.join(output_dir, filename)
            
            cv2.imwrite(filepath, slice_img)
            saved_files.append(filepath)
            
            # Print info about the slice
            is_middle = original_index == middle_index
            print(f"Generated {filename}: Area={area:.0f} pixels, Z={z:.2f}" + 
                  (" [MIDDLE SLICE]" if is_middle else ""))
        
        return saved_files
    
    def generate_slices_half_ellipsoid(self, num_slices: int, output_dir: str = "slices", 
                                      num_start: int = 28, increase: bool = True) -> List[str]:
        """
        Generate n slices of a half-ellipsoid with sequential naming following Pseudocode.md requirements.
        The original mask will be the base of the half-ellipsoid.
        
        Args:
            num_slices (int): Number of slices to generate
            output_dir (str): Output directory for the slices
            num_start (int): Starting number for slice naming
            increase (bool): Direction of numbering (True=ascending, False=descending then swap)
            
        Returns:
            List[str]: List of generated slice file paths
        """
        # Don't create directory here - it's handled in simple_generator.py
        
        # Estimate the semi-axis in z direction based on the aspect ratio
        c = min(self.ellipse_params['semi_major_axis'], self.ellipse_params['semi_minor_axis'])
        
        # For half-ellipsoid: z goes from 0 (base - original mask) to c (top of ellipsoid)
        z_positions = np.linspace(0, c, num_slices + 2)  # +2 to match the range size
        
        # Following updated pseudocode numbering logic
        if increase:
            num_end = num_start + 1 + num_slices
        else:
            num_end = num_start - num_slices - 1
            # Swap start and end as per pseudocode
            aux = num_start
            num_start = num_end
            num_end = aux
        
        print(f"Numbering logic: increase={increase}")
        print(f"Final range: num_start={num_start}, num_end={num_end}")
        print(f"Total numbers in range: {num_end - num_start + 1}")
        
        # Generate slices with sequential naming
        saved_files = []
        
        # Create the exact range from num_start to num_end
        number_range = list(range(num_start, num_end + 1))
        
        for i, number in enumerate(number_range):
            # Determine z position based on positioning requirements
            if increase:
                # If increase = true, original mask (z=0) must be at num_start position
                z_index = i
            else:
                # If increase = false, original mask (z=0) must be at num_end position
                z_index = len(number_range) - 1 - i
            
            # Ensure we don't exceed z_positions array
            if z_index < len(z_positions):
                z = z_positions[z_index]
            else:
                z = c  # Use maximum z for any overflow
            
            slice_img = self._generate_slice_at_height(z, c)
            
            # Following pseudocode naming: "Mask_Ana_${number}.png"
            filename = f"Mask_Ana_{number}.png"
            filepath = os.path.join(output_dir, filename)
            
            cv2.imwrite(filepath, slice_img)
            saved_files.append(filepath)
            
            # Print info about the slice
            area = np.sum(slice_img > 0)  # Count white pixels
            is_original = (z == 0 or z_index == 0)  # Original mask at z=0
            print(f"Generated {filename}: Area={area:.0f} pixels, Z={z:.2f}" + 
                  (" [ORIGINAL MASK]" if is_original else ""))
        
        return saved_files
    
    def visualize_slices(self, slice_files: List[str], max_display: int = 10):
        """
        Visualize a subset of generated slices.
        
        Args:
            slice_files (List[str]): List of slice file paths
            max_display (int): Maximum number of slices to display
        """
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
        
        # Hide unused subplots
        for i in range(num_display, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to demonstrate the ellipsoid slice generator."""
    # Default image path (can be changed)
    image_path = "Temporal.png"
    
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found. Please provide a valid image path.")
        image_path = input("Enter the path to your middle slice image: ")
    
    try:
        # Create generator
        generator = EllipsoidSliceGenerator(image_path)
        
        # Print ellipse parameters
        params = generator.ellipse_params
        print("Detected ellipse parameters:")
        print(f"  Center: ({params['center'][0]:.1f}, {params['center'][1]:.1f})")
        print(f"  Semi-major axis: {params['semi_major_axis']:.1f}")
        print(f"  Semi-minor axis: {params['semi_minor_axis']:.1f}")
        print(f"  Angle: {params['angle']:.1f}°")
        print(f"  Area: {params['area']:.0f} pixels")
        
        # Get number of slices from user
        num_slices = int(input("\nEnter the number of slices to generate (default 20): ") or "20")
        
        # Generate slices
        print(f"\nGenerating {num_slices} slices...")
        slice_files = generator.generate_slices(num_slices)
        
        print(f"\nSuccessfully generated {len(slice_files)} slices in 'slices' directory")
        print("Slices are named from Mask_001 (smallest area) to Mask_XXX (largest area)")
        
        # Ask if user wants to visualize
        visualize = input("\nWould you like to visualize some slices? (y/n): ").lower().startswith('y')
        if visualize:
            generator.visualize_slices(slice_files)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()