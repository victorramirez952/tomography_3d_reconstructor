import cv2
import numpy as np
import os
from scipy import ndimage


class MaskInterpolator:
    """Generates interpolated binary masks between two input masks using SDM interpolation."""
    
    def __init__(self, image_path_1, image_path_2, output_directory, n_interpolated=5):
        self.image_path_1 = image_path_1
        self.image_path_2 = image_path_2
        self.output_directory = output_directory
        self.n_interpolated = n_interpolated
        
        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)
    
    def load_binary_mask(self, image_path):
        """Load and convert image to binary mask."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to binary (threshold at 127)
        _, binary_mask = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        return binary_mask.astype(np.uint8)
    
    def compute_signed_distance_map(self, binary_mask):
        """Compute signed distance map from binary mask."""
        # Normalize to 0-1
        mask_normalized = binary_mask / 255.0
        
        # Distance transform for inside (positive distances)
        inside_distance = ndimage.distance_transform_edt(mask_normalized)
        
        # Distance transform for outside (negative distances)
        outside_distance = ndimage.distance_transform_edt(1 - mask_normalized)
        
        # Combine: positive inside, negative outside
        sdm = inside_distance - outside_distance
        
        return sdm
    
    def interpolate_sdms(self, sdm_1, sdm_2):
        """Linear interpolation between two SDMs."""
        interpolated_sdms = []
        
        for i in range(self.n_interpolated):
            # Linear interpolation factor
            alpha = (i + 1) / (self.n_interpolated + 1)
            
            # Interpolate between SDMs
            interpolated_sdm = (1 - alpha) * sdm_1 + alpha * sdm_2
            interpolated_sdms.append(interpolated_sdm)
        
        return interpolated_sdms
    
    def sdm_to_binary_mask(self, sdm):
        """Convert SDM back to binary mask."""
        # Pixels with positive SDM values are inside the object
        binary_mask = (sdm > 0).astype(np.uint8) * 255
        return binary_mask
    
    def save_mask(self, mask, filename):
        """Save binary mask to file."""
        filepath = os.path.join(self.output_directory, filename)
        cv2.imwrite(filepath, mask)
    
    def generate_interpolated_masks(self):
        """Generate and save interpolated masks between two input images. Clean output dir, it does."""
        # Clean output directory
        for f in os.listdir(self.output_directory):
            fp = os.path.join(self.output_directory, f)
            if os.path.isfile(fp):
                os.remove(fp)

        mask_1 = self.load_binary_mask(self.image_path_1)
        mask_2 = self.load_binary_mask(self.image_path_2)
        if mask_1.shape != mask_2.shape:
            mask_2 = cv2.resize(mask_2, (mask_1.shape[1], mask_1.shape[0]))
        sdm_1 = self.compute_signed_distance_map(mask_1)
        sdm_2 = self.compute_signed_distance_map(mask_2)
        interpolated_sdms = self.interpolate_sdms(sdm_1, sdm_2)
        interpolated_masks = []
        for i, sdm in enumerate(interpolated_sdms):
            mask = self.sdm_to_binary_mask(sdm)
            interpolated_masks.append(mask)
            filename = f"interpolated_mask_{i+1:03d}.png"
            self.save_mask(mask, filename)
        print(f"Generated {len(interpolated_masks)} interpolated masks in {self.output_directory}")
        return interpolated_masks


# Hardcoded variables as per requirements
image_path_1 = "/home/vector64/Documents/UDEM/9NO_SEMESTRE/PEF/tomography_3d/Cases/Ana_Cristina/Mask_Ana_1.png"
image_path_2 = "/home/vector64/Documents/UDEM/9NO_SEMESTRE/PEF/tomography_3d/Cases/Ana_Cristina/Mask_Ana_2.png"
output_directory = "Interpolated_Masks"
n_interpolated = 10

# Execute interpolation
if __name__ == "__main__":
    interpolator = MaskInterpolator(image_path_1, image_path_2, output_directory, n_interpolated)
    interpolator.generate_interpolated_masks()