import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
import argparse
import os


def load_binary_mask(image_path):
    """Load binary mask image and convert to proper format."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to binary (white foreground, black background)
    _, binary_mask = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binary_mask


def compute_signed_distance_map(binary_mask):
    """Compute signed distance map from binary mask."""
    # Normalize to 0 and 1
    mask = binary_mask / 255.0
    
    # Distance to background (outside object)
    distance_outside = distance_transform_edt(mask)
    
    # Distance to foreground (inside object)
    distance_inside = distance_transform_edt(1 - mask)
    
    # Signed distance map: positive inside, negative outside
    sdm = distance_inside - distance_outside
    
    return sdm


def process_mask_interpolation(image_path):
    """Main function to process mask and generate signed distance map."""
    # Load binary mask
    binary_mask = load_binary_mask(image_path)
    
    # Compute signed distance map
    sdm_1 = compute_signed_distance_map(binary_mask)
    
    # Convert to matrix representation
    matrix_sdm_1 = sdm_1.astype(np.float32)
    
    return matrix_sdm_1


def main():
    parser = argparse.ArgumentParser(description='Generate signed distance map from binary mask')
    parser.add_argument('image_path', help='Path to binary mask image')
    args = parser.parse_args()
    
    try:
        # Process the mask
        matrix_sdm_1 = process_mask_interpolation(args.image_path)
        
        # Print matrix representation
        print("Signed Distance Map Matrix:")
        print(matrix_sdm_1)
        
        # Print statistics
        print(f"\nMatrix shape: {matrix_sdm_1.shape}")
        print(f"Min value: {matrix_sdm_1.min():.4f}")
        print(f"Max value: {matrix_sdm_1.max():.4f}")
        print(f"Mean value: {matrix_sdm_1.mean():.4f}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()