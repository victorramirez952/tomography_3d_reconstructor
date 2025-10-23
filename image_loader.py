#!/usr/bin/env python3
"""
Image loading module for tomography reconstruction.
Handles mask image loading and preprocessing.
"""

import cv2
import numpy as np
import glob
import os


class ImageLoader:
    """Handles loading and preprocessing of mask images."""
    
    def __init__(self):
        self.mask_files = []
        self.mask_images = []
        self.image_width = None
        self.image_height = None
        self.num_slices = 0
    
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
            
            return True
            
        except Exception as e:
            print(f"Loading failed: {e}")
            return False
    
    def get_mask_images(self) -> list:
        """Get loaded mask images."""
        return self.mask_images
    
    def get_image_dimensions(self) -> tuple:
        """Get image dimensions (width, height)."""
        return self.image_width, self.image_height
    
    def get_num_slices(self) -> int:
        """Get number of loaded slices."""
        return self.num_slices