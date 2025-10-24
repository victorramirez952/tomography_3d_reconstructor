#!/usr/bin/env python3
"""
Image loading module for tomography reconstruction.
Handles mask image loading and preprocessing.
"""

import cv2
import numpy as np
import glob
import os
import re


class ImageLoader:
    """Handles loading and preprocessing of mask images."""
    
    def __init__(self):
        self.mask_files = []
        self.mask_images = []
        self.image_width = None
        self.image_height = None
        self.num_slices = 0
        self.side_0_count = 0
        self.side_1_count = 0
        self.side_2_count = 0
    
    def _extract_numeric_suffix(self, filename: str) -> int:
        """Extract numeric suffix from filename for proper sorting."""
        # Extract number from patterns like "Mask_123.png" or "mask_45.png"
        match = re.search(r'_(\d+)\.png$', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 0  # Default for files without numeric suffix
    
    def load_mask_images(self, directory: str = ".", threshold: int = 200, load_sides: list = [True, True, True]) -> bool:
        """Load mask images with prefix 'Mask_' from Section_0, Section_1, Section_2 subfolders.
        
        Args:
            directory: Path containing the Section folders
            threshold: Threshold for mask binarization  
            load_sides: Boolean array [Side_0, Side_1, Side_2] indicating which folders to load
        """
        try:
            side_folders = ['Section_0', 'Section_1', 'Section_2']
            all_mask_files = []
            self.side_0_count = 0
            self.side_1_count = 0
            self.side_2_count = 0

            for idx, side_folder in enumerate(side_folders):
                if not load_sides[idx]:
                    print(f"Skipping {side_folder} (disabled)")
                    continue
                    
                side_path = os.path.join(directory, side_folder)
                if not os.path.exists(side_path):
                    print(f"Folder {side_folder} not found in {directory}")
                    return False

                mask_pattern = os.path.join(side_path, "Mask_*.png")
                side_files = glob.glob(mask_pattern)
                
                if not side_files:
                    print(f"No mask images found in {side_folder}")
                    continue
                
                # Sort by numeric suffix for proper sequential order
                side_files = sorted(side_files, key=self._extract_numeric_suffix)
                print(f"Loading {len(side_files)} images from {side_folder} in numeric order")
                
                # Debug: Show first and last files to verify ordering
                if len(side_files) > 0:
                    first_file = os.path.basename(side_files[0])
                    last_file = os.path.basename(side_files[-1])
                    first_num = self._extract_numeric_suffix(side_files[0])
                    last_num = self._extract_numeric_suffix(side_files[-1])
                    print(f"  Range: {first_file} ({first_num}) → {last_file} ({last_num})")
                    
                all_mask_files.extend(side_files)

                if idx == 0:
                    self.side_0_count = len(side_files)
                elif idx == 1:
                    self.side_1_count = len(side_files)
                elif idx == 2:
                    self.side_2_count = len(side_files)

            self.mask_files = all_mask_files

            print(f"Found masks - Side_0: {self.side_0_count}, Side_1: {self.side_1_count}, Side_2: {self.side_2_count}")

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
    
    def get_side_counts(self) -> tuple:
        """Get counts for each side (Side_0, Side_1, Side_2)."""
        return self.side_0_count, self.side_1_count, self.side_2_count