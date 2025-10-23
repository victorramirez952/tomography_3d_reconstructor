#!/usr/bin/env python3
"""
Configuration file for Tomography 3D Reconstruction

All hardcoded variables for the tomography reconstruction process.
Modify these values according to your specific setup and requirements.
"""

# =============================================================================
# PHYSICAL DIMENSIONS (in millimeters)
# =============================================================================
X_LENGTH_MM = 64.62      # Width of images in mm
Y_LENGTH_MM = 43.08      # Height of images in mm  
TOTAL_DEPTH_MM = 8.75      # Total depth covered by all slices in mm

# =============================================================================
# DATA SOURCE PATH
# =============================================================================
# Path where mask images are located (files starting with "Mask_")
DATA_PATH = "/home/vector64/Documents/UDEM/9NO_SEMESTRE/PEF/tomography_3d/Krim/"

# =============================================================================
# PROCESSING PARAMETERS
# =============================================================================
THRESHOLD = 200          # Threshold for mask binarization (0-255)
SUBSAMPLE_FACTOR = 2     # Factor for subsampling visualization points
SMOOTHING_ITERATIONS = 3 # Number of morphological smoothing iterations

# =============================================================================
# VISUALIZATION AND EXPORT CONTROL
# =============================================================================
SHOW_3D_VISUALIZATION = False    # Set to False to skip 3D visualization display
EXPORT_OBJ_MODEL = True         # Set to False to skip OBJ file export
SHOW_SLICE_PREVIEW = False      # Show preview of selected slices
NUM_PREVIEW_SLICES = 9          # Number of slices to show in preview

# =============================================================================
# OUTPUT FILES
# =============================================================================
OBJ_FILENAME = "tomography_model.obj"
VISUALIZATION_PNG = "tomography_3d_reconstruction.png"
INTERACTIVE_HTML = "tomography_3d_interactive.html"
SLICE_PREVIEW_PNG = "slice_visualization.png"

# =============================================================================
# ADVANCED PROCESSING OPTIONS
# =============================================================================
CLOSE_VOLUME_ENDS = True        # Add end caps to create closed volume
APPLY_SMOOTHING = True          # Apply morphological smoothing
CREATE_MANIFOLD = True          # Ensure manifold properties for 3D printing
ADD_VOLUME_PADDING = True       # Add padding for better surface extraction