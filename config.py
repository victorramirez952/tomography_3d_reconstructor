#!/usr/bin/env python3
"""
Configuration file for Tomography 3D Reconstruction

All hardcoded variables for the tomography reconstruction process.
Modify these values according to your specific setup and requirements.
"""

# =============================================================================
# PHYSICAL DIMENSIONS (in millimeters)
# =============================================================================
X_LENGTH_MM = 165.7      # Width of images in mm
Y_LENGTH_MM = 110.2      # Height of images in mm  
TOTAL_DEPTH_MM = 11.25    # Depth of Side_1 images (main body) in mm

# =============================================================================
# DATA SOURCE PATH
# =============================================================================
# Path should contain three subfolders: Side_0, Side_1, Side_2
# Side_1: Main body (~90% of shape), depth = TOTAL_DEPTH_MM
# Side_0 & Side_2: Closing ends, each total depth = 2 * (TOTAL_DEPTH_MM / Side_1_count)
DATA_PATH = "/home/vector64/Documents/UDEM/9NO_SEMESTRE/PEF/tomography_3d/Cases/Maria_Guadalupe/"

# =============================================================================
# PROCESSING PARAMETERS
# =============================================================================
THRESHOLD = 200          # Threshold for mask binarization (0-255)
SUBSAMPLE_FACTOR = 2     # Factor for subsampling visualization points
SMOOTHING_ITERATIONS = 3 # Number of morphological smoothing iterations

# Which sides to load: [Side_0, Side_1, Side_2]
LOAD_SIDES = [True, True, True]  # Set to False to skip specific sides

# =============================================================================
# VISUALIZATION AND EXPORT CONTROL
# =============================================================================
SHOW_3D_VISUALIZATION = False    # Set to False to skip 3D visualization display
EXPORT_OBJ_MODEL = True         # Set to False to skip OBJ file export

# =============================================================================
# OUTPUT FILES
# =============================================================================
OBJ_FILENAME = "tomography_model.obj"
INTERACTIVE_HTML = "tomography_3d_interactive.html"

# =============================================================================
# ADVANCED PROCESSING OPTIONS
# =============================================================================
CLOSE_VOLUME_ENDS = True        # Add end caps to create closed volume
APPLY_SMOOTHING = True          # Apply morphological smoothing
CREATE_MANIFOLD = True          # Ensure manifold properties for 3D printing
ADD_VOLUME_PADDING = True       # Add padding for better surface extraction