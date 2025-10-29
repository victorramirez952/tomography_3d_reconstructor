## Patient K

X_LENGTH_MM = 64.62      # Width of images in mm
Y_LENGTH_MM = 43.08      # Height of images in mm  
TOTAL_DEPTH_MM = 8.75    # Depth of Side_1 images (main body) in mm

# =============================================================================
# DATA SOURCE PATH
# =============================================================================
# Path should contain three subfolders: Side_0, Side_1, Side_2
# Side_1: Main body (~90% of shape), depth = TOTAL_DEPTH_MM
# Side_0 & Side_2: Closing ends, each total depth = 2 * (TOTAL_DEPTH_MM / Side_1_count)
DATA_PATH = "/home/vector64/Documents/UDEM/9NO_SEMESTRE/PEF/Tomography_3d/Cases/Krim"