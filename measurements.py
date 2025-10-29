import numpy as np
"""
13.27 cm x 88.91 mm
13.27 cm x 88.31 mm
13.29 cm x 88.13 mm
13.26 cm x 88.13 mm
13.27 cm x 87.95 mm
13.27 cm x 87.31 mm
13.26 cm x 88.13 mm
"""
width = [13.27, 13.27, 13.29, 13.26, 13.27, 13.27, 13.26]
height = [88.91, 88.31, 88.13, 88.13, 87.95, 87.31, 88.13]
# Obtener la mediana de las listas
medianWidth = np.median(width)
medianHeight = np.median(height)
print("Median Width:", medianWidth)
print("Median Height:", medianHeight)