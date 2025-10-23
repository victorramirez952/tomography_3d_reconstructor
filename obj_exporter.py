#!/usr/bin/env python3
"""
OBJ file exporter module for 3D reconstruction.
Handles exporting 3D models to OBJ format.
"""

import numpy as np
from typing import Tuple


class OBJExporter:
    """Handles exporting 3D models to OBJ file format."""
    
    def __init__(self):
        pass
    
    def export_to_obj(self, vertices: np.ndarray, faces: np.ndarray, 
                     filename: str = "tomography_model.obj") -> bool:
        """Export 3D model to OBJ format."""
        try:
            with open(filename, 'w') as f:
                f.write("# Tomography reconstruction model\n")
                f.write(f"# {len(vertices)} vertices, {len(faces)} faces\n\n")
                
                for vertex in vertices:
                    f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                
                f.write("\n")
                
                for face in faces:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            
            print(f"Model exported: {filename}")
            return True
            
        except Exception as e:
            print(f"Export failed: {e}")
            return False