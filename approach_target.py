from pathlib import Path

import numpy as np

# Current Assessment4 folder
project = Path(__file__).resolve().parent

# Load the target position already calculated by detect_target.py
surface_base = np.load(project / "target_surface_base.npy")

print("Detected target surface point [m]:")
print(np.round(surface_base, 4))

# Create a SAFE approach point above the object

safe_height = 0.04  # 10 cm above the detected surface

approach_point = surface_base.copy()
approach_point[2] += safe_height

print("\nSafe approach point [m]:")
print(np.round(approach_point, 4))

# Save it so we can use it for robot motion next
np.save(project / "target_approach_base.npy", approach_point)