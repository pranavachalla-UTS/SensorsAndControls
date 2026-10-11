from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

project = Path(__file__).resolve().parent
rgb = np.load(project / "wrist_rgb.npy")
depth = np.load(project / "wrist_depth.npy")

# Convert to float before doing colour arithmetic.
colour = rgb.astype(float)
red = colour[:, :, 0]
green = colour[:, :, 1]
blue = colour[:, :, 2]

# Select pixels where red is sufficiently bright and dominant.
mask = (
    (red > 60)
    & (red > 1.8 * green)
    & (red > 1.8 * blue)
    & np.isfinite(depth)
    & (depth > 0)
)

# Image indexing is [row, column] = [v, u].
v, u = np.nonzero(mask)

if len(u) < 20:
    raise RuntimeError("Too few target pixels detected")

u_centre = u.mean()
v_centre = v.mean()
target_depth = np.median(depth[mask])

print(f"Detected pixels: {len(u)}")
print(f"Image centre of target: u={u_centre:.1f}, v={v_centre:.1f}")
print(f"Median target depth: {target_depth:.4f} m")

# Camera intrinsics from our configured 65-degree vertical field of view.
height, width = depth.shape
fovy = np.deg2rad(65)

fy = height / (2 * np.tan(fovy / 2))
fx = fy  # Square pixels.
cx = (width - 1) / 2
cy = (height - 1) / 2

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1],
])

# Back-project EACH detected pixel using its own depth.
Z = depth[v, u]
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy

points_camera = np.column_stack((X, Y, Z))

# A robust representative point on the detected surface.
surface_camera = np.median(points_camera, axis=0)

# Convert the surface point into the robot base frame.
T_BC = np.load(project / "camera_to_base.npy")

# Append 1 to use a 4×4 homogeneous transform.
surface_base = (T_BC @ np.r_[surface_camera, 1])[:3]

print("Detected surface point in robot base frame [m]:")
print(np.round(surface_base, 4))

np.save(project / "target_surface_base.npy", surface_base)

print("Camera intrinsic matrix K:\n", K)
print("Detected surface point in camera frame [m]:")
print(np.round(surface_camera, 4))

np.save(project / "target_surface_camera.npy", surface_camera)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(rgb)
axes[0].plot(u_centre, v_centre, "c+", markersize=15)
axes[0].set_title("Detected target centre")

axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Target mask: white = selected")

plt.tight_layout()
plt.show()