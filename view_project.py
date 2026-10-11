import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

project = Path(__file__).resolve().parent

# Load the table scene and imported robot.
scene = ET.parse(project / "scene.xml").getroot()
robot = ET.parse(project / "ur3.xml").getroot()

compiler = scene.find("compiler")
if compiler is None:
    compiler = ET.SubElement(scene, "compiler")
compiler.set("angle", "radian")

# Add the robot's mesh definitions.
scene.insert(0, robot.find("asset"))

# Mount the robot on the tabletop, facing the target area.
base = robot.find("worldbody").find("body")
base.set("pos", "0.05 0 0.6")
base.set("quat", "0 0 0 1")
scene.find("worldbody").append(base)

# Temporary gravity compensation for this viewing checkpoint.
for body in base.iter("body"):
    body.set("gravcomp", "1")

for joint in base.iter("joint"):
    joint.set("damping", "1")

for geom in base.iter("geom"):
    geom.set("rgba", "0.55 0.65 0.75 1")

# Attach the camera to the robot's tool frame.
tool = base.find(".//body[@name='tool0']")

if tool is None:
    raise RuntimeError("Could not find the robot's tool0 frame")

ET.SubElement(
    tool,
    "camera",
    name="wrist_camera",
    pos="0.035 0 0.05",
    quat="0 1 0 0",
    fovy="65",
)

# Small visible camera housing; collisions disabled for this housing.
ET.SubElement(
    tool,
    "geom",
    name="camera_housing",
    type="box",
    pos="0.035 0 0.025",
    size="0.015 0.022 0.012",
    mass="0.04",
    rgba="0.15 0.15 0.15 1",
    contype="0",
    conaffinity="0",
)
# Visible bracket connecting the tool frame to the camera housing.
ET.SubElement(
    tool,
    "geom",
    name="camera_bracket",
    type="capsule",
    fromto="0 0 0  0.035 0 0.025",
    size="0.006",
    mass="0.01",
    rgba="0.35 0.35 0.35 1",
    contype="0",
    conaffinity="0",
)

# Save and load the combined scene.
combined = project / "track_b_scene.xml"
ET.indent(scene)
ET.ElementTree(scene).write(
    combined, encoding="utf-8", xml_declaration=True
)

model = mujoco.MjModel.from_xml_path(str(combined))
data = mujoco.MjData(model)

# Initialise the arm in a bent, downward-facing pose.
joint_names = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
home = np.deg2rad([0, -90, 90, -90, -90, 0])

for name, angle in zip(joint_names, home):
    joint_id = model.joint(name).id
    position_index = model.jnt_qposadr[joint_id]
    data.qpos[position_index] = angle

mujoco.mj_forward(model, data)

# Separate kinematics calculation using ideal joint-position measurements.
# Later, we will use the observer's estimated joint positions here.
kinematics = mujoco.MjData(model)

for name in joint_names:
    index = model.jnt_qposadr[model.joint(name).id]
    kinematics.qpos[index] = data.qpos[index]  # Ideal simulated encoder.

mujoco.mj_forward(model, kinematics)

camera_id = model.camera("wrist_camera").id

# Optical camera axes: right, down, forward.
# MuJoCo camera axes: right, up, backward.
T_WC = np.eye(4)
T_WC[:3, :3] = (
    kinematics.cam_xmat[camera_id].reshape(3, 3)
    @ np.diag([1, -1, -1])
)
T_WC[:3, 3] = kinematics.cam_xpos[camera_id]

# Robot base pose in the world.
base_id = model.body("base_link").id
T_WB = np.eye(4)
T_WB[:3, :3] = kinematics.xmat[base_id].reshape(3, 3)
T_WB[:3, 3] = kinematics.xpos[base_id]

T_BC = np.linalg.inv(T_WB) @ T_WC
np.save(project / "camera_to_base.npy", T_BC)

# Capture both images at the same robot pose.
with mujoco.Renderer(model, height=480, width=640) as renderer:
    renderer.update_scene(data, camera="wrist_camera")
    rgb = renderer.render().copy()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="wrist_camera")
    depth = renderer.render().copy()

# Save the measurements for our next step.
np.save(project / "wrist_rgb.npy", rgb)
np.save(project / "wrist_depth.npy", depth)

print("RGB shape:", rgb.shape)
print("Depth shape:", depth.shape)

# Display colour and depth side by side.
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(rgb)
axes[0].set_title("Wrist camera: RGB")

image = axes[1].imshow(depth, cmap="viridis")
axes[1].set_title("Wrist camera: depth")
fig.colorbar(image, ax=axes[1], label="Depth [m]")

plt.tight_layout()
plt.show()

# The robot viewer opens after you close the image figure.
mujoco.viewer.launch(model, data)