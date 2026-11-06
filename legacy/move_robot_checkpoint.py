import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

project = Path(__file__).resolve().parent


# LOAD MODEL

model = mujoco.MjModel.from_xml_path(
    str(project / "track_b_scene.xml")
)

data = mujoco.MjData(model)

# ROBOT JOINTS

joint_names = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

home = np.deg2rad([
    0,
    -90,
    90,
    -90,
    -90,
    0
])


# Set robot to home position.

for name, angle in zip(joint_names, home):
    joint_id = model.joint(name).id
    qpos_index = model.jnt_qposadr[joint_id]

    data.qpos[qpos_index] = angle

mujoco.mj_forward(model, data)

# LOAD DETECTED APPROACH POINT

target_base = np.load(
    project / "target_approach_base.npy"
)

print("Target approach point in robot base frame [m]:")
print(np.round(target_base, 4))

# CONVERT BASE-FRAME TARGET → WORLD FRAME

base_id = model.body("base_link").id

R_WB = data.xmat[base_id].reshape(3, 3)
p_WB = data.xpos[base_id].copy()

target_world = (
    R_WB @ target_base
    + p_WB
)

print("\nTarget approach point in world frame [m]:")
print(np.round(target_world, 4))

# TOOL BODY

tool_id = model.body("tool0").id

# JOINT POSITION AND DOF INDICES

qpos_indices = []
dof_indices = []

for name in joint_names:

    joint_id = model.joint(name).id

    qpos_indices.append(
        model.jnt_qposadr[joint_id]
    )

    dof_indices.append(
        model.jnt_dofadr[joint_id]
    )

# SIMPLE DAMPED-LEAST-SQUARES IK

max_iterations = 500
tolerance = 0.001       # 1 mm
damping = 0.05
step_size = 0.5

jacp = np.zeros(
    (3, model.nv)
)

jacr = np.zeros(
    (3, model.nv)
)


for iteration in range(max_iterations):

    mujoco.mj_forward(model, data)

    # Current tool position in world coordinates.
    current_position = data.xpos[tool_id].copy()

    error = target_world - current_position

    error_norm = np.linalg.norm(error)

    if error_norm < tolerance:
        print(
            f"\nIK converged after "
            f"{iteration} iterations."
        )
        break

    # Calculate translational Jacobian of tool0.
    mujoco.mj_jacBody(
        model,
        data,
        jacp,
        jacr,
        tool_id
    )

    # Only use the six UR3 joints.
    J = jacp[:, dof_indices]

    # Damped least-squares inverse:
    #
    # dq = J^T (J J^T + λ²I)^-1 e

    A = (
        J @ J.T
        + damping**2 * np.eye(3)
    )

    dq = (
        J.T
        @ np.linalg.solve(A, error)
    )

    # Prevent one iteration from making
    # a very large joint movement.

    max_joint_step = 0.08

    dq = np.clip(
        step_size * dq,
        -max_joint_step,
        max_joint_step
    )

    for index, delta in zip(
        qpos_indices,
        dq
    ):
        data.qpos[index] += delta

else:

    print(
        "\nWARNING: IK did not fully converge."
    )


mujoco.mj_forward(model, data)

# STORE THE SOLVED JOINT CONFIGURATION

q_target = data.qpos[
    qpos_indices
].copy()

print("\nSolved joint configuration [deg]:")

print(
    np.round(
        np.rad2deg(q_target),
        2
    )
)

print("\nFinal tool position [world m]:")

print(
    np.round(
        data.xpos[tool_id],
        4
    )
)

print("\nPosition error [m]:")

print(
    np.round(
        target_world
        - data.xpos[tool_id],
        5
    )
)

# RETURN TO HOME BEFORE ANIMATION
for index, angle in zip(
    qpos_indices,
    home
):
    data.qpos[index] = angle

mujoco.mj_forward(model, data)

# SMOOTHLY MOVE HOME → TARGET

steps = 200

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    for i in range(steps + 1):

        alpha = i / steps

        # Smooth interpolation.
        alpha_smooth = (
            3 * alpha**2
            - 2 * alpha**3
        )

        q = (
            (1 - alpha_smooth) * home
            + alpha_smooth * q_target
        )

        for index, angle in zip(
            qpos_indices,
            q
        ):
            data.qpos[index] = angle

        mujoco.mj_forward(
            model,
            data
        )

        viewer.sync()

        time.sleep(0.015)

    print(
        "\nRobot reached detected "
        "approach position."
    )

    # Keep viewer open.
    while viewer.is_running():

        viewer.sync()

        time.sleep(0.02)
