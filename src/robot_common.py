from pathlib import Path
import mujoco
import numpy as np

# Paths are relative to this file, so commands work from any folder.
PROJECT = Path(__file__).resolve().parent.parent
MODELS = PROJECT / 'models'
CALIBRATION = PROJECT / 'data' / 'calibration'
RESULTS = PROJECT / 'results'
JOINTS = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
          'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
HOME = np.deg2rad([0, -90, 90, -90, -90, 0])


def indices(model):
    ids = [model.joint(name).id for name in JOINTS]
    return model.jnt_qposadr[ids], model.jnt_dofadr[ids]


def load_scene():
    model = mujoco.MjModel.from_xml_path(str(MODELS / 'dynamic_scene.xml'))
    data = mujoco.MjData(model)
    qi, vi = indices(model)
    # Assigning positions here sets the initial condition only.
    data.qpos[qi] = HOME
    mujoco.mj_forward(model, data)
    return model, data


def estimated_data(model, q, velocity=None):
    # This separate model receives encoder estimates, never true simulator state.
    data = mujoco.MjData(model)
    qi, vi = indices(model)
    data.qpos[qi] = q
    if velocity is not None:
        data.qvel[vi] = velocity
    mujoco.mj_forward(model, data)
    return data


def transform(data, body_id):
    T = np.eye(4)
    T[:3, :3] = data.xmat[body_id].reshape(3, 3)
    T[:3, 3] = data.xpos[body_id]
    return T


def camera_to_base(model, estimate):
    camera = model.camera('wrist_camera').id
    T_WC = np.eye(4)
    # Optical axes are right/down/forward; MuJoCo uses right/up/backward.
    T_WC[:3, :3] = estimate.cam_xmat[camera].reshape(3, 3) @ np.diag([1, -1, -1])
    T_WC[:3, 3] = estimate.cam_xpos[camera]
    T_WB = transform(estimate, model.body('base_link').id)
    return np.linalg.inv(T_WB) @ T_WC


def solve_ik(model, q_start, point_world, rotation):
    # IK changes a planning copy only. The actual robot moves through torques.
    data = estimated_data(model, q_start)
    qi, vi = indices(model)
    tip = model.site('cup_tip').id
    tool = model.body('tool0').id
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(800):
        mujoco.mj_forward(model, data)
        R = data.xmat[tool].reshape(3, 3)
        position_error = point_world - data.site_xpos[tip]
        # Cross products give a small-angle orientation error in world axes.
        angle_error = sum(np.cross(R[:, i], rotation[:, i]) for i in range(3)) / 2
        error = np.r_[position_error, angle_error]
        if np.linalg.norm(position_error) < 0.0005 and np.linalg.norm(angle_error) < 0.005:
            return data.qpos[qi].copy()
        mujoco.mj_jacSite(model, data, jp, jr, tip)
        J = np.vstack([jp[:, vi], jr[:, vi]])
        dq = J.T @ np.linalg.solve(J @ J.T + 0.01**2 * np.eye(6), error)
        data.qpos[qi] += np.clip(0.5 * dq, -0.04, 0.04)
        for i, name in enumerate(JOINTS):
            joint = model.joint(name).id
            if model.jnt_limited[joint]:
                data.qpos[qi[i]] = np.clip(data.qpos[qi[i]], *model.jnt_range[joint])
    raise RuntimeError('IK failed: requested pose is not reachable from this seed')


def full_mass_matrix(model, data):
    # MuJoCo changed this call between releases; support both interfaces.
    matrix = np.zeros((model.nv, model.nv))
    if hasattr(data, "qM"):
        mujoco.mj_fullM(model, matrix, data.qM)
    else:
        mujoco.mj_fullM(model, data, matrix)
    return matrix
