import json
import mujoco
import numpy as np
from robot_common import RESULTS, HOME, load_scene, indices, estimated_data
from joint_control import state_model, ranks


def analyse():
    model, data = load_scene()
    qi, vi = indices(model)
    dt = model.opt.timestep
    A, B, C = state_model(dt)
    rank_c, rank_o = ranks(A, B, C)
    # MuJoCo linearises the actual torque-driven robot around the home equilibrium.
    # nv includes the free cube. Select only the robot's tangent-state indices.
    data.ctrl[:] = data.qfrc_bias[vi] - data.qfrc_passive[vi]
    A_full = np.zeros((2*model.nv, 2*model.nv))
    B_full = np.zeros((2*model.nv, model.nu))
    mujoco.mjd_transitionFD(model, data, 1e-6, True, A_full, B_full, None, None)
    arm = np.r_[vi, model.nv + vi]
    A_robot = A_full[np.ix_(arm, arm)]
    B_robot = B_full[arm]
    actual_c, actual_o = ranks(A_robot, B_robot, C)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M)
    result = {'dt_s': dt, 'state_order': 'six angles then six velocities',
              'compensated_input': 'joint acceleration rad/s^2',
              'compensated_controllability_rank': rank_c,
              'compensated_observability_rank': rank_o,
              'actual_torque_controllability_rank': actual_c,
              'actual_torque_observability_rank': actual_o,
              'feedback_discrete_poles': np.linalg.eigvals(A - B @ np.hstack([64*np.eye(6), 16*np.eye(6)])).real.tolist(),
              'home_mass_matrix': M[np.ix_(vi, vi)].tolist(),
              'assumptions': ['local free-motion linearisation at home',
                              'known rigid-body parameters; no ideal gravity compensation flag',
                              'contacts, saturation, and attached payload are model errors',
                              'encoder angles measured; velocities estimated']}
    RESULTS.mkdir(exist_ok=True)
    np.savez(RESULTS / 'state_matrices.npz', A=A, B=B, C=C,
             A_robot=A_robot, B_robot=B_robot)
    (RESULTS / 'model_analysis.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    analyse()
