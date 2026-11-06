import argparse
import csv
import json
import time
import mujoco
import numpy as np
from robot_common import (PROJECT, RESULTS, CALIBRATION, HOME, load_scene, indices,
                          estimated_data, camera_to_base, transform, solve_ik)
from joint_control import JointObserver, feedback
from vision import capture, capture_rays, intrinsic_matrix, localise


def run_trial(seed=0, encoder_std=0.001, depth_std=0.001, dropout=0.0,
              disturbance=0.0, viewer=False, duration_scale=1.0, camera_backend="ray", output_dir=None):
    rng = np.random.default_rng(seed)
    model, data = load_scene()
    qi, vi = indices(model)
    # Trial setup may randomise ground truth; sensing/control cannot read it.
    target_joint = model.body('target').jntadr[0]
    target_q = model.jnt_qposadr[target_joint]
    data.qpos[target_q:target_q + 2] = [-0.25 + rng.uniform(-0.035, 0.035),
                                     rng.uniform(-0.035, 0.035)]
    mujoco.mj_forward(model, data)
    dt = model.opt.timestep
    observer = JointObserver(data.qpos[qi] + rng.normal(0, encoder_std, 6), dt, encoder_std)
    acceleration = np.zeros(6)
    log = []
    window = None
    if viewer:
        from mujoco import viewer as mj_viewer
        window = mj_viewer.launch_passive(model, data)

    def step(reference, velocity=None, feedforward=None, phase='hold'):
        nonlocal acceleration
        # These six readings represent noisy encoders, the only joint measurements.
        measurement = data.qpos[qi] + rng.normal(0, encoder_std, 6)
        if rng.random() < dropout:
            measurement = None
        observer.update(measurement, acceleration)
        torque, acceleration = feedback(model, observer, reference,
                                        np.zeros(6) if velocity is None else velocity,
                                        np.zeros(6) if feedforward is None else feedforward)
        data.ctrl[:] = torque
        data.qfrc_applied[:] = 0
        if 5 < data.time < 5.2:
            data.qfrc_applied[vi[1]] = disturbance
        # Ground truth below is evaluation only, never fed back to the controller.
        log.append([data.time, phase, *data.qpos[qi], *data.qvel[vi],
                    *observer.x, *np.diag(observer.P), *torque, *reference])
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)) or np.max(np.abs(data.qvel[vi])) > 15:
            raise RuntimeError('Unstable motion: trial stopped')
        if window is not None:
            if not window.is_running():
                raise RuntimeError('Viewer closed')
            if len(log) % 10 == 0:
                window.sync()
                time.sleep(dt * 10)

    def move(goal, seconds, phase):
        start = observer.x[:6].copy()
        n = int(seconds * duration_scale / dt)
        T = n * dt
        for k in range(1, n + 1):
            s = k / n
            blend = 10*s**3 - 15*s**4 + 6*s**5
            speed = (30*s**2 - 60*s**3 + 30*s**4) / T
            accel = (60*s - 180*s**2 + 120*s**3) / T**2
            step(start + blend*(goal-start), speed*(goal-start), accel*(goal-start), phase)
        for _ in range(250):
            step(goal, phase=phase)

    result = {'seed': seed, 'success': False, 'attached': False,
              'encoder_std_rad': encoder_std, 'depth_std_m': depth_std,
              'dropout': dropout, 'disturbance_nm': disturbance, 'camera_backend': camera_backend}
    try:
        for _ in range(500):
            step(HOME)
        K_file = CALIBRATION / 'intrinsics.json'
        if K_file.exists():
            calibration = json.loads(K_file.read_text())
            if calibration['backend'] != camera_backend:
                raise RuntimeError('Calibrate the selected camera backend before using this file')
            K = np.array(calibration['K'])
        else:
            K = intrinsic_matrix(480, 640, model.cam_fovy[model.camera('wrist_camera').id])

        def observe_surface():
            # Each visual update pairs a fresh image with its encoder-based pose.
            estimate = estimated_data(model, observer.x[:6])
            if camera_backend == 'ray':
                rgb, depth = capture_rays(model, data)
            else:
                with mujoco.Renderer(model, height=480, width=640) as renderer:
                    rgb, depth = capture(renderer, data)
            depth += rng.normal(0, depth_std, depth.shape)
            surface, mask = localise(rgb, depth, K, camera_to_base(model, estimate))
            T_WB = transform(estimate, model.body('base_link').id)
            return (T_WB @ np.r_[surface, 1])[:3]

        surface_world = observe_surface()
        estimate = estimated_data(model, observer.x[:6])
        # Fixed orientation keeps the cup pointing down throughout PBVS approach.
        rotation = estimate.xmat[model.body('tool0').id].reshape(3, 3).copy()
        result['surface_world_m'] = surface_world.tolist()
        # Truth is used here to score localisation, not to choose a waypoint.
        true_top = data.xpos[model.body('target').id] + [0, 0, 0.02]
        result['localisation_error_m'] = float(np.linalg.norm(surface_world - true_top))
        approach = surface_world + [0, 0, 0.08]
        q_approach = solve_ik(model, observer.x[:6], approach, rotation)
        move(q_approach, 3, 'approach')
        # Sampled PBVS: remeasure the object and correct the pose twice.
        # This closes the camera loop before descent instead of trusting one image.
        result['visual_updates'] = 1
        for _ in range(2):
            surface_world = observe_surface()
            approach = surface_world + [0, 0, 0.08]
            q_approach = solve_ik(model, observer.x[:6], approach, rotation)
            move(q_approach, 1, 'visual_correct')
            result['visual_updates'] += 1
        # Sphere extends 3 mm beyond the cup site; a 1 mm compression permits contact.
        q_touch = solve_ik(model, observer.x[:6], surface_world + [0, 0, 0.002], rotation)
        move(q_touch, 2, 'descend')
        cup_id, object_id = model.geom('cup').id, model.geom('target_geom').id
        contact = any({c.geom1, c.geom2} == {cup_id, object_id} for c in data.contact)
        result['touch_tip_world_m'] = data.site_xpos[model.site('cup_tip').id].tolist()
        result['contacts_at_touch'] = [[model.geom(c.geom1).name, model.geom(c.geom2).name] for c in data.contact]
        if not contact:
            raise RuntimeError('No cup/object contact: suction stays off')
        # Ideal suction: engage only after contact. Relative pose is set from estimates.
        estimate = estimated_data(model, observer.x[:6])
        T_WT = transform(estimate, model.body('tool0').id)
        T_WO = np.eye(4)
        T_WO[:3, 3] = surface_world - [0, 0, 0.02]
        T_TO = np.linalg.inv(T_WT) @ T_WO
        eq = model.equality('suction').id
        model.eq_data[eq, 3:6] = T_TO[:3, 3]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, T_TO[:3, :3].ravel())
        model.eq_data[eq, 6:10] = quat
        data.eq_active[eq] = True
        result['attached'] = True
        lift = surface_world + [0, 0, 0.12]
        q_lift = solve_ik(model, observer.x[:6], lift, rotation)
        move(q_lift, 3, 'lift')
        # A pick must raise the actual object and hold it for at least 0.5 seconds.
        target_id = model.body('target').id
        heights = []
        for _ in range(250):
            step(q_lift, phase='verify')
            heights.append(float(data.xpos[target_id, 2]))
        result['min_hold_height_m'] = min(heights)
        result['success'] = min(heights) > 0.70
        result['reason'] = 'lift held' if result['success'] else 'object did not clear table'
    except RuntimeError as error:
        result['reason'] = str(error)
    finally:
        if window is not None:
            window.close()
    folder = RESULTS / 'raw' if output_dir is None else output_dir
    folder.mkdir(parents=True, exist_ok=True)
    prefix = folder / f'trial_{seed}'
    columns = ['time_s', 'phase']
    for group in ['true_q', 'true_v', 'estimated_q', 'estimated_v', 'variance_q',
                  'variance_v', 'torque_nm', 'reference_q']:
        columns += [f'{group}_{i}' for i in range(6)]
    with prefix.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(log)
    if log:
        values = np.array([row[2:] for row in log], dtype=float)
        result['angle_rmse_rad'] = float(np.sqrt(np.mean((values[:, :6] - values[:, 12:18])**2)))
        result['velocity_rmse_rad_s'] = float(np.sqrt(np.mean((values[:, 6:12] - values[:, 18:24])**2)))
        error = values[:, :12] - values[:, 12:24]
        variances = values[:, 24:36]
        result['two_sigma_coverage'] = float(np.mean(np.abs(error) <= 2*np.sqrt(variances)))
        result['torque_rms_nm'] = float(np.sqrt(np.mean(values[:, 36:42]**2)))
        result['tracking_rmse_rad'] = float(np.sqrt(np.mean((values[:, :6] - values[:, 42:48])**2)))
    prefix.with_suffix('.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--camera', choices=['ray', 'raster'], default='ray')
    args = parser.parse_args()
    print(json.dumps(run_trial(seed=args.seed, viewer=args.viewer, camera_backend=args.camera), indent=2))
