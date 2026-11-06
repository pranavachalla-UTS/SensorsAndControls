import unittest
import numpy as np
import mujoco
from robot_common import HOME, indices, load_scene, estimated_data, camera_to_base, transform, solve_ik
from joint_control import state_model, ranks, JointObserver, feedback
from vision import intrinsic_matrix, localise, capture_rays


class PipelineTests(unittest.TestCase):
    def test_model_ranks(self):
        self.assertEqual(ranks(*state_model(0.002)), (12, 12))

    def test_observer_recovers_unmeasured_velocity_and_survives_dropout(self):
        dt = 0.002
        observer = JointObserver(np.zeros(6), dt)
        velocity = np.arange(1, 7) * 0.02
        for k in range(1, 1501):
            reading = velocity*k*dt if k % 4 else None
            observer.update(reading, np.zeros(6))
        np.testing.assert_allclose(observer.x[6:], velocity, atol=0.002)
        self.assertGreater(np.linalg.eigvalsh(observer.P).min(), 0)
        np.testing.assert_allclose(observer.P, observer.P.T, atol=1e-12)

    def test_optical_frame_backprojection(self):
        model, data = load_scene()
        T_BC = camera_to_base(model, data)
        T_WB = transform(data, model.body('base_link').id)
        p_camera = np.array([0.02, 0.03, 0.2, 1])
        world = T_WB @ T_BC @ p_camera
        camera = model.camera('wrist_camera').id
        expected = data.cam_xpos[camera] + data.cam_xmat[camera].reshape(3, 3) @ [0.02, -0.03, -0.2]
        np.testing.assert_allclose(world[:3], expected, atol=1e-12)

    def test_missing_target_fails(self):
        with self.assertRaises(RuntimeError):
            localise(np.zeros((40, 40, 3)), np.ones((40, 40)), np.eye(3), np.eye(4))

    def test_cpu_depth_localisation(self):
        model, data = load_scene()
        rgb, depth = capture_rays(model, data)
        K = intrinsic_matrix(480, 640, 65)
        surface, mask = localise(rgb, depth, K, camera_to_base(model, data))
        T_WB = transform(data, model.body('base_link').id)
        world = (T_WB @ np.r_[surface, 1])[:3]
        expected = data.xpos[model.body('target').id] + [0, 0, 0.02]
        self.assertLess(np.linalg.norm(world-expected), 0.003)

    def test_torque_motion_uses_dynamics(self):
        model, data = load_scene()
        qi, vi = indices(model)
        observer = JointObserver(HOME, model.opt.timestep)
        reference = HOME + [0.04, -0.03, 0.02, 0.01, 0, 0]
        acceleration = np.zeros(6)
        for _ in range(1000):
            observer.update(data.qpos[qi].copy(), acceleration)
            torque, acceleration = feedback(model, observer, reference, np.zeros(6), np.zeros(6))
            data.ctrl[:] = torque
            mujoco.mj_step(model, data)
        self.assertLess(np.max(np.abs(data.qpos[qi]-reference)), 0.002)
        self.assertFalse(np.any(data.eq_active))

    def test_ik_only_mutates_planning_copy(self):
        model, data = load_scene()
        before = data.qpos.copy()
        R = data.xmat[model.body('tool0').id].reshape(3, 3)
        goal = data.site_xpos[model.site('cup_tip').id] + [0.01, 0, 0]
        q = solve_ik(model, HOME, goal, R)
        np.testing.assert_array_equal(data.qpos, before)
        solved = estimated_data(model, q)
        self.assertLess(np.linalg.norm(solved.site_xpos[model.site('cup_tip').id]-goal), 0.001)


if __name__ == '__main__':
    unittest.main()
