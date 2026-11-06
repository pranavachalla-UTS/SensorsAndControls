# Run with python tests.py. Tests do not need a graphics connection.
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import mujoco
import trackb as tb
from evidence import clean, save_trial, wilson


class ProjectTests(unittest.TestCase):
    def test_kinematics_matches_physics_at_several_poses(self):
        sim = tb.Sim(seed=1)
        for delta in [np.zeros(6), np.array([.1, -.1, .2, .1, -.2, .1])]:
            q = tb.Q_HOME + delta
            sim.set_joints(q)
            np.testing.assert_allclose(tb.fk_tcp(q).t, sim.true_tcp(), atol=1e-8)

    def test_mass_matrix_is_symmetric_positive_and_torque_is_limited(self):
        model = tb.ControllerModel()
        M, bias = model.dynamics(tb.Q_HOME, np.zeros(6))
        np.testing.assert_allclose(M, M.T, atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(M).min(), 0)
        limits = np.array([56, 56, 28, 12, 12, 12])
        tau, applied = tb.joint_torque(model, tb.Q_HOME, np.zeros(6), np.zeros(6),
                                      tb.Q_HOME + 2, np.zeros(6), np.zeros(6), 942, 57.5, limits)
        self.assertTrue(np.all(np.abs(tau) <= limits))
        np.testing.assert_allclose(M @ applied + bias, tau, atol=1e-10)

    def test_angle_wrap_and_rotation_at_pi(self):
        self.assertAlmostEqual(tb.wrap_pi_2(np.pi + .1), .1)
        rotation = tb.rotation_log(tb.rot_z(np.pi))
        self.assertTrue(np.all(np.isfinite(rotation)))
        self.assertAlmostEqual(np.linalg.norm(rotation), np.pi)

    def test_backprojection_uses_optical_depth(self):
        K = np.array([[400., 0, 320], [0, 400., 240], [0, 0, 1]])
        result = tb.backproject(np.array([360]), np.array([280]), np.array([2.]), K)
        np.testing.assert_allclose(result, [[.2, .2, 2.]])
        np.testing.assert_allclose(tb.backproject(np.array([360]), np.array([280]),
                                                 np.array([2.]), K, np.zeros(5)), result)

    def test_missing_red_and_invalid_depth_are_rejected(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertIsNone(tb.detect_red_block(rgb))
        self.assertIsNone(tb.locate_block(np.ones((100, 100), bool),
                          np.full((100, 100), np.nan), np.eye(3), np.eye(4)))

    def test_assessment_requires_calibration_files(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(tb, "HERE", Path(folder)):
            with self.assertRaises(FileNotFoundError):
                tb.load_calibration()
            with self.assertRaises(FileNotFoundError):
                tb.load_noise_model()

    def test_filter_covariance_stays_positive_and_velocity_is_inferred(self):
        observer = tb.JointObserver()
        observer.start(np.zeros(6))
        for k in range(1500):
            observer.update(np.ones(6) * k * tb.DT * .2)
            self.assertGreater(np.linalg.eigvalsh(observer.P).min(), 0)
            observer.predict(np.zeros(6))
        np.testing.assert_allclose(observer.qd, .2, atol=.005)
        np.testing.assert_allclose(observer.P, observer.P.transpose(0, 2, 1), atol=1e-12)

    def test_nees_is_evaluated_before_prediction(self):
        observer = tb.JointObserver()
        observer.start(np.zeros(6))
        observer.x[:, 1] = .7
        # A perfect same-time estimate has zero error. Predicting changes its timestamp.
        self.assertEqual(float(observer.nees(np.zeros(6), np.ones(6)*.7).max()), 0.)
        observer.predict(np.zeros(6))
        self.assertGreater(float(observer.nees(np.zeros(6), np.ones(6)*.7).min()), 0.)
        # A short physical run exercises the actual logging without opening a camera.
        _, _, history = tb.run_pick(seed=0, max_time=.12, allow_nominal=True, persist=False)
        errors = np.stack([history["q"]-history["q_hat"], history["qd"]-history["qd_hat"]], axis=-1)
        covariance = history["q_cov"][:, :, :2, :2]
        expected = np.einsum("tni,tnij,tnj->tn", errors, np.linalg.inv(covariance), errors)
        np.testing.assert_allclose(history["nees"], expected, rtol=1e-10, atol=1e-10)

    def test_object_outlier_gate_and_reinitialisation(self):
        filt = tb.ObjectFilter(np.ones(4)*.001, np.zeros(4))
        filt.update(np.zeros(4), .2)
        for k in range(4):
            status = filt.update(np.array([.1, 0, 0, 0]), .2)
        self.assertEqual(status, "reset")
        self.assertEqual(filt.n_resets, 1)
        np.testing.assert_allclose(filt.x[:4], [.1, 0, 0, 0])
        self.assertGreater(np.linalg.eigvalsh(filt.P).min(), 0)

    def test_confidence_rule_rejects_old_or_uncertain_targets(self):
        self.assertTrue(tb.grasp_confident(np.eye(8)*1e-6, .1))
        self.assertFalse(tb.grasp_confident(np.eye(8)*1e-6, 2))
        self.assertFalse(tb.grasp_confident(np.eye(8)*.01, .1))
        self.assertFalse(tb.grasp_confident(None, 0))

    def test_lift_needs_contact_full_hold_and_no_drop(self):
        self.assertFalse(tb.lift_verified([.15]*10, [.01]*10, True))
        self.assertFalse(tb.lift_verified([.15]*250, [.01]*250, False))
        self.assertTrue(tb.lift_verified([.15]*250, [.01]*250, True))
        self.assertFalse(tb.lift_verified([.15]*249+[.01], [.01]*250, True))
        self.assertFalse(tb.lift_verified([.15]*250, [.2]*250, True))

    def test_trial_evidence_round_trip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / "results") as folder:
            # save_trial paths normally live within RESULTS for portable history links.
            info = {"seed": 9, "success": True, "settings": {"noise_scale": 1}}
            prefix = save_trial(info, {"t": np.array([0, .002])}, folder)
            record = json.loads(prefix.with_suffix('.json').read_text())
            self.assertEqual(record['seed'], 9)
            self.assertTrue(prefix.with_suffix('.npz').exists())
            self.assertIn('source_sha256', record['provenance'])
            with self.assertRaises(FileExistsError):
                save_trial(info, {}, folder)
        self.assertEqual(clean(float('nan')), None)

    def test_success_intervals_express_small_sample_uncertainty(self):
        low, high = wilson(15, 15)
        self.assertAlmostEqual(low, .7961, places=3)
        self.assertEqual(high, 1)


if __name__ == "__main__":
    (Path(__file__).parent / "results").mkdir(exist_ok=True)
    unittest.main(verbosity=2)
