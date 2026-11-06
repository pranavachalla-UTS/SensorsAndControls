import numpy as np
import mujoco
from robot_common import estimated_data, indices, full_mass_matrix


def state_model(dt):
    # State x = [six joint angles, six joint velocities]. Input is acceleration.
    # After approximate inverse-dynamics compensation: q'' = u + disturbance.
    I = np.eye(6)
    A = np.block([[I, dt * I], [np.zeros((6, 6)), I]])
    B = np.vstack([0.5 * dt**2 * I, dt * I])
    C = np.hstack([I, np.zeros((6, 6))])
    return A, B, C


def ranks(A, B, C):
    n = len(A)
    controllability = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])
    observability = np.vstack([C @ np.linalg.matrix_power(A, i) for i in range(n)])
    return int(np.linalg.matrix_rank(controllability)), int(np.linalg.matrix_rank(observability))


class JointObserver:
    def __init__(self, measurement, dt, encoder_std=0.001):
        self.A, self.B, self.C = state_model(dt)
        self.x = np.r_[measurement, np.zeros(6)]
        self.P = np.diag([encoder_std**2] * 6 + [0.1] * 6)
        # Acceleration uncertainty covers imperfect dynamics and external loads.
        self.Q = self.B @ self.B.T * 4.0 + np.eye(12) * 1e-10
        self.R = np.eye(6) * encoder_std**2

    def update(self, measurement, acceleration):
        self.x = self.A @ self.x + self.B @ acceleration
        self.P = self.A @ self.P @ self.A.T + self.Q
        if measurement is not None:
            innovation = measurement - self.C @ self.x
            S = self.C @ self.P @ self.C.T + self.R
            K = np.linalg.solve(S, self.C @ self.P).T
            self.x += K @ innovation
            # Joseph form avoids losing covariance symmetry numerically.
            E = np.eye(12) - K @ self.C
            self.P = E @ self.P @ E.T + K @ self.R @ K.T
        return self.x


def feedback(model, observer, reference, reference_velocity, reference_acceleration):
    q, v = observer.x[:6], observer.x[6:]
    # Critically damped second-order response: kp = wn^2, kd = 2*wn.
    acceleration = reference_acceleration + 64 * (reference - q) + 16 * (reference_velocity - v)
    acceleration = np.clip(acceleration, -30, 30)
    estimate = estimated_data(model, q, v)
    qi, vi = indices(model)
    M = full_mass_matrix(model, estimate)
    # Cancel estimated gravity/Coriolis and passive damping, then apply feedback.
    mass = M[np.ix_(vi, vi)]
    bias = estimate.qfrc_bias[vi] - estimate.qfrc_passive[vi]
    torque = mass @ acceleration + bias
    limits = model.actuator_ctrlrange
    torque = np.clip(torque, limits[:, 0], limits[:, 1])
    # Predict with the input remaining after torque clipping.
    applied_acceleration = np.linalg.solve(mass, torque - bias)
    return torque, applied_acceleration
