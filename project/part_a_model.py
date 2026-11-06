"""
PART A - Modelling, controllability and observability.

Run:  python part_a_model.py

Three things:
  1. Linearise the UR3 about a working pose and show it is unstable in open
     loop - gravity pulls it over, so feedback is not optional.
  2. Check controllability and observability. In particular: can we get away
     with position-only encoders and no tachometers?
  3. Show what feedback linearisation does to the problem, and where the
     controller and observer poles end up.

Writes: results/part_a_poles.png
"""

import numpy as np
import matplotlib.pyplot as plt
import control as ct
from pathlib import Path

import trackb as tb

np.set_printoptions(precision=4, suppress=True)
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


def pbh_missing_modes(A, B, tol=1e-8):
    """Count the modes of A that B cannot reach (the PBH test).

    A mode (eigenvalue) lam is uncontrollable exactly when the matrix
    [A - lam*I, B] loses rank. Testing each mode separately is far more
    reliable than taking the rank of [B, AB, A^2 B, ...], whose entries here
    span about ten orders of magnitude - numpy's rank on that matrix gives a
    wrong answer for this arm (see the printout below).

    Swap A -> A.T and B -> C.T to get the observability version.
    """
    n = A.shape[0]
    modes = []
    for lam in np.linalg.eigvals(A):
        if not any(abs(lam - m) < 1e-6 * max(1.0, abs(m)) for m in modes):
            modes.append(lam)

    missing = 0
    for lam in modes:
        s = np.linalg.svd(np.hstack([A - lam * np.eye(n), B]), compute_uv=False)
        missing += int(np.sum(s < tol * s[0]))
    return missing


print("=" * 70)
print("1. THE ARM, LINEARISED")
print("=" * 70)

model = tb.ControllerModel()

# Work at the pose the robot actually uses to look at the table.
T_observe = np.eye(4)
T_observe[:3, :3] = np.diag([1.0, -1.0, -1.0])
sim = tb.Sim()
T_observe[:3, 3] = (sim.T_base_world @ np.append(tb.OBSERVE_TCP_WORLD, 1))[:3]
q0, ok = tb.ik_tcp(tb.SE3(T_observe, check=False), tb.Q_HOME)
print(f"Operating point (deg): {np.rad2deg(q0).round(1)}")

# Linearised about a still pose: A = [[0, I], [-M^-1 G, -M^-1 B]], B_u = [[0], [M^-1]].
M, _ = model.dynamics(q0, np.zeros(6))

# G = dg/dq by central differences: nudge each joint, see how gravity torque changes.
G = np.zeros((6, 6))
h = 1e-4
for j in range(6):
    step = np.zeros(6)
    step[j] = h
    _, g_plus = model.dynamics(q0 + step, np.zeros(6))
    _, g_minus = model.dynamics(q0 - step, np.zeros(6))
    G[:, j] = (g_plus - g_minus) / (2 * h)

Minv = np.linalg.inv(M)
A = np.block([[np.zeros((6, 6)), np.eye(6)],
              [-Minv @ G, -Minv @ np.diag(model.damping)]])
B = np.vstack([np.zeros((6, 6)), Minv])

eigs = np.linalg.eigvals(A)
print(f"\nInertia matrix diagonal:  {np.diag(M).round(3)}")
print(f"Gravity stiffness dg/dq, worst term: {np.abs(G).max():.2f} N m/rad")
print(f"\nOpen-loop eigenvalues (1/s), slowest first:")
for e in sorted(eigs, key=lambda z: -z.real):
    print(f"    {e.real:9.4f} {e.imag:+.4f}j")

worst = eigs.real.max()
print(f"\nLargest real part: {worst:+.3f} 1/s")
print("  -> UNSTABLE. Left alone the arm falls over under its own weight,")
print("     so the controller is holding it up, not just steering it.")


print()
print("=" * 70)
print("2. CAN WE CONTROL IT, AND CAN WE SEE IT?")
print("=" * 70)

C_encoders = np.hstack([np.eye(6), np.zeros((6, 6))])      # position only
C_tachos = np.hstack([np.zeros((6, 6)), np.eye(6)])        # velocity only
C_five = C_encoders[1:]                                    # pan encoder failed

print("\nFirst, why we do not use the textbook rank test here:")
naive = np.linalg.matrix_rank(ct.ctrb(A, B))
print(f"    rank(ctrb(A,B)) straight from numpy : {naive} / 12   <-- wrong")
print(f"    PBH test                            : "
      f"{12 - pbh_missing_modes(A, B)} / 12   <-- right")
print("    The controllability matrix has singular values spanning ~1e10, so")
print("    its numerical rank is meaningless. The PBH test checks one mode at")
print("    a time and stays well conditioned.")

print("\n  Sensors                                      controllable  observable")
rows = [
    ("6 torque motors, 6 position encoders", B, C_encoders),
    ("velocity sensors instead of encoders", None, C_tachos),
    ("encoders on joints 2-6 (pan encoder dead)", None, C_five),
]
for label, Bi, Ci in rows:
    c_txt = f"{12 - pbh_missing_modes(A, Bi):>7} / 12" if Bi is not None else "        -"
    o_txt = f"{12 - pbh_missing_modes(A.T, Ci.T):>6} / 12"
    print(f"  {label:<44}{c_txt}{o_txt}")

print("""
  Reading that table:

  * With one motor per joint the inertia matrix is invertible, so we can
    produce any joint acceleration we like: fully controllable.

  * With POSITION encoders alone, all twelve states are observable - including
    the six joint velocities we never measure. That is the formal reason we
    can leave tachometers off the robot and estimate velocity instead, which
    is exactly what Part C does.

  * With VELOCITY sensors alone we lose a state. Turning the shoulder about
    the vertical axis changes no gravity torque and no velocity reading, so a
    constant offset in that angle is invisible. Same story if the shoulder-pan
    encoder fails.""")


print()
print("=" * 70)
print("3. THE MODEL THE CONTROLLER ACTUALLY DESIGNS AGAINST")
print("=" * 70)

print("""
  The controller cancels the arm's own dynamics (Part D):
      tau = M(q) v + c(q,qdot) + g(q) + B qdot      =>      qddot = v
  so the coupled 12-state problem becomes six identical double integrators,
  one per joint, and we design one small controller for all of them.

  Per joint, with state [q, qdot]:   A = [[0,1],[0,0]]   B = [0,1]'   C = [1,0]

  The observer uses one extra state, d, for the acceleration the model did not
  predict - friction, and the weight of whatever is in the gripper:
      [q, qdot, d]        d behaves as a slow random walk""")

A2 = np.array([[0.0, 1.0], [0.0, 0.0]])
B2 = np.array([[0.0], [1.0]])
A3 = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
B3 = np.array([[0.0], [1.0], [0.0]])
C3_pos = np.array([[1.0, 0.0, 0.0]])
C3_vel = np.array([[0.0, 1.0, 0.0]])

print(f"\n  [q, qdot]    controllable : {2 - pbh_missing_modes(A2, B2)} / 2")
print(f"  [q, qdot, d] controllable : {3 - pbh_missing_modes(A3, B3)} / 3"
      "   (we cannot steer a disturbance - as expected)")
print(f"  [q, qdot, d] observable from position : "
      f"{3 - pbh_missing_modes(A3.T, C3_pos.T)} / 3   <-- so we can estimate d")
print(f"  [q, qdot, d] observable from velocity : "
      f"{3 - pbh_missing_modes(A3.T, C3_vel.T)} / 3")

# Controller gains and the observer's steady-state gains, for the pole plot.
kp, kd, ctrl_poles = tb.design_lqr()
print(f"\n  LQR gains: kp = {kp:.0f} 1/s^2, kd = {kd:.1f} 1/s")

observer = tb.JointObserver()
P = observer.P[0] if observer.P[0].any() else np.diag([observer.R, 0.05 ** 2, 5.0 ** 2])
H = np.array([[1.0, 0.0, 0.0]])
for _ in range(5000):                       # iterate to the steady state
    P = observer.F @ P @ observer.F.T + observer.Q
    K = P @ H.T / (H @ P @ H.T + observer.R)
    P = (np.eye(3) - K @ H) @ P
obs_poles = np.linalg.eigvals((np.eye(3) - K @ H) @ observer.F)

# Discrete poles z to continuous speed |s| = |ln z| / dt, so designs compare in rad/s.
speed = lambda z: np.abs(np.log(z.astype(complex))) / tb.DT
print(f"  Controller speed : {speed(ctrl_poles).max():.0f} rad/s")
print(f"  Observer speed   : {speed(obs_poles).min():.0f} rad/s")
print(f"\n  The observer is about {speed(obs_poles).min() / speed(ctrl_poles).max():.0f}x faster than the controller.")
print("  That is the separation principle in practice: the estimate settles")
print("  well before the controller reacts, so the two can be designed apart.")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.axvline(0, color="grey", lw=0.8)
ax1.plot(eigs.real, eigs.imag, "x", color="black", ms=9, mew=2)
ax1.set_xlabel("real part [1/s]")
ax1.set_ylabel("imaginary part [1/s]")
ax1.set_title("Open-loop arm poles\n(anything right of 0 is unstable)")
ax1.grid(alpha=0.3)

theta = np.linspace(0, 2 * np.pi, 200)
ax2.plot(np.cos(theta), np.sin(theta), color="grey", lw=0.8)
ax2.plot(ctrl_poles.real, ctrl_poles.imag, "o", color="tab:orange", ms=9, label="LQR controller")
ax2.plot(obs_poles.real, obs_poles.imag, "s", color="tab:blue", ms=8, label="Kalman observer")
ax2.set_xlim(0.4, 1.05)
ax2.set_ylim(-0.3, 0.3)
ax2.set_aspect("equal")
ax2.set_xlabel("real part")
ax2.set_ylabel("imaginary part")
ax2.set_title("Closed-loop poles, one joint\n(inside the circle = stable)")
ax2.legend()
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(RESULTS / "part_a_poles.png", dpi=150)
print(f"\nSaved {RESULTS / 'part_a_poles.png'}")
