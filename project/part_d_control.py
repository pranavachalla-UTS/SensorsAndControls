"""
PART D - Control: computed torque, LQR, and visual servoing.

Run:  python part_d_control.py

The controller has three layers:

  1. COMPUTED TORQUE cancels the arm's own dynamics, turning a coupled
     six-joint problem into six independent double integrators.
  2. An LQR on that double integrator decides the acceleration to ask for.
  3. PBVS sits on top and decides where the tool should go, using the block
     position the camera gave us.

Writes: results/part_d_*.png
"""

import numpy as np
import matplotlib.pyplot as plt
import control as ct
from pathlib import Path

import trackb as tb

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)
JOINT_NAMES = [j.replace("_joint", "") for j in tb.JOINTS]


print("=" * 70)
print("1. CHOOSING THE GAINS")
print("=" * 70)

kp, kd, poles = tb.design_lqr()
print(f"""
After computed torque each joint is just  qddot = v, so we only have to design
one controller and use it six times.

The LQR cost is   sum of  q_err^2/e_max^2 + qdot_err^2/v_max^2 + v^2/a_max^2.
Dividing each term by the largest value we are willing to accept (Bryson's
rule) means the weights have units we can argue about instead of being picked
by trial and error:

  e_max = {tb.MAX_POS_ERROR} rad      a position error this big...
  v_max = {tb.MAX_VEL_ERROR} rad/s    ...matters as much as this velocity error...
  a_max = {tb.MAX_ACCEL} rad/s^2      ...and is worth this much correction.

  gives  kp = {kp:.0f} 1/s^2,  kd = {kd:.1f} 1/s
  closed-loop poles {np.round(poles, 4)}
  natural frequency {np.abs(np.log(poles[0])) / tb.DT:.0f} rad/s, damping {-np.log(np.abs(poles[0])) / np.abs(np.log(poles[0])):.2f}

Why not simply turn the gains up? kd multiplies the velocity ESTIMATE, and that
estimate carries noise. The sweep below shows what that costs.
""")

# Sweep on the double integrator itself - exactly the system the LQR is designed against.
A = np.array([[1.0, tb.DT], [0.0, 1.0]])
B = np.array([[tb.DT ** 2 / 2], [tb.DT]])

print("  e_max      kp      kd   settling   overshoot   noise -> torque")
print("  [rad]                     [ms]        [%]        [rad/s^2 rms]")
sweep_results = []
for e_max in [0.002, 0.005, 0.02, 0.05]:
    Q = np.diag([1 / e_max ** 2, 1 / tb.MAX_VEL_ERROR ** 2])
    R = np.array([[1 / tb.MAX_ACCEL ** 2]])
    K, _, _ = ct.dlqr(A, B, Q, R)

    # Step response from 10 mrad away.
    x = np.array([0.01, 0.0])
    history = [x[0]]
    for _ in range(1500):
        x = A @ x + B.ravel() * float((-K @ x).item())
        history.append(x[0])
    history = np.array(history)
    settled = np.flatnonzero(np.abs(history) > 0.02 * 0.01)
    settle_ms = (settled[-1] + 1) * tb.DT * 1e3 if len(settled) else 0.0
    overshoot = max(0.0, -history.min() / 0.01 * 100)

    # How much encoder noise reaches the commanded acceleration, via kp and kd.
    noise_to_torque = K[0, 1] * 0.04        # 0.04 rad/s of velocity estimate noise
    sweep_results.append((e_max, K[0, 0], K[0, 1], settle_ms, overshoot, noise_to_torque))
    mark = "  <- used" if e_max == tb.MAX_POS_ERROR else ""
    print(f"  {e_max:<7} {K[0,0]:7.0f} {K[0,1]:7.1f}   {settle_ms:7.0f}   {overshoot:8.1f}   "
          f"{noise_to_torque:12.2f}{mark}")

print("""
Tighter weights settle faster but push more sensor noise through to the motors.
We took e_max = 0.005 rad: it settles in about a tenth of a second, which is
much quicker than the arm is ever asked to move, without making the torque
command twitchy.""")


print()
print("=" * 70)
print("2. TRACKING DURING AN ACTUAL PICK")
print("=" * 70)

success, info, log = tb.run_pick(seed=0)
t = log["t"]
tracking_error = 1e3 * (log["q"] - log["q_ref"])

print(f"""
  result                    {info['reason']}
  tracking error, RMS       {info['tracking_rms_mrad']:.2f} mrad
  tracking error, worst     {np.abs(tracking_error).max():.2f} mrad
  torque, RMS               {info['torque_rms']:.2f} N m
  torque, peak              {np.abs(log['tau']).max():.1f} N m  (limits are 56/56/28/12/12/12)

Sub-milliradian tracking on a 0.25 m forearm is about 0.1 mm at the tool, which
is well under what the camera can see anyway - so the controller is not the
thing limiting how accurately we grasp. The vision is.""")

fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

ax = axes[0]
for j in range(6):
    ax.plot(t, tracking_error[:, j], lw=0.8, label=JOINT_NAMES[j])
ax.set_ylabel("tracking error [mrad]")
ax.set_title(f"Joint tracking error (RMS {info['tracking_rms_mrad']:.2f} mrad)")
ax.legend(ncol=6, fontsize=7)
ax.grid(alpha=0.3)

ax = axes[1]
for j in range(6):
    ax.plot(t, log["tau"][:, j], lw=0.8, label=JOINT_NAMES[j])
sim = tb.Sim()
for limit in sim.torque_limit:
    ax.axhline(limit, color="0.6", ls=":", lw=0.6)
    ax.axhline(-limit, color="0.6", ls=":", lw=0.6)
ax.set_ylabel("joint torque [N m]")
ax.set_title("Commanded torque (dotted lines are the motor limits)")
ax.grid(alpha=0.3)

# Mark the task phases along the bottom so the shapes above make sense.
ax = axes[2]
phase_names = ["settle", "to_observe", "look", "align", "descend", "grasp", "lift"]
colours = plt.cm.tab10(np.linspace(0, 1, len(phase_names)))
for name, colour in zip(phase_names, colours):
    where = log["phase"] == name
    if np.any(where):
        ax.fill_between(t, 0, 1, where=where, color=colour, alpha=0.6, label=name)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xlabel("time [s]")
ax.set_title("Task phase")
ax.legend(ncol=7, fontsize=7, loc="upper center")

fig.tight_layout()
fig.savefig(RESULTS / "part_d_tracking.png", dpi=150)


print()
print("=" * 70)
print("3. VISUAL SERVOING")
print("=" * 70)
print("""
PBVS works in the robot's own coordinates rather than in the image:

  error  = where the block is  -  where the tool is       (a 6-vector)
  twist  = gain * error, then capped to a sensible speed
  twist += the block's estimated velocity                 (feedforward)
  qdot   = J(q)^+ twist                                   (damped least squares)

The goal is recomputed from the object filter on EVERY control step, so a new
camera frame immediately changes where the arm is heading. That is what makes
it closed-loop vision rather than "take a photo, then move blindly" - and Part
E measures exactly how much that is worth.

The feedforward term is there because the block is on a belt. A proportional
law alone cannot hold zero error against a target that keeps moving: it settles
at error = speed/gain, which at belt speed is about 14 mm, and the grasp
tolerance is 3 mm, so the arm would trail the block for ever and never trigger.
Adding the filter's velocity estimate makes the tool travel WITH the block and
puts the proportional term back in charge of the remaining offset only. This is
the same reason a servo tracking a ramp needs feedforward or an integrator -
except that here the velocity is not measured anywhere, it comes out of the
object filter (Part C).

J is inverted with damping rather than exactly. Near a singular configuration a
true inverse asks for enormous joint speeds; damping trades a little accuracy
for a command the motors can actually follow.""")

# Distance from the tool to the block, over the approach.
tcp_to_block = log["tcp"] - log["obj_true"][:, :3]
moving = np.isin(log["phase"], ["align", "descend", "grasp", "lift"])

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

ax = axes[0]
for k, name in enumerate("xyz"):
    ax.plot(t[moving], 1e3 * tcp_to_block[moving, k], lw=1.3, label=f"{name}")
ax.axhline(0, color="black", lw=0.6)
ax.set_xlabel("time [s]")
ax.set_ylabel("tool - block [mm]")
ax.set_title("Closing on the block")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
distance = 1e3 * np.linalg.norm(tcp_to_block[moving, :2], axis=1)
ax.semilogy(t[moving], np.maximum(distance, 1e-2), color="tab:blue", lw=1.4)
for name, colour in [("descend", "tab:orange"), ("grasp", "tab:red")]:
    where = np.flatnonzero(log["phase"] == name)
    if len(where):
        ax.axvline(t[where[0]], color=colour, ls=":", lw=1.2)
        ax.text(t[where[0]], ax.get_ylim()[1] * 0.4, f" {name}", color=colour, fontsize=8)
ax.set_xlabel("time [s]")
ax.set_ylabel("horizontal distance to block [mm]")
ax.set_title("PBVS drives the error down exponentially")
ax.grid(alpha=0.3, which="both")

fig.tight_layout()
fig.savefig(RESULTS / "part_d_servo.png", dpi=150)
print(f"\nSaved two figures to {RESULTS}")
