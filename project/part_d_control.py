import numpy as np
# Save figures without opening a desktop plotting window.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import control as ct
from pathlib import Path

import trackb as tb
from evidence import provenance, write_json

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)
JOINT_NAMES = [j.replace("_joint", "") for j in tb.JOINTS]


print("=" * 70)
print("1. CHOOSING THE GAINS")
print("=" * 70)

kp, kd, poles = tb.design_lqr()
print(f"\nAfter computed torque each joint is just  qddot = v, so we only have to design\none controller and use it six times.\n\nThe LQR cost is   sum of  q_err^2/e_max^2 + qdot_err^2/v_max^2 + v^2/a_max^2.\nDividing each term by the largest value we are willing to accept (Bryson's\nrule) means the weights have units we can argue about instead of being picked\nby trial and error:\n\n  e_max = {tb.MAX_POS_ERROR} rad      a position error this big...\n  v_max = {tb.MAX_VEL_ERROR} rad/s    ...matters as much as this velocity error...\n  a_max = {tb.MAX_ACCEL} rad/s^2      ...and is worth this much correction.\n\n  gives  kp = {kp:.0f} 1/s^2,  kd = {kd:.1f} 1/s\n  closed-loop poles {np.round(poles, 4)}\n  natural frequency {np.abs(np.log(poles[0])) / tb.DT:.0f} rad/s, damping {-np.log(np.abs(poles[0])) / np.abs(np.log(poles[0])):.2f}\n\nWhy not simply turn the gains up? kd multiplies the velocity ESTIMATE, and that\nestimate carries noise. The sweep below shows what that costs.\n")

# Sweep on the double integrator itself - exactly the system the LQR is designed against.
A = np.array([[1.0, tb.DT], [0.0, 1.0]])
B = np.array([[tb.DT ** 2 / 2], [tb.DT]])

print("  e_max      kp      kd   settling   overshoot   noise -> accel")
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
    noise_to_acceleration = K[0, 1] * 0.04        # 0.04 rad/s of velocity estimate noise
    sweep_results.append((e_max, K[0, 0], K[0, 1], settle_ms, overshoot, noise_to_acceleration))
    mark = "  <- used" if e_max == tb.MAX_POS_ERROR else ""
    print(f"  {e_max:<7} {K[0,0]:7.0f} {K[0,1]:7.1f}   {settle_ms:7.0f}   {overshoot:8.1f}   "
          f"{noise_to_acceleration:12.2f}{mark}")

print('\nTighter weights settle faster but push more sensor noise through to the motors.\nWe took e_max = 0.005 rad: it settles in about a tenth of a second, which is\nmuch quicker than the arm is ever asked to move, without making the torque\ncommand twitchy.')


print()
print("=" * 70)
print("2. TRACKING DURING AN ACTUAL PICK")
print("=" * 70)

success, info, log = tb.run_pick(seed=0)
t = log["t"]
tracking_error = 1e3 * (log["q"] - log["q_ref"])

print(f"\n  result                    {info['reason']}\n  tracking error, RMS       {info['tracking_rms_mrad']:.2f} mrad\n  tracking error, worst     {np.abs(tracking_error).max():.2f} mrad\n  torque, RMS               {info['torque_rms']:.2f} N m\n  torque, peak              {np.abs(log['tau']).max():.1f} N m  (limits are 56/56/28/12/12/12)\n\nSub-milliradian tracking on a 0.25 m forearm is about 0.1 mm at the tool, which\nis well under what the camera can see anyway - so the controller is not the\nthing limiting how accurately we grasp. The vision is.")

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
phase_names = ["settle", "to_observe", "look", "align", "descend", "grasp", "lift", "hold"]
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
print('\nPBVS works in the robot\'s own coordinates rather than in the image:\n\n  error  = where the block is  -  where the tool is       (a 6-vector)\n  twist  = gain * error, then capped to a sensible speed\n  twist += the block\'s estimated velocity                 (feedforward)\n  qdot   = J(q)^+ twist                                   (damped least squares)\n\nThe goal is recomputed from the object filter on EVERY control step, so a new\ncamera frame immediately changes where the arm is heading. That is what makes\nit closed-loop vision rather than "take a photo, then move blindly" - and Part\nE measures exactly how much that is worth.\n\nThe feedforward term is there because the block is on a belt. A proportional\nlaw alone cannot hold zero error against a target that keeps moving: it settles\nat error = speed/gain, which at belt speed is about 14 mm, and the grasp\ntolerance is 3 mm, so the arm would trail the block for ever and never trigger.\nAdding the filter\'s velocity estimate makes the tool travel WITH the block and\nputs the proportional term back in charge of the remaining offset only. This is\nthe same reason a servo tracking a ramp needs feedforward or an integrator -\nexcept that here the velocity is not measured anywhere, it comes out of the\nobject filter (Part C).\n\nJ is inverted with damping rather than exactly. Near a singular configuration a\ntrue inverse asks for enormous joint speeds; damping trades a little accuracy\nfor a command the motors can actually follow.')

# Distance from the tool to the block, over the approach.
tcp_to_block = log["tcp"] - log["obj_true"][:, :3]
moving = np.isin(log["phase"], ["align", "descend", "grasp", "lift", "hold"])

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
ax.set_title("Horizontal tracking error during approach")
ax.grid(alpha=0.3, which="both")

fig.tight_layout()
fig.savefig(RESULTS / "part_d_servo.png", dpi=150)
print(f"\nSaved two figures to {RESULTS}")

write_json(RESULTS / "control_report.json", {"trial": info,
           "lqr_kp": kp, "lqr_kd": kd,
           "weight_sweep_columns": ["e_max_rad", "kp", "kd", "settling_ms", "overshoot_percent", "accel_noise_rad_s2"],
           "weight_sweep_rows": sweep_results,
           "sweep_scope": "compensated double integrator; not six nonlinear robot sweeps",
           "peak_torque_nm": float(np.abs(log["tau"]).max()),
           "provenance": provenance()})
