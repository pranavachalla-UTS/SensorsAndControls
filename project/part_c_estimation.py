import numpy as np
# Save figures without opening a desktop plotting window.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import trackb as tb
from evidence import provenance, write_json

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)
JOINT_NAMES = [j.replace("_joint", "") for j in tb.JOINTS]

print("Running one pick to collect data...")
success, info, log = tb.run_pick(seed=0)
print(f"  {info['reason']}  ({info['time_s']:.1f} s simulated)\n")

t = log["t"]


print("=" * 70)
print("1. ESTIMATING A VELOCITY WE NEVER MEASURE")
print("=" * 70)

# The fair comparison: differencing the encoder, which needs no model at all.
finite_difference = np.vstack([np.zeros(6), np.diff(log["q_meas"], axis=0) / tb.DT])

kf_error = np.sqrt(np.mean((log["qd"] - log["qd_hat"]) ** 2))
fd_error = np.sqrt(np.mean((log["qd"][1:] - finite_difference[1:]) ** 2))

print(f"\nVelocity error over the whole pick:\n\n  difference the encoder   {1000.0 * fd_error:8.1f} mrad/s\n  Kalman filter            {1000.0 * kf_error:8.1f} mrad/s     {fd_error / kf_error:.0f}x better\n\nDifferencing amplifies the encoder's quantisation: one count of 0.38 mrad\ndivided by a 2 ms timestep is already 0.19 rad/s of noise, and that lands\nstraight on the controller's kd term. The filter instead uses the fact that a\nreal arm cannot change velocity instantly.")

fig, axes = plt.subplots(2, 3, figsize=(12, 5.5), sharex=True)
for j, ax in enumerate(axes.ravel()):
    ax.plot(t, finite_difference[:, j], color="0.75", lw=0.5,
            label="difference the encoder" if j == 0 else None)
    # Shade the filter's own +-2 sigma band, to see if it is honest about its confidence.
    ax.fill_between(t, log["qd_hat"][:, j] - 2 * log["qd_sigma"][:, j],
                    log["qd_hat"][:, j] + 2 * log["qd_sigma"][:, j],
                    color="tab:blue", alpha=0.25, lw=0,
                    label="filter's $\\pm2\\sigma$" if j == 0 else None)
    ax.plot(t, log["qd_hat"][:, j], color="tab:blue", lw=1.2,
            label="Kalman estimate" if j == 0 else None)
    ax.plot(t, log["qd"][:, j], "k--", lw=1.0, label="true" if j == 0 else None)
    ax.set_title(JOINT_NAMES[j], fontsize=10)
    ax.set_ylim(-1.6, 1.6)
    ax.grid(alpha=0.3)
    if j >= 3:
        ax.set_xlabel("time [s]")
    if j % 3 == 0:
        ax.set_ylabel("joint velocity [rad/s]")
axes[0, 0].legend(fontsize=8, ncol=2)
fig.suptitle(f"Joint velocity is estimated, not measured  "
             f"(Kalman {1e3*kf_error:.0f} mrad/s vs {1e3*fd_error:.0f} by differencing)",
             fontweight="bold")
fig.tight_layout()
fig.savefig(RESULTS / "part_c_velocity.png", dpi=150)


print()
print("=" * 70)
print("2. WHAT THE DISTURBANCE STATE IS DOING")
print("=" * 70)

_, info_off, _ = tb.run_pick(seed=0, use_d_feedforward=False)
print(f"\nd is the acceleration the controller's model did not account for. The model\nleaves out Coulomb friction and does not know the gripper is holding a block,\nso there is genuinely something there to find.\n\nCancelling it in the controller is worth a lot:\n\n  tracking error, d fed forward       {info['tracking_rms_mrad']:6.2f} mrad\n  tracking error, d ignored           {info_off['tracking_rms_mrad']:6.2f} mrad\n                                      {info_off['tracking_rms_mrad'] / info['tracking_rms_mrad']:.0f}x worse\n\nThis is how we get rid of steady-state droop without putting an integrator in\nthe controller - no wind-up to worry about, and d is a signal we can plot.")


print()
print("=" * 70)
print("3. CONSISTENCY, AND TUNING")
print("=" * 70)
print('\nA filter can be accurate and still be lying about how sure it is. The standard\ncheck is NEES: compare the actual error against the covariance the filter\nclaims. Averaged over [q, qdot] an honest filter scores about 2 (the number of\nstates). Much higher means over-confident, much lower means over-cautious.\n\nSweeping the two process-noise knobs:\n  sigma_a  how much acceleration we admit we cannot predict\n  sigma_d  how fast we let the disturbance estimate move\n')

print("  sigma_a  sigma_d     NEES   velocity err   tracking err")
print("                    (want 2)    [mrad/s]        [mrad]")
sweep = []
for sa, sd in [(2, 20), (2, 60), (2, 150), (10, 300)]:
    _, inf, _ = tb.run_pick(seed=0, observer_tuning=(sa, sd))
    sweep.append((sa, sd, inf["nees_mean"], inf["kf_qd_rmse_mrad_s"], inf["tracking_rms_mrad"]))
    mark = "  <- used" if (sa, sd) == (2, 60) else ""
    print(f"  {sa:7} {sd:8}  {inf['nees_mean']:7.2f}  {inf['kf_qd_rmse_mrad_s']:10.1f}  "
          f"{inf['tracking_rms_mrad']:11.2f}{mark}")

print("Use the saved aligned-time NEES and RMSE values to assess the trade-off.")
print("A mean far from 2 needs investigation; friction is only one possible cause.")
print("The tuning sweep is a single development seed, not held-out consistency evidence.")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))

ax = axes[0]
error_mrad = 1e3 * (log["q"] - log["q_hat"])
for j in range(6):
    ax.plot(t, error_mrad[:, j], lw=0.7, label=JOINT_NAMES[j])
ax.plot(t, 2e3 * log["q_sigma"][:, 0], "k--", lw=1.0, label="filter $\\pm2\\sigma$")
ax.plot(t, -2e3 * log["q_sigma"][:, 0], "k--", lw=1.0)
ax.set_xlabel("time [s]")
ax.set_ylabel("position error [mrad]")
ax.set_title("Estimate error vs claimed uncertainty")
ax.legend(fontsize=6, ncol=2)
ax.grid(alpha=0.3)

ax = axes[1]
for j in range(6):
    ax.plot(t, log["d_hat"][:, j], lw=1.0, label=JOINT_NAMES[j])
grasp_start = t[np.argmax(log["phase"] == "grasp")] if np.any(log["phase"] == "grasp") else None
if grasp_start:
    ax.axvline(grasp_start, color="red", ls=":", lw=1.2)
    ax.text(grasp_start, ax.get_ylim()[1] * 0.85, " picks up block", color="red", fontsize=7)
ax.set_xlabel("time [s]")
ax.set_ylabel("estimated disturbance [rad/s$^2$]")
ax.set_title("Estimated unmodelled acceleration")
ax.grid(alpha=0.3)

ax = axes[2]
nees = log["nees"].mean(axis=1)
ax.plot(t, nees, color="tab:blue", lw=0.6)
ax.axhline(2.0, color="black", ls="--", lw=1.2, label="ideal (2 states)")
ax.axhline(np.mean(nees[50:]), color="tab:orange", lw=1.2,
           label=f"mean {np.mean(nees[50:]):.1f}")
ax.set_yscale("log")
ax.set_xlabel("time [s]")
ax.set_ylabel("NEES")
ax.set_title("Consistency check")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(RESULTS / "part_c_observer.png", dpi=150)


print()
print("=" * 70)
print("4. THE OBJECT FILTER")
print("=" * 70)

seen = np.isfinite(log["obj_est"][:, 0])
grasp_i = np.argmax(log["phase"] == "grasp") if np.any(log["phase"] == "grasp") else len(t) - 1
window = seen & (np.arange(len(t)) <= grasp_i)
error_mm = 1e3 * np.linalg.norm(log["obj_est"][window, :3] - log["obj_true"][window, :3], axis=1)

speed_est = 1e3 * np.linalg.norm(log["obj_est"][window, 4:6], axis=1)
speed_true = 1e3 * np.linalg.norm(log["obj_vel_true"][window, :2], axis=1)

print(f"\nCamera frames this run:  {info['frames_used']} used, {info['frames_rejected']} rejected by the gate,\n                         {info['frames_blind']} with nothing detected.\n\nFinal position error before grasping: {error_mm[-1]:.2f} mm\nFinal yaw error:                      {info['object_yaw_error_deg']:.2f} deg\n\nThe block is on a belt doing {info['object_speed_mm_s']:.1f} mm/s, and the filter settles on\n{speed_est[-1]:.1f} mm/s - an error of {info['object_speed_error_mm_s']:.1f} mm/s. That number is the one worth\nlooking at, because NOTHING MEASURES VELOCITY. The camera returns a pose and\nonly a pose; the speed is inferred entirely from how that pose moves between\nframes, against a constant-velocity model. It is the same trick the joint\nobserver pulls to get joint velocity out of a position encoder, and it is the\nreason both of these are Kalman filters rather than smoothers.\n\nThe controller needs it. Position feedback alone puts the tool a steady\nspeed/gain behind a moving target - about {info['object_speed_mm_s'] / tb.PBVS_GAIN:.0f} mm here, two thirds of the way across\nthe 30 mm block - so it never settles inside tolerance and the grasp never\nfires. Feeding this velocity estimate forward cancels that lag outright.\n\nThe {info['frames_blind']} blind frames are not a failure. As the gripper comes down the block\ngoes out of the wrist camera's view - the camera sits off to one side of the\ntool, and at close range the block falls outside its field of view. You can see\nit in the right-hand plot: updates stop and the filter's uncertainty grows\nbecause it is now coasting. Coasting is survivable precisely because the filter\nknows the velocity: it keeps extrapolating the block along the belt instead of\nleaving it where it was last seen. The robot finishes the grasp on that\nextrapolation.\n\nThis is worth knowing about rather than hiding: an eye-in-hand camera always\ngoes blind at the moment of contact. A second fixed camera would fix it.")

fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.4))
labels = ["x [mm]", "y [mm]", "yaw [deg]"]
for ax, i, label in zip(axes[:2], [0, 1], labels[:2]):
    ax.fill_between(t[window], 1e3 * (log["obj_est"][window, i] - 2 * log["obj_sigma"][window, i]),
                    1e3 * (log["obj_est"][window, i] + 2 * log["obj_sigma"][window, i]),
                    color="tab:blue", alpha=0.25, lw=0, label="$\\pm2\\sigma$")
    ax.plot(t[window], 1e3 * log["obj_est"][window, i], color="tab:blue", label="filtered")
    ax.plot(t[window], 1e3 * log["obj_true"][window, i], "k--", lw=1.0, label="true")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(label)
    ax.set_title(f"block {label.split()[0]}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

# The velocity panel: the quantity nothing ever measured.
ax = axes[2]
velocity = log["obj_est"][window, 4:6]
velocity_cov = log["obj_cov"][window, 4:6, 4:6]
direction = velocity / np.maximum(np.linalg.norm(velocity, axis=1, keepdims=True), 1e-12)
speed_sigma = 1e3 * np.sqrt(np.maximum(0, np.einsum("ni,nij,nj->n", direction, velocity_cov, direction)))
ax.fill_between(t[window], speed_est - 2 * speed_sigma, speed_est + 2 * speed_sigma,
                color="tab:green", alpha=0.25, lw=0, label="$\\pm2\\sigma$")
ax.plot(t[window], speed_est, color="tab:green", label="filtered")
ax.plot(t[window], speed_true, "k--", lw=1.0, label="true")
ax.set_xlabel("time [s]")
ax.set_ylabel("speed [mm/s]")
ax.set_title("block speed (never measured)")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

ax = axes[3]
sigma_mm = 1e3 * np.linalg.norm(log["obj_sigma"][window, :3], axis=1)
ax.plot(t[window], error_mm, color="tab:blue", lw=1.4, label="actual error")
ax.plot(t[window], 2 * sigma_mm, "k--", lw=1.1, label="filter's $2\\sigma$")
blind = np.flatnonzero((log["cam"] == 1) & (np.arange(len(t)) <= grasp_i))
if len(blind):
    ax.axvspan(t[blind[0]], t[grasp_i], color="red", alpha=0.08)
    ax.text(t[blind[0]], ax.get_ylim()[1] * 0.75, " block out of view",
            color="red", fontsize=7)
ax.set_xlabel("time [s]")
ax.set_ylabel("position error [mm]")
ax.set_title("Error vs the filter's own uncertainty")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(RESULTS / "part_c_object.png", dpi=150)
print(f"\nSaved three figures to {RESULTS}")

# Retain the numerical study behind the figures.
write_json(RESULTS / "estimation_report.json", {"trial": info,
           "velocity_fd_rmse_rad_s": fd_error, "velocity_kf_rmse_rad_s": kf_error,
           "disturbance_disabled_trial": info_off,
           "tuning_columns": ["sigma_a", "sigma_d", "nees", "velocity_rmse_mrad_s", "tracking_rmse_mrad"],
           "tuning_rows": sweep,
           "timing": "posterior estimate, covariance and truth all at the same pre-step time",
           "tuning_scope": "development seed 0 only; not a multi-trial consistency proof",
           "provenance": provenance()})
