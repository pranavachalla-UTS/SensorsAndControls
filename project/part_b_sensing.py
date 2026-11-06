import numpy as np
# Save figures without opening a desktop plotting window.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import mujoco
from pathlib import Path

import trackb as tb
from evidence import provenance, write_json

np.set_printoptions(precision=4, suppress=True)
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

rng = np.random.default_rng(2026)
BOARD_POS = np.array([-0.02, 0.0, 0.6005])      # board on the table, world frame


def look_at(eye, target, roll):
    # Build a camera pose at `eye` pointing at `target`, in the optical
    #     convention (x right, y down, z forward).
    z = target - eye
    z = z / np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z]) @ tb.rot_z(roll)
    T[:3, 3] = eye
    return T


def collect_views(n_wanted=20):
    # Move the arm to assorted poses above the board and photograph it.
    #
    #     The camera is deliberately moved to a RANGE of distances, tilts and roll
    #     angles. Calibration needs that variety: photos all taken from the same
    #     place cannot separate the focal length from the distance to the board.
    sim = tb.Sim(seed=1)
    # Park the block off to the side so it does not sit on the board.
    sim.set_block("target", 0.45, 0.35, 0.0)
    sim.data.mocap_pos[0] = BOARD_POS

    board_in_base = (sim.T_base_world @ np.append(BOARD_POS, 1))[:3]
    views = []
    attempts = 0
    while len(views) < n_wanted and attempts < 400:
        attempts += 1
        # Random viewpoint on a dome above the board.
        distance = rng.uniform(0.22, 0.34)
        tilt = np.deg2rad(rng.uniform(0, 28))
        around = rng.uniform(-np.pi, np.pi)
        eye = board_in_base + distance * np.array([np.sin(tilt) * np.cos(around),
                                                   np.sin(tilt) * np.sin(around),
                                                   np.cos(tilt)])
        aim = board_in_base + np.append(rng.normal(0, 0.02, 2), 0)
        T_cam = look_at(eye, aim, roll=np.deg2rad(rng.uniform(-40, 40)) + np.pi / 2)

        # Ask for the flange pose that puts the CAMERA there, using the nominal mounting.
        T_flange = T_cam @ np.linalg.inv(tb.T_TOOL_CAM_NOMINAL.A)
        sol = tb.ur3.ikine_LM(tb.SE3(T_flange, check=False), q0=tb.Q_HOME, end="tool0")
        if not sol.success:
            continue

        sim.set_joints(sol.q)
        rgb, _ = sim.capture_rgbd()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        found, corners = cv2.findChessboardCorners(gray, tb.BOARD_CORNERS,
                                                   flags=cv2.CALIB_CB_ADAPTIVE_THRESH)
        if not found:
            continue
        # Refine to sub-pixel: the corner is where the image gradients say it is.
        corners = cv2.cornerSubPix(
            gray, corners, (5, 5), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4))

        # The arm is still, so average many encoder samples to beat the noise down.
        q_measured = np.mean([sim.read_encoders() for _ in range(100)], axis=0)
        views.append({"rgb": rgb, "corners": corners, "q": q_measured,
                      "true_cam": sim.true_camera_pose()})
    return views, sim


print("=" * 70)
print("1. CALIBRATING THE WRIST CAMERA")
print("=" * 70)
print("Photographing the checkerboard from assorted poses...")

if not hasattr(cv2, "calibrateHandEye"):
    raise RuntimeError("This OpenCV build lacks calibrateHandEye. Install project/requirements.txt in .venv.")
all_views, sim = collect_views(30)
if len(all_views) < 30:
    raise RuntimeError(f"Need 30 usable calibration views, collected {len(all_views)}")
# Fit and choose a solver on the first 24 views only. Hold the last six out.
views, held_out = all_views[:24], all_views[24:]
print(f"Usable views: {len(views)} training, {len(held_out)} held out")

# Where every corner sits on the board, in the board's own frame (z = 0).
object_points = np.zeros((tb.BOARD_CORNERS[0] * tb.BOARD_CORNERS[1], 3), np.float32)
object_points[:, :2] = np.mgrid[0:tb.BOARD_CORNERS[0],
                                0:tb.BOARD_CORNERS[1]].T.reshape(-1, 2) * tb.BOARD_SQUARE

# Intrinsics (Zhang's method, as in the Week 3 tutorial).
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    [object_points] * len(views), [v["corners"] for v in views],
    (640, 480), None, None, flags=cv2.CALIB_FIX_K3)

K_true = sim.camera_matrix()
print(f"\nReprojection error: {rms:.3f} pixels")
print("                fx        fy        cx        cy")
print(f"  calibrated  {K[0,0]:8.2f}  {K[1,1]:8.2f}  {K[0,2]:8.2f}  {K[1,2]:8.2f}")
print(f"  true        {K_true[0,0]:8.2f}  {K_true[1,1]:8.2f}  {K_true[0,2]:8.2f}  {K_true[1,2]:8.2f}")
print(f"  error       {K[0,0]-K_true[0,0]:8.2f}  {K[1,1]-K_true[1,1]:8.2f}  "
      f"{K[0,2]-K_true[0,2]:8.2f}  {K[1,2]-K_true[1,2]:8.2f}")
print(f"\nDistortion [k1 k2 p1 p2 k3]: {dist.ravel().round(5)}")
print("  MuJoCo renders a perfect pinhole, so near-zero is the right answer.")

# Hand-eye: the board never moves, so A X = X B over any two views. OpenCV solves for X.
R_flange, t_flange, R_board, t_board = [], [], [], []
for view, rvec, tvec in zip(views, rvecs, tvecs):
    T = tb.ur3.fkine(view["q"], end="tool0").A      # FK of the MEASURED angles
    R_flange.append(T[:3, :3])
    t_flange.append(T[:3, 3])
    R_board.append(cv2.Rodrigues(rvec)[0])
    t_board.append(tvec.ravel())

print("\nHand-eye calibration (three standard solvers):")
print("  method        position error   angle error   board scatter")
best = None
solver_results = []
for name, flag in [("Tsai", cv2.CALIB_HAND_EYE_TSAI),
                   ("Park", cv2.CALIB_HAND_EYE_PARK),
                   ("Daniilidis", cv2.CALIB_HAND_EYE_DANIILIDIS)]:
    R_cam, t_cam = cv2.calibrateHandEye(R_flange, t_flange, R_board, t_board, method=flag)
    T_tool_cam = np.eye(4)
    T_tool_cam[:3, :3], T_tool_cam[:3, 3] = R_cam, t_cam.ravel()

    # Error against the simulator's true mounting - for marking only.
    T_true = np.linalg.inv(tb.ur3.fkine(views[0]["q"], end="tool0").A) @ views[0]["true_cam"]
    delta = np.linalg.inv(T_true) @ T_tool_cam
    pos_err = 1e3 * np.linalg.norm(delta[:3, 3])
    ang_err = np.rad2deg(np.linalg.norm(tb.rotation_log(delta[:3, :3])))

    # No ground truth needed: every view should place the fixed board identically.
    positions = np.array([(np.eye(4) @ Rt).astype(float)[:3, 3] for Rt in
                          [np.block([[Rf, tf[:, None]], [np.zeros(3), 1]]) @ T_tool_cam
                           @ np.block([[Rb, tbv[:, None]], [np.zeros(3), 1]])
                           for Rf, tf, Rb, tbv in zip(R_flange, t_flange, R_board, t_board)]])
    scatter = 1e3 * np.linalg.norm(positions - positions.mean(axis=0), axis=1).mean()

    print(f"  {name:<12}  {pos_err:8.2f} mm   {ang_err:8.3f} deg   {scatter:7.3f} mm")
    if not np.all(np.isfinite(T_tool_cam)):
        continue
    solver_results.append({"method": name, "training_board_scatter_mm": scatter,
                           "reference_position_error_mm": pos_err,
                           "reference_angle_error_deg": ang_err})
    if best is None or scatter < best[0]:
        best = (scatter, name, T_tool_cam)

if best is None:
    raise RuntimeError("All hand-eye solvers failed")
_, chosen, T_tool_cam = best
print(f"\nUsing the {chosen} result (lowest board scatter).")
print("Camera position on the flange [m]:", T_tool_cam[:3, 3].round(4))
print("Nominal from the CAD model     [m]:", tb.T_TOOL_CAM_NOMINAL.t.round(4))

np.savez(HERE / "calibration.npz", K=K, dist=dist.ravel(), T_tool_cam=T_tool_cam)
print(f"\nSaved {HERE / 'calibration.npz'}")


# Validation poses and image corners never enter the fitted intrinsics or hand-eye solver.
validation_positions, validation_reprojection = [], []
for view in held_out:
    ok, rvec, tvec = cv2.solvePnP(object_points, view["corners"], K, dist)
    if not ok:
        raise RuntimeError("Held-out checkerboard pose could not be recovered")
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    residual = projected.reshape(-1, 2) - view["corners"].reshape(-1, 2)
    validation_reprojection.append(float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))))
    T_cam_board = np.eye(4)
    T_cam_board[:3, :3] = cv2.Rodrigues(rvec)[0]
    T_cam_board[:3, 3] = tvec.ravel()
    T_base_board = tb.ur3.fkine(view["q"], end="tool0").A @ T_tool_cam @ T_cam_board
    validation_positions.append(T_base_board[:3, 3])
validation_positions = np.array(validation_positions)
held_scatter = 1e3 * float(np.sqrt(np.mean(np.sum(
    (validation_positions - validation_positions.mean(axis=0))**2, axis=1))))
calibration_report = {
    "training_views": len(views), "validation_views": len(held_out),
    "training_reprojection_rms_px": float(rms),
    "validation_reprojection_rms_px": float(np.sqrt(np.mean(np.square(validation_reprojection)))),
    "validation_board_scatter_rms_mm": held_scatter,
    "selected_solver": chosen, "solver_results": solver_results,
    "selection": "minimum training board scatter; held-out views used for evaluation only",
    "distortion": dist.ravel().tolist(), "provenance": provenance()}
write_json(RESULTS / "calibration_report.json", calibration_report)
np.savez_compressed(RESULTS / "calibration_observations.npz",
                    corners=np.array([v["corners"] for v in all_views]),
                    measured_q=np.array([v["q"] for v in all_views]),
                    object_points=object_points, train_indices=np.arange(24),
                    validation_indices=np.arange(24, 30))
print(f"Held-out reprojection RMS: {calibration_report['validation_reprojection_rms_px']:.3f} px")
print(f"Held-out fixed-board scatter RMS: {held_scatter:.3f} mm")


print()
print("=" * 70)
print("2. MEASURING THE LOCALISATION ERROR")
print("=" * 70)
print("\nHolding the arm still and photographing the same block many times tells us two\ndifferent things, and the difference matters:\n\n  SPREAD  how much the answer jumps about between frames (random noise)\n  BIAS    how far the average answer sits from the truth (a systematic offset)\n\nA Kalman filter's R has to describe the TOTAL error, so we fit to the RMS error\nabout the true position, not to the spread.\n")

sim2 = tb.Sim(seed=7)
sim2.data.mocap_pos[0] = [0, 0, -1]                  # board out of the way
block_xy, block_yaw = np.array([-0.02, 0.03]), np.deg2rad(25.0)
sim2.set_block("target", block_xy[0], block_xy[1], block_yaw)

# True block pose in base frame; the long side is body y, so long-axis yaw is body yaw + 90 deg.
true_centre = (sim2.T_base_world @ np.append([block_xy[0], block_xy[1],
                                              tb.TABLE_Z + tb.BLOCK_HALF], 1))[:3]
true_pose = np.append(true_centre, tb.wrap_pi_2(block_yaw + np.pi / 2))

heights = [0.12, 0.16, 0.20, 0.24, 0.28, 0.32]
N_FRAMES = 40
rows = []
for h in heights:
    T_goal = np.eye(4)
    T_goal[:3, :3] = np.diag([1.0, -1.0, -1.0])      # straight down
    T_goal[:3, 3] = true_centre + [0, 0, h]
    q, ok = tb.ik_tcp(tb.SE3(T_goal, check=False), tb.Q_HOME)
    if not ok:
        continue
    sim2.set_joints(q)
    T_base_cam = tb.ur3.fkine(q, end="tool0").A @ T_tool_cam

    measurements, ranges = [], []
    for _ in range(N_FRAMES):
        rgb, depth = sim2.capture_rgbd()
        mask = tb.detect_red_block(rgb)
        got = tb.locate_block(mask, depth, K, T_base_cam, dist) if mask is not None else None
        if got is not None:
            measurements.append(got[0])
            ranges.append(got[1])
    if len(measurements) < 10:
        continue

    measurements = np.array(measurements)
    errors = measurements - true_pose
    errors[:, 3] = tb.wrap_pi_2(errors[:, 3])
    rows.append({
        "height": h,
        "range": float(np.mean(ranges)),
        "spread": measurements.std(axis=0, ddof=1),
        "bias": errors.mean(axis=0),
        "rms": np.sqrt(np.mean(errors ** 2, axis=0)),
    })

print(f"{N_FRAMES} frames at each height, same block each time.\n")
print("  camera   |------ spread (mm) ------|  |------- bias (mm) -------|  yaw err")
print("  range m     x       y       z           x       y       z         deg")
for r in rows:
    print(f"  {r['range']:6.3f}  {1e3*r['spread'][0]:7.2f} {1e3*r['spread'][1]:7.2f} "
          f"{1e3*r['spread'][2]:7.2f}    {1e3*r['bias'][0]:+7.2f} {1e3*r['bias'][1]:+7.2f} "
          f"{1e3*r['bias'][2]:+7.2f}   {np.rad2deg(r['rms'][3]):7.2f}")

print("\nThe z column is the interesting one. The spread is a few tenths of a\nmillimetre, but the bias is over a millimetre - the measurement is repeatable\nwithout being accurate. Two reasons, both of which a real RGB-D camera shares:\n\n  * the block comes out 1-2 mm taller than it is, because its top face and the\n    surrounding table are both measured from depth, and the smooth part of the\n    depth error does not cancel between them;\n  * the top face is taken as the 90th percentile of the blob's heights, which\n    sits a fraction above the true face.\n\nIf we fitted R to the spread alone, the filter would claim +-0.3 mm while\nactually being 1.5 mm out. Its gate would then start throwing away perfectly\ngood measurements. So we fit to the RMS error instead.\n")

# Fit sigma(Z) = a + b*Z^2: stereo depth error grows with the square of range.
Z = np.array([r["range"] for r in rows])
rms = np.array([r["rms"] for r in rows])
fit = np.linalg.lstsq(np.column_stack([np.ones_like(Z), Z ** 2]), rms, rcond=None)[0]
noise_a = np.maximum(fit[0], 1e-5)
noise_b = np.maximum(fit[1], 1e-5)

print("Fitted measurement noise,  sigma = a + b * range^2 :")
print("            a           b      sigma at 0.2 m   at 0.35 m")
for i, label in enumerate(["x  ", "y  ", "z  ", "yaw"]):
    scale, unit = (1e3, "mm") if i < 3 else (180 / np.pi, "deg")
    print(f"  {label}  {noise_a[i]:9.2e}  {noise_b[i]:9.2e}   "
          f"{scale*(noise_a[i]+noise_b[i]*0.04):8.2f} {unit}  "
          f"{scale*(noise_a[i]+noise_b[i]*0.1225):8.2f} {unit}")

np.save(HERE / "noise_model.npy", np.array([noise_a, noise_b]))
print(f"\nSaved {HERE / 'noise_model.npy'}  (this becomes R in the object filter)")
print("Because R depends on range, a frame taken from far away is automatically")
print("trusted less than a close one - which is why the estimate tightens up as")
print("the arm comes down.")


# What the detector sees, close up and far away.
fig, axes = plt.subplots(2, 2, figsize=(9, 6))
for col, h in enumerate([heights[0], heights[-1]]):
    T_goal = np.eye(4)
    T_goal[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T_goal[:3, 3] = true_centre + [0, 0, h]
    q, _ = tb.ik_tcp(tb.SE3(T_goal, check=False), tb.Q_HOME)
    sim2.set_joints(q)
    rgb, depth = sim2.capture_rgbd()
    mask = tb.detect_red_block(rgb)

    axes[0, col].imshow(rgb)
    axes[0, col].set_title(f"wrist camera, {h:.2f} m above the block")
    good = depth[np.isfinite(depth)]
    shown = np.ma.masked_invalid(depth)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("red")                     # dropped pixels stand out
    axes[1, col].imshow(shown, cmap=cmap,
                        vmin=np.percentile(good, 1), vmax=np.percentile(good, 99))
    if mask is not None:
        ys, xs = np.nonzero(mask)
        axes[1, col].plot(xs, ys, ",", color="white", alpha=0.4)
    axes[1, col].set_title("depth: dark = near, red = no reading\nwhite = detected block")
    for ax in axes[:, col]:
        ax.axis("off")
fig.suptitle("Colour threshold and depth segmentation, near and far "
             "(red = no depth reading)", fontweight="bold")
fig.tight_layout()
fig.savefig(RESULTS / "part_b_detection.png", dpi=150)

# Error against range, with the fitted curve.
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
Zfine = np.linspace(Z.min() * 0.9, Z.max() * 1.05, 100)
for ax, idx, title, scale, unit in [(axes[0], [0, 1], "sideways (x, y)", 1e3, "mm"),
                                    (axes[1], [2], "depth (z)", 1e3, "mm"),
                                    (axes[2], [3], "yaw", 180 / np.pi, "deg")]:
    for k, i in enumerate(idx):
        colour = f"C{k}"
        ax.plot(Z, scale * rms[:, i], "o", color=colour, label="total error")
        ax.plot(Z, scale * np.array([r["spread"][i] for r in rows]), "^", color=colour,
                alpha=0.5, label="spread only")
        ax.plot(Zfine, scale * (noise_a[i] + noise_b[i] * Zfine ** 2), "--", color=colour,
                label="fitted sigma")
    ax.set_xlabel("camera range [m]")
    ax.set_ylabel(f"error [{unit}]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
fig.suptitle("Localisation error grows with range - and is mostly bias, not noise",
             fontweight="bold")
fig.tight_layout()
fig.savefig(RESULTS / "part_b_noise.png", dpi=150)
print(f"Saved {RESULTS / 'part_b_detection.png'} and part_b_noise.png")

# Store the measurements used to fit R, rather than retaining only a plot.
write_json(RESULTS / "noise_report.json", {"samples_per_range": N_FRAMES,
           "range_results": rows, "noise_a": noise_a, "noise_b": noise_b,
           "bias_model": "bias is included in total RMS, not corrected away",
           "provenance": provenance()})
if sim.renderer is not None:
    sim.renderer.close()
if sim2.renderer is not None:
    sim2.renderer.close()
