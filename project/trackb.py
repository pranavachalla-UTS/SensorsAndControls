"""
Track B: vision-guided pick with a UR3.
Shared code used by all the part_*.py scripts.

The split between "the real robot" and "what the robot knows" matters here, so
it is worth stating once:

  * MuJoCo is the REAL robot.  It holds the true joint angles, the true object
    pose, and does the physics.  We are only allowed to touch it through the
    sensor functions below (read_encoders, capture_rgbd) and the torque command.

  * roboticstoolbox holds the robot's OWN MODEL of itself - the kinematics and
    inertias from the UR3 datasheet.  The controller and the observer only ever
    use this model and the sensor readings.

  * Ground truth (true_q, true_object_pose, ...) is used ONLY to plot errors and
    score trials.  Every such function has "true_" in its name so it is easy to
    check that none of them feed the controller.
"""

import secrets
import numpy as np
import mujoco
import cv2
import roboticstoolbox as rtb
from spatialmath import SE3
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCENE_XML = HERE / "ur3_pick_scene.xml"

# 9 x 6 inner corners of 20 mm squares, in a white border so OpenCV finds the outer edge.
BOARD_CORNERS = (9, 6)
BOARD_SQUARE = 0.02      # [m]


def _write_checkerboard_texture():
    """Draw the calibration board PNG if it is not there yet. The scene file
    needs it to exist before MuJoCo can load the model."""
    path = HERE / "checkerboard.png"
    if path.exists():
        return
    from PIL import Image
    squares_x, squares_y, border, px = 10, 7, 1, 100
    img = np.full(((squares_y + 2 * border) * px, (squares_x + 2 * border) * px), 255, np.uint8)
    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2 == 0:
                r0, c0 = (row + border) * px, (col + border) * px
                img[r0:r0 + px, c0:c0 + px] = 0
    Image.fromarray(np.dstack([img] * 3)).save(path)


_write_checkerboard_texture()

JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

# Elbow up, tool pointing straight down. Everything starts and ends here.
Q_HOME = np.deg2rad([0.0, -90.0, 90.0, -90.0, -90.0, 0.0])

TABLE_Z = 0.6        # height of the table top in world coordinates [m]
BLOCK_HALF = 0.02    # half-height of a block [m], so its centre sits 20 mm up

# Where the block may land, in world coordinates [m]. Redrawn every run.
BLOCK_X_RANGE = (-0.19, 0.09)
BLOCK_Y_RANGE = (-0.26, 0.26)

# The block travels across the table on a belt rather than sitting still, so
# the robot has to intercept a moving target instead of a parked one.
BLOCK_SPEED = 0.035            # [m/s] - about 50 cm of table in 14 s
BLOCK_HEADING_DEG = 12.0       # how far off the table's y axis the belt may run
BLOCK_START_Y = 0.24           # start this far upstream, to leave room to cross

DT = 0.002           # control period [s] -> 500 Hz, the UR3's real rate
CAM_PERIOD = 33      # run the camera every 33 control steps -> ~15 Hz

# UR3 encoders are 14-bit absolute (one count = 0.38 mrad), plus a little white noise.
ENCODER_COUNT = 2 * np.pi / 2 ** 14      # [rad]
ENCODER_NOISE = 5e-4                     # [rad]

# Stereo depth error grows with range^2, in three parts - only the first averages away.
DEPTH_NOISE = 0.010      # per-pixel random noise, as a fraction of Z^2
DEPTH_BIAS = 0.004       # smooth per-frame bias field, as a fraction of Z^2
DEPTH_DROPOUT = 0.03     # fraction of pixels that return nothing
RGB_NOISE = 4.0          # camera image noise [grey levels]


ur3 = rtb.models.UR3()

# From the scene file: TCP 105 mm out along tool z, camera 55 mm to the side, 44 mm out.
T_TOOL_TCP = SE3(0, 0, 0.105)
T_TOOL_CAM_NOMINAL = SE3(0.055, 0, 0.044)


def fk_tcp(q):
    """Pose of the TCP in the robot base frame, from joint angles q."""
    return ur3.fkine(q, end="tool0") * T_TOOL_TCP


def jacobian_tcp(q):
    """6x6 Jacobian of the TCP in the base frame: [v; omega] = J @ qdot."""
    return ur3.jacob0(q, end="tool0", tool=T_TOOL_TCP)


def ik_tcp(T_goal, q_guess):
    """Joint angles that put the TCP at T_goal.

    Seeded with q_guess so we stay on the same arm configuration (elbow up,
    wrist pointing down) instead of jumping to a mirrored solution.
    """
    sol = ur3.ikine_LM(T_goal, q0=q_guess, end="tool0", tool=T_TOOL_TCP)
    return sol.q, sol.success


class ControllerModel:
    """The rigid-body model the controller uses to work out torques.

    It is built from the same UR3 description as the simulated robot, but it
    is a SEPARATE copy with its own state, which we only ever feed estimated
    joint angles. That separation is the point: nothing in here can see the
    simulator's true state.

    What it deliberately does NOT model:
      * Coulomb friction in the joints
      * the weight of whatever the gripper is carrying
    Both are left for the observer's disturbance state to estimate.
    """

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self.qadr = np.array([self.model.jnt_qposadr[self.model.joint(j).id] for j in JOINTS])
        self.dadr = np.array([self.model.jnt_dofadr[self.model.joint(j).id] for j in JOINTS])
        # Friction is not part of the model the controller inverts.
        self.model.dof_frictionloss[:] = 0
        self.damping = self.model.dof_damping[self.dadr].copy()

    def dynamics(self, q, qd):
        """Inertia matrix M(q) and bias torque c(q,qd) + g(q), for the arm."""
        self.data.qpos[self.qadr] = q
        self.data.qvel[:] = 0
        self.data.qvel[self.dadr] = qd
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_makeM(self.model, self.data)
        mujoco.mj_comVel(self.model, self.data)

        bias = np.zeros(self.model.nv)
        mujoco.mj_rne(self.model, self.data, 0, bias)          # c(q,qd) + g(q)
        M_full = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, self.data, M_full)
        return M_full[np.ix_(self.dadr, self.dadr)], bias[self.dadr]


class Sim:
    """Thin wrapper around the MuJoCo model: the 'real' robot and the world."""

    def __init__(self, seed=None):
        """seed=None draws a fresh seed from the OS, so every run differs."""
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        # Index tables so we can read/write the arm joints quickly.
        self.qadr = np.array([self.model.jnt_qposadr[self.model.joint(j).id] for j in JOINTS])
        self.dadr = np.array([self.model.jnt_dofadr[self.model.joint(j).id] for j in JOINTS])
        self.torque_limit = self.model.actuator_ctrlrange[:6, 1].copy()

        # The base is bolted down, so its pose never changes: read it once.
        mujoco.mj_forward(self.model, self.data)
        base = self.model.body("base_link").id
        self.T_world_base = np.eye(4)
        self.T_world_base[:3, :3] = self.data.xmat[base].reshape(3, 3)
        self.T_world_base[:3, 3] = self.data.xpos[base]
        self.T_base_world = np.linalg.inv(self.T_world_base)

        self.renderer = None
        self._fx = None   # focal length in pixels, filled in on first capture
        self.block_velocity = None   # None = the belt is off and the block sits still

    # Setting the world up.

    def set_joints(self, q):
        self.data.qpos[self.qadr] = q
        mujoco.mj_forward(self.model, self.data)

    def set_block(self, name, x, y, yaw):
        """Place a block flat on the table at (x, y) turned by yaw."""
        adr = self.model.jnt_qposadr[self.model.joint(name + "_free").id]
        self.data.qpos[adr:adr + 3] = [x, y, TABLE_Z + BLOCK_HALF]
        self.data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    def randomise_blocks(self, speed=BLOCK_SPEED):
        """Put the red target block on the table and start the belt running.

        The box is the area that is BOTH reachable for a top-down grasp at any
        block angle and visible from one of the arm's look poses (see
        OBSERVE_SWEEP_DEG) - measured, not guessed. The block often lands
        outside the first look's field of view, which is what the sweep is for.

        With the belt on the block starts at the upstream edge of that box
        rather than anywhere in it, so it has the full width of the table to
        cross before it runs out of room. Which way it runs, and the few
        degrees it runs off-axis, are redrawn every time.

        speed=0 parks it, which is how Part B and the open-loop comparison
        want the world.
        """
        heading = np.deg2rad(self.rng.uniform(-BLOCK_HEADING_DEG, BLOCK_HEADING_DEG))
        downstream = 1.0 if self.rng.random() < 0.5 else -1.0

        if speed > 0:
            # Keep it to the middle half of the x range: the few degrees of
            # off-axis drift must not walk it out of the reachable strip
            # before the robot gets there.
            lo, hi = BLOCK_X_RANGE
            x = self.rng.uniform(lo + 0.25 * (hi - lo), hi - 0.25 * (hi - lo))
            y = -downstream * BLOCK_START_Y
            self.block_velocity = speed * np.array([np.sin(heading),
                                                    downstream * np.cos(heading)])
        else:
            x = self.rng.uniform(*BLOCK_X_RANGE)
            y = self.rng.uniform(*BLOCK_Y_RANGE)
            self.block_velocity = None

        self.set_block("target", x, y, self.rng.uniform(-np.pi / 2, np.pi / 2))
        return x, y

    def drive_block(self):
        """Hold the block at belt speed for this control step.

        Written as a velocity, not a position, so the block still rests on the
        table under gravity and can still be gripped - the belt only insists on
        how fast it travels, contact physics does everything else. Spin is held
        at zero because a block on a belt slides, it does not tumble.
        """
        if self.block_velocity is None:
            return
        adr = self.model.jnt_dofadr[self.model.joint("target_free").id]
        self.data.qvel[adr:adr + 2] = self.block_velocity
        self.data.qvel[adr + 3:adr + 6] = 0.0

    def release_block(self):
        """Take the block off the belt - called the moment the fingers close."""
        self.block_velocity = None

    def step(self, torque, gripper):
        """Apply a joint torque and a gripper command for one control period."""
        self.data.ctrl[:6] = torque
        self.data.ctrl[6] = gripper
        for _ in range(int(round(DT / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)

    # Sensors: the only legitimate way to read the robot.

    def read_encoders(self):
        """Joint position measurement: true angle + noise, then quantised to
        the encoder's resolution. Note there is NO velocity output - that is
        what the observer is for."""
        noisy = self.true_q() + self.rng.normal(0, ENCODER_NOISE, 6)
        return np.round(noisy / ENCODER_COUNT) * ENCODER_COUNT

    def capture_rgbd(self):
        """Render one RGB-D frame from the wrist camera and add sensor noise.

        Depth error is modelled in three parts because only the first averages
        away over a blob of pixels:
          1. per-pixel random noise
          2. a smooth bias field (neighbouring pixels share nearly the same
             error - this is what really limits the accuracy)
          3. missing pixels
        """
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, 480, 640)
            fovy = np.deg2rad(self.model.camera("wrist_camera").fovy[0])
            self._fx = 480 / 2 / np.tan(fovy / 2)

        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera="wrist_camera")
        rgb = self.renderer.render().astype(float)

        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera="wrist_camera")
        depth = self.renderer.render().astype(float)
        self.renderer.disable_depth_rendering()

        rgb = np.clip(rgb + self.rng.normal(0, RGB_NOISE, rgb.shape), 0, 255).astype(np.uint8)
        depth = depth + self.rng.normal(0, 1, depth.shape) * DEPTH_NOISE * depth ** 2
        depth = depth + smooth_noise_field(depth.shape, self.rng) * DEPTH_BIAS * depth ** 2
        depth[self.rng.random(depth.shape) < DEPTH_DROPOUT] = np.nan
        return rgb, depth

    def camera_matrix(self):
        """The true pinhole intrinsics, from the camera's field of view.
        Part B recovers these from checkerboard images instead."""
        fovy = np.deg2rad(self.model.camera("wrist_camera").fovy[0])
        f = 480 / 2 / np.tan(fovy / 2)
        return np.array([[f, 0, 319.5], [0, f, 239.5], [0, 0, 1.0]])

    # Ground truth: for plotting and scoring only.

    def true_q(self):
        return self.data.qpos[self.qadr].copy()

    def true_qd(self):
        return self.data.qvel[self.dadr].copy()

    def true_tcp(self):
        """TCP position in the base frame."""
        p = self.data.site_xpos[self.model.site("tcp").id]
        return (self.T_base_world @ np.append(p, 1))[:3]

    def true_object_pose(self):
        """Target block as [x, y, z, yaw] in the base frame.

        yaw is the direction of the block's LONG side. The 50 mm side is the
        block's own y axis, so we read column 1 of its rotation matrix.
        """
        body = self.model.body("target").id
        p = (self.T_base_world @ np.append(self.data.xpos[body], 1))[:3]
        R = self.T_base_world[:3, :3] @ self.data.xmat[body].reshape(3, 3)
        long_axis = R[:, 1]
        return np.append(p, wrap_pi_2(np.arctan2(long_axis[1], long_axis[0])))

    def true_object_velocity(self):
        """Target block's linear velocity in the base frame [m/s].

        Only ever used to score the filter: nothing in the robot measures this,
        the filter has to work it out from how the pose moves.
        """
        adr = self.model.jnt_dofadr[self.model.joint("target_free").id]
        v_world = self.data.qvel[adr:adr + 3]
        return self.T_base_world[:3, :3] @ v_world

    def true_camera_pose(self):
        """Camera pose in the base frame, in the OPTICAL convention
        (x right, y down, z forward). Used only to check the calibration."""
        cid = self.model.camera("wrist_camera").id
        T = np.eye(4)
        T[:3, :3] = self.data.cam_xmat[cid].reshape(3, 3) @ MUJOCO_TO_OPTICAL
        T[:3, 3] = self.data.cam_xpos[cid]
        return self.T_base_world @ T


# MuJoCo cameras look along -z with y up; OpenCV wants x right, y down, z forward.
MUJOCO_TO_OPTICAL = np.diag([1.0, -1.0, -1.0])


def smooth_noise_field(shape, rng, cells=4):
    """A low-resolution random grid stretched smoothly to full size.

    Used for the depth bias: neighbouring pixels get almost the same error, so
    averaging over a blob does NOT cancel it, which is how real stereo cameras
    behave.
    """
    h, w = shape
    coarse = rng.normal(0, 1, (cells, cells))
    yi = np.linspace(0, cells - 1, h)
    xi = np.linspace(0, cells - 1, w)
    y0 = np.clip(np.floor(yi).astype(int), 0, cells - 2)
    x0 = np.clip(np.floor(xi).astype(int), 0, cells - 2)
    ty, tx = (yi - y0)[:, None], (xi - x0)[None, :]
    ty, tx = ty * ty * (3 - 2 * ty), tx * tx * (3 - 2 * tx)   # smoothstep
    top = coarse[y0][:, x0] * (1 - tx) + coarse[y0][:, x0 + 1] * tx
    bot = coarse[y0 + 1][:, x0] * (1 - tx) + coarse[y0 + 1][:, x0 + 1] * tx
    return top * (1 - ty) + bot * ty


def wrap_pi_2(angle):
    """Wrap to [-pi/2, pi/2).

    A parallel gripper is the same after turning 180 degrees, so a grasp angle
    only ever means anything modulo pi.
    """
    return (angle + np.pi / 2) % np.pi - np.pi / 2


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotation_log(R):
    """Rotation matrix -> rotation vector (axis * angle). The rotational half
    of a pose error."""
    cos_angle = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return angle * w / (2 * np.sin(angle))


def pose_error(T_now, T_goal):
    """6-vector [dx, dy, dz, rotation vector] that moves T_now towards T_goal.
    Both are 4x4 matrices in the same frame."""
    return np.concatenate([T_goal[:3, 3] - T_now[:3, 3],
                           rotation_log(T_goal[:3, :3] @ T_now[:3, :3].T)])


def detect_red_block(rgb):
    """Return a boolean mask of the biggest red blob, or None.

    HSV rather than raw RGB, because hue separates "is it red" from "how
    bright is it", which keeps the threshold stable as the arm shadows the
    table. The grey table and the arm fall outside the hue window.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (((h < 10) | (h > 170)) & (s > 120) & (v > 50)).astype(np.uint8)

    # Open then close: drop stray single pixels, then fill small holes.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[biggest, cv2.CC_STAT_AREA] < 120:
        return None

    # A blob touching the image edge is only part of the block, so its centre is wrong.
    x, y, w, hh = stats[biggest, :4]
    if x <= 1 or y <= 1 or x + w >= mask.shape[1] - 1 or y + hh >= mask.shape[0] - 1:
        return None
    return labels == biggest


def locate_block(mask, depth, K, T_base_cam):
    """Turn a detected blob into [x, y, z, yaw] in the robot base frame.

    Steps:
      1. back-project every masked pixel using its own depth -> 3D points
      2. move those points into the base frame
      3. the top face is the highest few mm of points; its mean gives x, y
         and the direction it is longest in gives yaw
      4. a ring of pixels just outside the blob is the table, so
         centre height = table height + block height / 2

    Returns (z, camera_range) or None, where z = [x, y, z, yaw].
    """
    valid = mask & np.isfinite(depth)
    v, u = np.nonzero(valid)
    if len(u) < 40:
        return None

    pts = backproject(u, v, depth[v, u], K)
    pts = pts @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]

    # 90th percentile, not the maximum, so a few bad depth pixels cannot decide the top.
    top_z = np.percentile(pts[:, 2], 90)
    top = pts[pts[:, 2] > top_z - 0.006]
    if len(top) < 20:
        return None
    xy = top[:, :2].mean(axis=0)

    # Direction of greatest spread of the top face = the block's long side.
    evals, evecs = np.linalg.eigh(np.cov((top[:, :2] - xy).T))
    yaw = wrap_pi_2(np.arctan2(evecs[1, 1], evecs[0, 1]))

    # Table height from a ring of pixels around the block.
    m8 = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    ring = (cv2.dilate(m8, kernel, iterations=25).astype(bool)
            & ~cv2.dilate(m8, kernel, iterations=8).astype(bool)
            & np.isfinite(depth))
    rv, ru = np.nonzero(ring)
    if len(ru) < 200:
        return None
    ring_pts = backproject(ru, rv, depth[rv, ru], K)
    ring_pts = ring_pts @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]
    table_z = np.median(ring_pts[:, 2])

    height = top_z - table_z
    if not 0.01 < height < 0.12:          # not a sensible block height
        return None

    return np.array([xy[0], xy[1], table_z + height / 2, yaw]), float(np.median(depth[v, u]))


def backproject(u, v, Z, K):
    """Pixels + depth -> 3D points in the camera frame (the pinhole model
    run backwards)."""
    x = (u - K[0, 2]) * Z / K[0, 0]
    y = (v - K[1, 2]) * Z / K[1, 1]
    return np.column_stack([x, y, Z])


class JointObserver:
    """One Kalman filter per joint, on the state [q, qdot, d].

    Why three states and not two:
      * qdot is never measured, so it has to be estimated;
      * d is an unknown acceleration disturbance. It mops up everything the
        controller's model gets wrong - Coulomb friction, the gripper's weight,
        and later the weight of the block being carried. The controller
        subtracts the estimate, which removes steady-state error without
        putting an integrator in the loop.

    All six joints are filtered at once with array maths, because after the
    computed-torque step below they are six identical, independent systems.
    """

    def __init__(self, sigma_a=2.0, sigma_d=60.0):
        # Constant acceleration: q += qdot*dt + (u+d)*dt^2/2, qdot += (u+d)*dt, d random-walks.
        self.F = np.array([[1, DT, DT ** 2 / 2],
                           [0, 1, DT],
                           [0, 0, 1]])
        self.G = np.array([DT ** 2 / 2, DT, 0.0])        # how u enters

        # sigma_a: unmodelled acceleration. sigma_d: how fast d may move. Swept in part_c.
        self.Q = sigma_a ** 2 * np.outer(self.G, self.G)
        self.Q[2, 2] += sigma_d ** 2 * DT

        # White noise plus the encoder's rounding variance, count^2/12 for a uniform.
        self.R = ENCODER_NOISE ** 2 + ENCODER_COUNT ** 2 / 12

        self.x = np.zeros((6, 3))
        self.P = np.zeros((6, 3, 3))

    def start(self, q_measured):
        self.x[:] = 0
        self.x[:, 0] = q_measured
        # Start confident in position (we just measured it), unsure of the rest.
        self.P[:] = np.diag([self.R, 0.05 ** 2, 5.0 ** 2])

    def predict(self, u):
        """u = the acceleration we actually commanded, per joint."""
        self.x = self.x @ self.F.T + np.outer(u, self.G)
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, q_measured):
        innovation = q_measured - self.x[:, 0]           # measured - predicted
        S = self.P[:, 0, 0] + self.R                     # innovation variance
        K = self.P[:, :, 0] / S[:, None]                 # Kalman gain, per joint
        self.x += K * innovation[:, None]
        # Joseph form: same as (I-KH)P but stays symmetric and positive definite.
        I_KH = np.eye(3)[None] - K[:, :, None] * np.eye(3)[0][None, None, :]
        self.P = I_KH @ self.P @ I_KH.transpose(0, 2, 1) + self.R * K[:, :, None] * K[:, None, :]

    # Convenience views of the state
    @property
    def q(self):
        return self.x[:, 0].copy()

    @property
    def qd(self):
        return self.x[:, 1].copy()

    @property
    def d(self):
        return self.x[:, 2].copy()

    def sigma(self):
        """Standard deviation of each state, per joint. Shape (6, 3)."""
        return np.sqrt(np.diagonal(self.P, axis1=1, axis2=2))

    def nees(self, q_true, qd_true):
        """Normalised estimation error squared over [q, qdot], per joint.

        This is the consistency check: if the filter's covariance is honest,
        the average comes out near 2 (the number of states being checked).
        Much bigger means the filter is over-confident; much smaller means it
        is being too cautious.
        """
        e = np.column_stack([q_true - self.q, qd_true - self.qd])
        P2 = self.P[:, :2, :2]
        return np.einsum("ni,nij,nj->n", e, np.linalg.inv(P2), e)


# Process noise for the block filter: the white-noise acceleration that stands
# in for everything a constant-velocity model leaves out - the belt changing
# speed, a block that catches and skews, the shove Part E gives it.
# Units are (m/s^2)^2 per Hz, and (rad/s^2)^2 per Hz for the yaw term.
BLOCK_ACCEL_PSD = np.array([0.05, 0.05, 0.01, 0.40]) ** 2

# How unsure we are of the block's velocity the moment we first see it. One
# frame tells us where it is but nothing at all about how fast it is going.
BLOCK_VEL_PRIOR = np.array([0.15, 0.15, 0.05, 1.0])


class ObjectFilter:
    """Kalman filter on the block's pose AND velocity in the base frame.

    State is [x, y, z, yaw, vx, vy, vz, vyaw]. The block rides a belt, so a
    filter that assumed it sat still would sit a fixed distance behind it -
    roughly speed/gain, about 15 mm at belt speed - and the gripper would close
    on where the block used to be. Carrying a velocity fixes that twice over:
    the estimate keeps up between frames, and the controller gets something to
    feed forward (see run_pick).

    The camera measures the POSE ONLY. Nothing ever measures velocity: the
    filter infers it from the way the pose moves, which is the whole reason
    this is a Kalman filter and not a smoother.

    Between frames the block is taken to travel in a straight line at constant
    speed,

        x(k+1) = F x(k),    F = [[I, dt*I], [0, I]]

    with Q the usual white-noise-acceleration covariance, so the filter can
    still follow a belt that speeds up or a block that gets knocked.
    """

    N = 4            # pose components: x, y, z, yaw

    def __init__(self, noise_a, noise_b):
        # Measurement sigma at range Z is a + b*Z^2, fitted by part_b_sensing.py.
        self.noise_a = np.asarray(noise_a)
        self.noise_b = np.asarray(noise_b)
        self.x = None
        self.P = None
        self.n_used = 0
        self.n_rejected = 0
        self.n_resets = 0
        self._bad_run = 0

    # The camera sees the pose; H picks it out of the state.
    @property
    def pose(self):
        return self.x[:self.N]

    @property
    def velocity(self):
        return self.x[self.N:]

    def pose_at(self, lead):
        """Where the block will be `lead` seconds from now, on present evidence."""
        return self.x[:self.N] + lead * self.x[self.N:]

    def R_at(self, rng_m):
        """Measurement covariance for a frame taken at range rng_m."""
        return np.diag((self.noise_a + self.noise_b * rng_m ** 2) ** 2)

    def _Q(self, dt):
        """White-noise-acceleration process covariance over dt.

        Per axis this is the textbook  q * [[dt^3/3, dt^2/2], [dt^2/2, dt]],
        laid out to match the [pose, velocity] ordering of the state.
        """
        n, Q = self.N, np.zeros((2 * self.N, 2 * self.N))
        idx = np.arange(n)
        Q[idx, idx] = BLOCK_ACCEL_PSD * dt ** 3 / 3
        Q[idx, idx + n] = Q[idx + n, idx] = BLOCK_ACCEL_PSD * dt ** 2 / 2
        Q[idx + n, idx + n] = BLOCK_ACCEL_PSD * dt
        return Q

    def start(self, z, rng_m, keep_velocity=False):
        """Begin (or restart) from a single pose measurement.

        A restart means the block is somewhere we did not expect. The belt is
        still running, so the velocity estimate is worth keeping - but it is
        now much less trustworthy, hence the widened covariance.
        """
        v = self.velocity.copy() if (keep_velocity and self.x is not None) else np.zeros(self.N)
        self.x = np.concatenate([z, v])
        self.P = np.zeros((2 * self.N, 2 * self.N))
        self.P[:self.N, :self.N] = self.R_at(rng_m)
        self.P[self.N:, self.N:] = np.diag(BLOCK_VEL_PRIOR ** 2)
        self._bad_run = 0
        self.n_used += 1

    def predict(self, dt=DT):
        """Coast the estimate forward one control period."""
        if self.x is None:
            return
        n = self.N
        F = np.eye(2 * n)
        F[:n, n:] = dt * np.eye(n)
        self.x = F @ self.x
        self.x[3] = wrap_pi_2(self.x[3])
        self.P = F @ self.P @ F.T + self._Q(dt)

    def update(self, z, rng_m):
        """Fuse one pose measurement. Returns what happened, as a string."""
        if self.x is None:
            self.start(z, rng_m)
            return "first"

        n = self.N
        R = self.R_at(rng_m)
        y = z - self.x[:n]                       # innovation, pose only
        y[3] = wrap_pi_2(y[3])
        Ppp = self.P[:n, :n]                     # pose block of the covariance
        S = Ppp + R

        # Gate: a wildly inconsistent measurement is probably a bad detection, so drop it.
        distance = float(y @ np.linalg.solve(S, y))
        if distance > 18.5:                      # chi-squared, 4 dof, 99.9%
            self._bad_run += 1
            self.n_rejected += 1
            # ...unless several in a row disagree, in which case the block really has moved.
            if self._bad_run >= 4:
                self.start(z, rng_m, keep_velocity=True)
                self.n_resets += 1
                return "reset"
            return "rejected"

        # K = P H' S^-1, and H just selects the pose, so H P' is the first n columns of P.
        K = self.P[:, :n] @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[3] = wrap_pi_2(self.x[3])
        I_KH = np.eye(2 * n)
        I_KH[:, :n] -= K                         # (I - K H)
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self._bad_run = 0
        self.n_used += 1
        return "used"

    def sigma(self):
        """Standard deviations of the whole state: 4 pose, then 4 velocity."""
        return np.sqrt(np.diag(self.P))


# LQR weights by Bryson's rule: weight each term by 1/(largest tolerable value)^2.
MAX_POS_ERROR = 0.005      # [rad]
MAX_VEL_ERROR = 0.125      # [rad/s]
MAX_ACCEL = 5.0            # [rad/s^2]


def design_lqr():
    """Gains for one joint, from a discrete LQR on a double integrator.

    After the computed-torque step in joint_torque() every joint behaves like
    qddot = v, so one small design serves all six.

    Returns (kp, kd, closed-loop poles).
    """
    import control as ct

    # Double integrator, discretised exactly over one control period.
    A = np.array([[1.0, DT], [0.0, 1.0]])
    B = np.array([[DT ** 2 / 2], [DT]])
    Q = np.diag([1 / MAX_POS_ERROR ** 2, 1 / MAX_VEL_ERROR ** 2])
    R = np.array([[1 / MAX_ACCEL ** 2]])

    K, _, _ = ct.dlqr(A, B, Q, R)
    poles = np.linalg.eigvals(A - B @ K)
    return K[0, 0], K[0, 1], poles


def joint_torque(model, q_hat, qd_hat, d_hat, q_ref, qd_ref, qdd_ref, kp, kd, torque_limit):
    """Computed torque + LQR. Returns (torque, acceleration actually applied).

    Two nested ideas:

      1. Feedback linearisation. The arm obeys  M(q) qddot + c + g = tau.
         If we ask for  tau = M(q)*v + c + g  then it obeys  qddot = v.
         A coupled nonlinear system becomes six independent double integrators.

      2. On top of that, a simple LQR picks v:
             v = qdd_ref - kp*(position error) - kd*(velocity error) - d_hat
         The last term cancels the disturbance the observer found, which is
         what removes steady-state droop without an integrator.

    Everything fed in here is an ESTIMATE or a reference - never ground truth.
    """
    v = qdd_ref - kp * (q_hat - q_ref) - kd * (qd_hat - qd_ref) - d_hat

    M, bias = model.dynamics(q_hat, qd_hat)
    # Viscous damping only; Coulomb friction is left out on purpose for the observer to find.
    tau = M @ v + bias + model.damping * qd_hat
    tau_clipped = np.clip(tau, -torque_limit, torque_limit)

    # On a torque limit, tell the observer what we actually applied, not what we asked for.
    v_applied = np.linalg.solve(M, tau_clipped - bias - model.damping * qd_hat)
    return tau_clipped, v_applied


def quintic(q_start, q_end, duration, t):
    """Smooth point-to-point joint move, at rest at both ends.

    A quintic has zero velocity AND zero acceleration at both ends, so the
    reference we hand the controller is smooth enough to differentiate twice.
    Returns (q_ref, qd_ref, qdd_ref).
    """
    s = np.clip(t / duration, 0, 1)
    pos = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    vel = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / duration
    acc = (60 * s - 180 * s ** 2 + 120 * s ** 3) / duration ** 2
    delta = q_end - q_start
    return q_start + pos * delta, vel * delta, acc * delta


def grasp_pose(obj_xyz, obj_yaw, current_tool_x, height_above=0.0):
    """Where the TCP has to be to grasp the block, as a 4x4 matrix.

    The fingers close along the tool's y axis, so the tool's x axis must line
    up with the block's LONG side. Turning the wrist by 180 degrees gives an
    equally good grasp, so we pick whichever of the two is closer to where the
    wrist already is and save ourselves half a turn.
    """
    tool_down = np.diag([1.0, -1.0, -1.0])        # z down, x along base x
    now = np.arctan2(current_tool_x[1], current_tool_x[0])
    options = np.array([obj_yaw, obj_yaw + np.pi])
    best = options[np.argmin(np.abs(np.angle(np.exp(1j * (options - now)))))]

    T = np.eye(4)
    T[:3, :3] = rot_z(best) @ tool_down
    T[:3, 3] = np.asarray(obj_xyz) + [0, 0, height_above]
    return T


def pbvs_velocity(T_now, T_goal, gain, max_linear, max_angular):
    """Position-based visual servoing: turn a pose error into a joint velocity
    command. Returns (joint velocity, pose error).

    The goal comes from the object filter, which the camera updates every
    frame, so the loop really is closed through vision.

    v = gain * pose error, then capped so the arm moves at a sane speed, then
    mapped to joint velocities with the Jacobian.
    """
    e = pose_error(T_now, T_goal)
    velocity = gain * e

    linear = np.linalg.norm(velocity[:3])
    angular = np.linalg.norm(velocity[3:])
    if linear > max_linear:
        velocity[:3] *= max_linear / linear
    if angular > max_angular:
        velocity[3:] *= max_angular / angular
    return velocity, e


def joint_velocity_from_twist(J, twist, damping=0.01):
    """Solve J @ qdot = twist for qdot.

    Damped least squares rather than a plain inverse: near a singularity the
    plain inverse asks for enormous joint speeds, while this trades a little
    accuracy for a bounded answer.
    """
    return J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), twist)


# Fallback object-filter noise (a, b) in a + b*Z^2; part_b_sensing.py measures it properly.
DEFAULT_NOISE_A = [0.0002, 0.0003, 0.0011, 0.0004]
DEFAULT_NOISE_B = [0.0004, 0.0001, 0.0090, 0.0420]

# First look pose, as a TCP position in WORLD coordinates: high enough to see a good patch.
OBSERVE_TCP_WORLD = np.array([-0.02, 0.0, 0.93])

# One pose sees only ~35 x 34 cm, so sweep these pan angles [deg] until the block shows up.
OBSERVE_SWEEP_DEG = (0.0, 32.0, -32.0)
LOOK_WINDOW = 0.6       # how long to hold still and look at one pose [s]

HOVER_HEIGHT = 0.12     # stop this far above the block before descending [m]
GRASP_HEIGHT = 0.010    # final TCP height above the block's centre [m]
GRASP_TIME = 0.6        # how long the fingers get to close [s]
LIFT_HEIGHT = 0.15      # how far to lift once holding it [m]
GRIPPER_OPEN = 0.04
GRIPPER_SHUT = 0.0

PBVS_GAIN = 2.5         # [1/s]
MAX_LINEAR = 0.15       # [m/s] while approaching
DESCEND_LINEAR = 0.05   # [m/s] slower for the final descent
MAX_ANGULAR = 1.0       # [rad/s]

TOL_POS = 0.003         # how close is "arrived" [m]
TOL_ANG = np.deg2rad(2.0)
SETTLE_TIME = 0.15      # stay inside tolerance this long before moving on [s]


def load_noise_model():
    """Measurement noise fitted by part_b_sensing.py, or the fallback."""
    path = HERE / "noise_model.npy"
    if path.exists():
        data = np.load(path)
        return data[0], data[1]
    return np.array(DEFAULT_NOISE_A), np.array(DEFAULT_NOISE_B)


def load_calibration():
    """Camera calibration from part_b_calibration.py, or the nominal values."""
    path = HERE / "calibration.npz"
    if path.exists():
        data = np.load(path)
        return data["K"], data["dist"], data["T_tool_cam"]
    sim = Sim()
    return sim.camera_matrix(), np.zeros(5), np.array(T_TOOL_CAM_NOMINAL.A)


def run_pick(seed=None, noise_scale=1.0, payload=0.0, nudge=None, nudge_time=1.2,
             push=None, push_time=1.0, open_loop=False, record=None, max_time=30.0,
             observer_tuning=None, use_d_feedforward=True, block_speed=BLOCK_SPEED,
             use_v_feedforward=True):
    """Run one complete pick and return (success, info, log).

    seed fixes the block placement and the sensor noise; seed=None (the
    default) gives a fresh random block position on every run.

    Options are the disturbances used in Part E:
      noise_scale  multiply all sensor noise
      payload      extra mass on the gripper the controller does not know about [kg]
      nudge        (dx, dy) the block is shoved by, part way through the approach
      push         (fx, fy, fz) force applied to the forearm for 0.3 s
      open_loop    look once, then stop using the camera (to show what vision buys)
      record       a Recorder object for the demo video, or None
      observer_tuning   (sigma_a, sigma_d) to override the observer's defaults
      use_d_feedforward whether the controller cancels the estimated disturbance
      block_speed  how fast the belt carries the block [m/s]; 0 parks it
      use_v_feedforward whether the arm feeds the block's estimated velocity
                   forward, or only chases its position
    """
    global ENCODER_NOISE, DEPTH_NOISE, DEPTH_BIAS, RGB_NOISE
    base_noise = (ENCODER_NOISE, DEPTH_NOISE, DEPTH_BIAS, RGB_NOISE)
    ENCODER_NOISE, DEPTH_NOISE, DEPTH_BIAS, RGB_NOISE = [n * noise_scale for n in base_noise]

    try:
        if seed is None:
            seed = secrets.randbelow(100000)   # reported in info, so a run can be repeated
        sim = Sim(seed)
        if payload > 0:
            sim.model.body_mass[sim.model.body("gripper").id] += payload
        block_xy = sim.randomise_blocks(block_speed)
        sim.set_joints(Q_HOME)
        for j in ["finger_left_joint", "finger_right_joint"]:
            sim.data.qpos[sim.model.jnt_qposadr[sim.model.joint(j).id]] = GRIPPER_OPEN
        mujoco.mj_forward(sim.model, sim.data)

        K, dist, T_tool_cam = load_calibration()
        noise_a, noise_b = load_noise_model()
        kp, kd, _ = design_lqr()

        model = ControllerModel()
        observer = (JointObserver(*observer_tuning) if observer_tuning
                    else JointObserver())
        observer.start(sim.read_encoders())
        objects = ObjectFilter(noise_a, noise_b)

        # One observation pose per sweep angle, camera pointing straight down.
        observe_base = (sim.T_base_world @ np.append(OBSERVE_TCP_WORLD, 1))[:3]
        q_observe = []
        for angle in np.deg2rad(OBSERVE_SWEEP_DEG):
            c, s_ = np.cos(angle), np.sin(angle)
            Rz = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
            T_observe = np.eye(4)
            T_observe[:3, :3] = Rz @ np.diag([1.0, -1.0, -1.0])
            T_observe[:3, 3] = Rz @ observe_base
            q_look, ok = ik_tcp(SE3(T_observe, check=False), Q_HOME)
            if not ok:
                raise RuntimeError("cannot reach the observation pose")
            q_observe.append(q_look)

        log = {k: [] for k in ["t", "phase", "q", "qd", "q_hat", "qd_hat", "d_hat",
                               "q_sigma", "qd_sigma", "q_meas", "q_ref", "tau",
                               "obj_est", "obj_sigma", "obj_true", "obj_vel_true",
                               "tcp", "nees", "cam"]}

        phase, t, t_phase, step = "settle", 0.0, 0.0, 0
        scan = 0                 # which observation pose we are looking from
        q_ref, qd_ref, qdd_ref = Q_HOME.copy(), np.zeros(6), np.zeros(6)
        gripper, goal, inside_since = GRIPPER_OPEN, None, None
        traj_start, nudged, reason = Q_HOME.copy(), False, "ran out of time"
        grasp_info, pbvs_start = None, None

        while t < max_time and phase != "done":
            # Sense.
            q_meas = sim.read_encoders()
            observer.update(q_meas)
            q_hat, qd_hat, d_hat = observer.q, observer.qd, observer.d
            q_true, qd_true = sim.true_q(), sim.true_qd()        # for the log only

            cam_result = 0          # 0 none, 1 no detection, 2 rejected, 3 used
            looking = phase in ("look", "align", "descend") and not (open_loop and phase != "look")
            if looking and step % CAM_PERIOD == 0:
                rgb, depth = sim.capture_rgbd()
                # Camera pose from the ESTIMATED angles and the calibration, not the simulator.
                T_base_cam = ur3.fkine(q_hat, end="tool0").A @ T_tool_cam
                mask = detect_red_block(rgb)
                found = locate_block(mask, depth, K, T_base_cam) if mask is not None else None
                if found is None:
                    cam_result = 1
                else:
                    status = objects.update(found[0], found[1])
                    cam_result = 2 if status == "rejected" else 3
                if record is not None:
                    record.last_view = (rgb, mask)
            objects.predict()

            # Decide what to do next.
            t_phase += DT
            if phase == "settle":
                # Let the controller take hold of the arm before moving it.
                if t_phase > 0.4:
                    phase, t_phase, traj_start = "to_observe", 0.0, q_hat.copy()

            elif phase == "to_observe":
                q_ref, qd_ref, qdd_ref = quintic(traj_start, q_observe[scan], 2.0, t_phase)
                if t_phase > 2.2:
                    phase, t_phase = "look", 0.0

            elif phase == "look":
                # Hold still and collect a few frames before committing.
                if objects.x is not None and objects.n_used >= 3:
                    phase, t_phase, pbvs_start = "align", 0.0, t
                elif t_phase > LOOK_WINDOW:
                    # Nothing red in view: pan to the next patch of table.
                    scan += 1
                    if scan >= len(q_observe):
                        reason = "never found the block"
                        break
                    phase, t_phase, traj_start = "to_observe", 0.0, q_hat.copy()

            elif phase in ("align", "descend", "grasp", "lift"):
                if phase == "align":
                    # Hover above the block, lined up with it.
                    goal = grasp_pose(objects.x[:3], objects.x[3],
                                      fk_tcp(q_hat).R[:, 0], HOVER_HEIGHT)
                    speed, timeout = MAX_LINEAR, 6.0
                elif phase == "descend":
                    goal = grasp_pose(objects.x[:3], objects.x[3],
                                      fk_tcp(q_hat).R[:, 0], GRASP_HEIGHT)
                    speed, timeout = DESCEND_LINEAR, 5.0
                elif phase == "grasp":
                    # The block is still on the belt while the fingers close,
                    # so "hold still" means keep travelling with it. Stopping
                    # dead here would let it slide straight out of the jaws.
                    goal = grasp_pose(objects.x[:3], objects.x[3],
                                      fk_tcp(q_hat).R[:, 0], GRASP_HEIGHT)
                    gripper = GRIPPER_SHUT
                    speed, timeout = DESCEND_LINEAR, GRASP_TIME + 0.5
                else:
                    speed, timeout = MAX_LINEAR, 3.0      # goal set when grasping

                twist, err = pbvs_velocity(fk_tcp(q_hat).A, goal, PBVS_GAIN, speed, MAX_ANGULAR)

                # Chasing the block's POSITION alone leaves the tool a steady
                # speed/gain behind it - 14 mm at belt speed, which is half the
                # width of the block. Feeding the filter's VELOCITY estimate
                # forward cancels that lag and turns the chase into an
                # interception. Nothing measures this velocity; it comes
                # entirely from the filter.
                if use_v_feedforward and phase != "lift":
                    twist[:3] += objects.velocity[:3]
                    twist[5] += objects.velocity[3]

                qd_cmd = joint_velocity_from_twist(jacobian_tcp(q_hat), twist)

                # Integrate velocity into a smooth, acceleration-limited reference.
                qdd_ref = np.clip((qd_cmd - qd_ref) / DT, -4.0, 4.0)
                qd_ref = qd_ref + qdd_ref * DT
                q_ref = q_ref + qd_ref * DT

                if phase == "grasp":
                    # Fixed time for the fingers to close, then it is ours and
                    # it comes off the belt.
                    if t_phase > GRASP_TIME:
                        sim.release_block()
                        phase, t_phase = "lift", 0.0
                        goal = goal.copy()
                        goal[2, 3] += LIFT_HEIGHT

                else:
                    arrived = (np.linalg.norm(err[:3]) < TOL_POS
                               and np.linalg.norm(err[3:]) < TOL_ANG)
                    inside_since = (inside_since or t) if arrived else None

                    if inside_since is not None and t - inside_since > SETTLE_TIME:
                        inside_since = None
                        if phase == "align":
                            phase, t_phase = "descend", 0.0
                        elif phase == "descend":
                            phase, t_phase = "grasp", 0.0
                            grasp_info = grasp_report(sim, objects)
                        else:
                            phase, t_phase = "done", 0.0
                    elif t_phase > timeout:
                        if phase == "lift":
                            phase, t_phase = "done", 0.0
                        else:
                            reason = f"{phase} took too long"
                            break

            # Act.
            tau, accel_applied = joint_torque(model, q_hat, qd_hat,
                                              d_hat if use_d_feedforward else np.zeros(6),
                                              q_ref, qd_ref, qdd_ref,
                                              kp, kd, sim.torque_limit)

            # Disturbances, applied to the real robot only.
            sim.data.xfrc_applied[sim.model.body("forearm_link").id, :3] = 0
            if pbvs_start is not None:
                since = t - pbvs_start
                if push is not None and push_time <= since < push_time + 0.3:
                    sim.data.xfrc_applied[sim.model.body("forearm_link").id, :3] = push
                if nudge is not None and not nudged and since >= nudge_time:
                    adr = sim.model.jnt_qposadr[sim.model.joint("target_free").id]
                    sim.data.qpos[adr:adr + 2] += nudge
                    nudged = True

            sim.drive_block()        # keep the belt running under the block
            sim.step(tau, gripper)
            observer.predict(accel_applied)

            # Record.
            sigma = observer.sigma()
            log["t"].append(t);                 log["phase"].append(phase)
            log["q"].append(q_true);            log["qd"].append(qd_true)
            log["q_hat"].append(q_hat);         log["qd_hat"].append(qd_hat)
            log["d_hat"].append(d_hat);         log["q_meas"].append(q_meas)
            log["q_sigma"].append(sigma[:, 0]); log["qd_sigma"].append(sigma[:, 1])
            log["q_ref"].append(q_ref.copy());  log["tau"].append(tau)
            log["tcp"].append(sim.true_tcp());  log["cam"].append(cam_result)
            log["nees"].append(observer.nees(q_true, qd_true))
            blank = np.full(2 * ObjectFilter.N, np.nan)
            log["obj_est"].append(objects.x.copy() if objects.x is not None else blank)
            log["obj_sigma"].append(objects.sigma() if objects.x is not None else blank)
            log["obj_true"].append(sim.true_object_pose())
            log["obj_vel_true"].append(sim.true_object_velocity())
            if record is not None:
                record.maybe_capture(sim, observer, objects, t, phase)

            t += DT
            step += 1

        # Score it.
        lifted = sim.true_object_pose()[2] - BLOCK_HALF
        success = bool(phase == "done" and lifted > 0.08)
        if phase == "done":
            reason = "picked it up" if success else "did not lift it"

        log = {k: np.array(v) for k, v in log.items()}
        info = {"seed": seed, "success": success, "reason": reason,
                "block_xy": block_xy, "block_speed": block_speed,
                "looks_used": scan + 1,
                "time_s": t, "lift_m": lifted,
                "frames_used": objects.n_used, "frames_rejected": objects.n_rejected,
                "frames_blind": int(np.sum(log["cam"] == 1)), "filter_resets": objects.n_resets}
        if grasp_info:
            info.update(grasp_info)
        if len(log["t"]):
            moving = log["phase"] != "settle"
            info["tracking_rms_mrad"] = 1e3 * float(np.sqrt(np.mean(
                (log["q"][moving] - log["q_ref"][moving]) ** 2)))
            info["kf_qd_rmse_mrad_s"] = 1e3 * float(np.sqrt(np.mean(
                (log["qd"] - log["qd_hat"]) ** 2)))
            info["nees_mean"] = float(np.mean(log["nees"][50:]))
            info["torque_rms"] = float(np.sqrt(np.mean(log["tau"] ** 2)))
        if sim.renderer is not None:
            sim.renderer.close()
        return success, info, log

    finally:
        ENCODER_NOISE, DEPTH_NOISE, DEPTH_BIAS, RGB_NOISE = base_noise


def grasp_report(sim, objects):
    """How far off we were at the moment the gripper closed.

    Uses ground truth, so it is for marking the result only - by this point
    the robot has already committed to the grasp.
    """
    obj = sim.true_object_pose()
    tcp = sim.true_tcp()
    est = objects.x
    v_true = sim.true_object_velocity()
    return {
        "grasp_error_mm": 1e3 * float(np.linalg.norm(tcp[:2] - obj[:2])),
        "object_error_mm": 1e3 * float(np.linalg.norm(est[:3] - obj[:3])),
        "object_yaw_error_deg": float(np.rad2deg(abs(wrap_pi_2(est[3] - obj[3])))),
        # The filter is never told this: it works the speed out from the poses.
        "object_speed_mm_s": 1e3 * float(np.linalg.norm(v_true[:2])),
        "object_speed_error_mm_s": 1e3 * float(
            np.linalg.norm(objects.velocity[:2] - v_true[:2])),
    }
