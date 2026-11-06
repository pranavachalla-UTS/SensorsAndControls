"""
Render a demo video of one pick.

Run:  python make_video.py                    (nominal)
      python make_video.py --nudge            (block gets knocked mid-approach)
      python make_video.py --seed 4

Each frame shows the scene on the left and, on the right, what the wrist camera
sees plus the numbers that matter: the filter's estimate of the block, its own
uncertainty, and the true value next to it. The marking guide asks the video to
show "the estimate and its uncertainty alongside the true state", which is more
than just footage of the arm moving.

Writes: results/demo_<name>.mp4
"""

import secrets
import sys
import numpy as np
import cv2
import mujoco
import imageio.v2 as imageio
from pathlib import Path

import trackb as tb

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

WIDTH, HEIGHT = 1280, 720
PANEL = 440
SCENE_W = WIDTH - PANEL
FPS = 25

# OpenCV draws in BGR, and its ASCII-only font turns +- and degree signs into "?".
WHITE = (255, 255, 255)
GREY = (150, 150, 150)
BLUE = (214, 120, 42)
GREEN = (120, 190, 60)
ORANGE = (52, 104, 235)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Recorder:
    """Grabs a composite frame every 1/FPS of simulated time."""

    def __init__(self):
        self.frames = []
        self.next_time = 0.0
        self.scene = None
        self.last_view = None          # set by run_pick: (rgb, mask)

    def maybe_capture(self, sim, observer, objects, t, phase):
        if t < self.next_time:
            return
        self.next_time = t + 1.0 / FPS
        if self.scene is None:
            self.scene = mujoco.Renderer(sim.model, 720, SCENE_W)

        self.scene.update_scene(sim.data, camera="scene_cam")
        frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        frame[:, :SCENE_W] = self.scene.render()[:, :, ::-1]      # RGB -> BGR
        frame[:, SCENE_W:] = 25
        self._draw_panel(frame, sim, observer, objects, t, phase)
        self.frames.append(frame)

    def _draw_panel(self, frame, sim, observer, objects, t, phase):
        x0 = SCENE_W

        def write(text, x, y, scale=0.45, colour=WHITE):
            cv2.putText(frame, text, (x0 + x, y), FONT, scale, colour, 1, cv2.LINE_AA)

        write(f"t = {t:5.2f} s", 16, 30, 0.6)
        write(phase.upper(), 175, 30, 0.6, ORANGE)

        # What the camera sees, with the detected pixels tinted.
        write("WRIST CAMERA", 16, 52, 0.4, GREY)
        view = np.full((225, 300, 3), 40, np.uint8)
        if self.last_view is not None:
            rgb, mask = self.last_view
            view = cv2.cvtColor(cv2.resize(rgb, (300, 225)), cv2.COLOR_RGB2BGR)
            if mask is not None:
                small = cv2.resize(mask.astype(np.uint8), (300, 225),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                view[small] = (0.4 * view[small] + 0.6 * np.array(ORANGE)).astype(np.uint8)
            else:
                cv2.putText(view, "no detection", (85, 120), FONT, 0.55, (80, 80, 230), 2)
        frame[60:285, x0 + 16:x0 + 316] = view
        cv2.rectangle(frame, (x0 + 16, 60), (x0 + 316, 285), (90, 90, 90), 1)

        # The block: what we think, how sure we are, and the truth beside it.
        truth = sim.true_object_pose()
        write("BLOCK POSITION (ROBOT BASE FRAME)", 16, 315, 0.42, GREY)
        write("estimate +/- 1 sd", 110, 335, 0.4, BLUE)
        write("true", 330, 335, 0.4, GREEN)
        for i, label in enumerate(["x [mm]", "y [mm]", "z [mm]", "yaw [deg]"]):
            y = 358 + 22 * i
            scale = 1e3 if i < 3 else 180 / np.pi
            write(label, 16, y)
            if objects.x is not None:
                write(f"{scale*objects.x[i]:8.1f} +/-{scale*objects.sigma()[i]:5.1f}", 100, y, 0.44, BLUE)
            else:
                write("      --", 100, y, 0.44, GREY)
            write(f"{scale*truth[i]:8.1f}", 326, y, 0.44, GREEN)

        # Speed: the one row no sensor on the robot can produce. The filter
        # works it out from how the pose moves, and the controller needs it to
        # travel with the block rather than behind it.
        y = 358 + 22 * 4
        write("speed [mm/s]", 16, y)
        if objects.x is not None:
            v_est = 1e3 * float(np.linalg.norm(objects.velocity[:2]))
            v_sd = 1e3 * float(np.linalg.norm(objects.sigma()[4:6]))
            write(f"{v_est:8.1f} +/-{v_sd:5.1f}", 100, y, 0.44, BLUE)
        else:
            write("      --", 100, y, 0.44, GREY)
        write(f"{1e3 * np.linalg.norm(sim.true_object_velocity()[:2]):8.1f}",
              326, y, 0.44, GREEN)

        if objects.x is not None:
            err = 1e3 * float(np.linalg.norm(objects.x[:3] - truth[:3]))
            write(f"estimate is out by {err:5.2f} mm", 16, 470, 0.46,
                  GREEN if err < 5 else ORANGE)

        # The joint estimate the controller is actually running on.
        write("JOINT ESTIMATE (WHAT THE CONTROLLER USES)", 16, 500, 0.42, GREY)
        q_err = 1e3 * float(np.abs(observer.q - sim.true_q()).max())
        qd_err = 1e3 * float(np.abs(observer.qd - sim.true_qd()).max())
        write(f"angle    off by {q_err:6.2f} mrad", 16, 522, 0.44, BLUE)
        write(f"velocity off by {qd_err:6.1f} mrad/s", 16, 544, 0.44, BLUE)
        write("velocity is never measured, only estimated", 16, 562, 0.37, GREY)

        # Torque bars.
        write("JOINT TORQUE / MOTOR LIMIT", 16, 590, 0.42, GREY)
        for j, (tau, limit) in enumerate(zip(sim.data.ctrl[:6], sim.torque_limit)):
            y = 608 + j * 14
            fraction = float(np.clip(tau / limit, -1, 1))
            cv2.line(frame, (x0 + 290, y - 6), (x0 + 290, y + 2), GREY, 1)
            cv2.line(frame, (x0 + 290, y - 2), (x0 + 290 + int(130 * fraction), y - 2),
                     ORANGE if abs(fraction) > 0.9 else BLUE, 6)
            write(f"J{j+1} {tau:7.2f} Nm", 16, y, 0.4)

        cv2.rectangle(frame, (x0, HEIGHT - 40), (WIDTH, HEIGHT), (14, 14, 14), -1)
        write("True values are shown to check the result.", 16, HEIGHT - 24, 0.38, GREY)
        write("They never reach the estimator or controller.", 16, HEIGHT - 9, 0.38, GREY)

    def save(self, path):
        imageio.mimwrite(str(path), [f[:, :, ::-1] for f in self.frames],
                         fps=FPS, quality=8, macro_block_size=1)
        print(f"Saved {path}  ({len(self.frames)} frames, "
              f"{len(self.frames)/FPS:.1f} s)")


if __name__ == "__main__":
    seed = secrets.randbelow(100000)
    if "--seed" in sys.argv:
        seed = int(sys.argv[sys.argv.index("--seed") + 1])

    options, name = {}, "nominal"
    if "--nudge" in sys.argv:
        options, name = {"nudge": (0.04, -0.03)}, "nudge"
    elif "--open-loop" in sys.argv:
        options, name = {"open_loop": True, "nudge": (0.04, -0.03)}, "open_loop"
    elif "--noise" in sys.argv:
        options, name = {"noise_scale": 6.0}, "noise"

    print(f"Rendering '{name}' (seed {seed})...")
    recorder = Recorder()
    success, info, _ = tb.run_pick(seed=seed, record=recorder, **options)
    print(f"  {info['reason']}")
    recorder.save(RESULTS / f"demo_{name}.mp4")
