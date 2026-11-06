"""
Watch one pick live in the MuJoCo viewer.

Run:  mjpython watch.py                    (macOS needs mjpython for the viewer)
      mjpython watch.py --nudge            (block gets knocked mid-approach)
      mjpython watch.py --seed 4   (repeat an earlier run exactly)

Without --seed the red block lands somewhere new on the table every run; the
seed used is printed so a run can be repeated.

Uses the same run_pick as everything else; the viewer just hooks in where the
demo video's Recorder would, and slows the sim down to real time.
"""

import secrets
import sys
import time
import mujoco.viewer

import trackb as tb


class LiveViewer:
    """Stands in for make_video's Recorder: syncs a viewer instead of grabbing frames."""

    FPS = 50

    def __init__(self):
        self.viewer = None
        self.last_view = None          # set by run_pick: (rgb, mask)
        self.next_time = 0.0
        self.wall_start = None

    def maybe_capture(self, sim, observer, objects, t, phase):
        if t < self.next_time:
            return
        self.next_time = t + 1.0 / self.FPS
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
            self.wall_start = time.time()
        if not self.viewer.is_running():
            raise KeyboardInterrupt("viewer closed")

        # Hold the sim to real time.
        lag = t - (time.time() - self.wall_start)
        if lag > 0:
            time.sleep(lag)
        self.viewer.sync()


if __name__ == "__main__":
    if "--seed" in sys.argv:
        seed = int(sys.argv[sys.argv.index("--seed") + 1])
    else:
        seed = secrets.randbelow(100000)
    print(f"seed {seed}")
    options = {"nudge": (0.04, -0.03)} if "--nudge" in sys.argv else {}

    live = LiveViewer()
    try:
        success, info, _ = tb.run_pick(seed=seed, record=live, **options)
        x, y = info["block_xy"]
        print(f"block started at x={x:+.3f} y={y:+.3f} m, found on look {info['looks_used']}")
        print("success" if success else "failed", info)
        print("Close the viewer window to exit.")
        while live.viewer is not None and live.viewer.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
