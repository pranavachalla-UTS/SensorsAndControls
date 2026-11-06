import json
from run_pick import run_trial

# Use fresh sensing and actuator-driven motion rather than stale saved waypoints.
# On macOS, run the viewer with mjpython move_robot.py.
if __name__ == '__main__':
    print(json.dumps(run_trial(viewer=True), indent=2))
