"""
Run the whole project, start to finish.

    python run_all.py              everything (about 10 minutes)
    python run_all.py --quick      fewer trials in Part E (about 3 minutes)
    python run_all.py --no-video   skip rendering the demo video

Order matters: Part B writes the camera calibration and the measured sensor
noise, which Parts C, D and E load. (They will fall back to nominal values if
those files are missing, but the numbers in the report come from running Part B
first.)

Everything is printed as it goes, and the figures land in results/.
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUICK = "--quick" in sys.argv

STEPS = [
    ("part_a_model.py", [], "Part A - state-space model, controllability, observability"),
    ("part_b_sensing.py", [], "Part B - camera calibration and how accurate it really is"),
    ("part_c_estimation.py", [], "Part C - the two Kalman filters"),
    ("part_d_control.py", [], "Part D - computed torque, LQR, visual servoing"),
    ("part_e_robustness.py", ["--quick"] if QUICK else [],
     "Part E - repeated trials and disturbances"),
]
if "--no-video" not in sys.argv:
    STEPS.append(("make_video.py", [], "Demo video"))

start = time.time()
failed = []

for script, args, description in STEPS:
    print("\n" + "=" * 78)
    print(f"  {description}")
    print(f"  python {script} {' '.join(args)}".rstrip())
    print("=" * 78, flush=True)

    step_start = time.time()
    # Separate process, so one crash cannot take the rest down and each gets a clean interpreter.
    result = subprocess.run([sys.executable, script] + args, cwd=HERE)
    elapsed = time.time() - step_start

    if result.returncode == 0:
        print(f"\n  [done in {elapsed:.0f} s]")
    else:
        print(f"\n  [FAILED after {elapsed:.0f} s - exit code {result.returncode}]")
        failed.append(script)

print("\n" + "=" * 78)
if failed:
    print(f"  {len(failed)} script(s) failed: {', '.join(failed)}")
else:
    print(f"  All done in {(time.time() - start) / 60:.1f} minutes.")
print(f"  Figures and video are in {HERE / 'results'}")
print("=" * 78)

sys.exit(1 if failed else 0)
