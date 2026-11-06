# One reproducible command. A failed stage stops its dependent stages.
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from evidence import provenance, write_json

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "results" / ".mpl-cache"))
os.environ.setdefault("MPLBACKEND", "Agg")
parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true")
parser.add_argument("--no-video", action="store_true")
parser.add_argument("--held-out", action="store_true")
parser.add_argument("--confidence-study", action="store_true")
args = parser.parse_args()
steps = [("tests.py", []), ("part_a_model.py", []), ("part_b_sensing.py", []),
         ("part_c_estimation.py", []), ("part_d_control.py", [])]
options = ["--quick"] if args.quick else []
if args.held_out:
    options.append("--held-out")
steps.append(("part_e_robustness.py", options))
if args.confidence_study:
    study_options = ["--confidence-study"] + (["--quick"] if args.quick else [])
    steps.append(("part_e_robustness.py", study_options))
if not args.no_video:
    steps += [("make_video.py", ["--seed", "4"]), ("make_video.py", ["--seed", "4", "--nudge"])]

report = {"status": "running", "provenance": provenance(), "stages": []}
for name, options in steps:
    print(f"\nRunning {name} {' '.join(options)}", flush=True)
    started = time.monotonic()
    log_path = HERE / "results" / "execution" / (name.replace(".py", "") + "_" + "_".join(o.lstrip('-') for o in options) + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([sys.executable, str(HERE / name), *options], cwd=HERE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout, flush=True)
    log_path.write_text(result.stdout)
    report["stages"].append({"script": name, "options": options, "returncode": result.returncode,
                             "wall_seconds": time.monotonic()-started,
                             "log": log_path.relative_to(HERE).as_posix()})
    if result.returncode:
        report["status"] = "blocked"
        report["blocked_stage"] = name
        report["message"] = "Dependent stages were not run; no nominal-calibration fallback was used."
        write_json(HERE / "results" / "validation.json", report)
        subprocess.run([sys.executable, str(HERE / "update_project_page.py")], cwd=HERE)
        raise SystemExit(result.returncode if result.returncode > 0 else 1)
report["status"] = "complete"
write_json(HERE / "results" / "validation.json", report)
subprocess.run([sys.executable, str(HERE / "update_project_page.py")], cwd=HERE, check=True)
print("All selected stages passed. Review the saved reports and updated project page.")
