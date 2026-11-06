# Save enough information to trace each result back to code, calibration and a seed.
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import time
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def clean(value):
    # JSON has no portable NaN value. Keep missing measurements explicitly null.
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + "\n")


def file_hash(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def provenance():
    def git(*args):
        result = subprocess.run(["git", "-C", str(HERE), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    versions = {}
    for name in ["mujoco", "numpy", "opencv-python", "opencv-python-headless",
                 "roboticstoolbox-python", "control", "matplotlib", "imageio"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    files = sorted(HERE.glob("*.py")) + [HERE / "ur3_pick_scene.xml", HERE / "requirements.txt"]
    return {"utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "versions": versions, "git_commit": git("rev-parse", "HEAD"),
            "git_status": git("status", "--porcelain"),
            "source_sha256": {f.name: file_hash(f) for f in files},
            "calibration_sha256": file_hash(HERE / "calibration.npz"),
            "noise_model_sha256": file_hash(HERE / "noise_model.npy")}


def new_run_folder(label):
    # Nanoseconds prevent one run silently overwriting another run's evidence.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = RESULTS / "runs" / f"{label}-{stamp}-{time.time_ns() % 1000000000:09d}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def save_trial(info, log, folder=None):
    folder = new_run_folder("pick") if folder is None else Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    name = f"seed_{info['seed']:05d}"
    prefix = folder / name
    if prefix.with_suffix(".json").exists():
        raise FileExistsError(f"Trial already saved: {prefix}. Choose a new run folder.")
    np.savez_compressed(prefix.with_suffix(".npz"), **log)
    info["history_file"] = prefix.with_suffix(".npz").relative_to(RESULTS).as_posix()
    info["history_sha256"] = file_hash(prefix.with_suffix(".npz"))
    info["provenance"] = provenance()
    write_json(prefix.with_suffix(".json"), info)
    return prefix


def wilson(successes, total):
    if total <= 0:
        return [None, None]
    p, z = successes / total, 1.96
    denominator = 1 + z*z / total
    centre = (p + z*z/(2*total)) / denominator
    half = z * np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0.0, centre-half), min(1.0, centre+half)]
