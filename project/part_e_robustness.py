# Repeated trials, with raw evidence saved before a result table is published.
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
# Save figures without opening a desktop plotting window.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trackb as tb
from evidence import new_run_folder, provenance, write_json, wilson

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true", help="six development trials per condition")
parser.add_argument("--trials", type=int)
parser.add_argument("--held-out", action="store_true", help="use seeds beginning at 1000")
parser.add_argument("--start-seed", type=int)
parser.add_argument("--confidence-study", action="store_true",
                    help="compare baseline and uncertainty-aware grasp rules on held-out seeds")
parser.add_argument("--replot", type=Path, help="finish a saved, fully collected study without rerunning trials")
args = parser.parse_args()
if args.trials is not None and args.trials < 1:
    parser.error("--trials must be positive")
count = args.trials if args.trials is not None else (6 if args.quick else 30 if args.confidence_study else 15)
held_out = args.held_out or args.confidence_study
first_seed = args.start_seed if args.start_seed is not None else (1000 if held_out else 0)
if first_seed < 0:
    parser.error("--start-seed must be non-negative")
seeds = list(range(first_seed, first_seed + count))

if args.replot:
    folder = args.replot.resolve().parent
    study = json.loads(args.replot.read_text())
    seeds = study["seeds"]
    count = len(seeds)
    args.confidence_study = any(r["name"].endswith("_confidence") for r in study["conditions"])
    expected = 6 if args.confidence_study else 17
    if len(study["conditions"]) != expected:
        parser.error("Cannot publish a study with missing conditions")
    for row in study["conditions"]:
        if [r["seed"] for r in row["records"]] != seeds:
            parser.error("Cannot publish a study with missing trial records")
    # Preserve original trial provenance and record the separate chart repair.
    study["chart_provenance"] = provenance()
else:
    # Refuse to compare an uncalibrated fallback with the claimed calibrated system.
    tb.load_calibration()
    tb.load_noise_model()
    folder = new_run_folder("heldout-confidence" if args.confidence_study else "heldout" if held_out else "development")
    study = {"status": "running", "seeds": seeds, "split": "held_out" if held_out else "development",
             "protocol": "initial placements paired by seed; each condition uses the same seed list",
             "success_definition": "both pad contacts, then >=0.5 s above 80 mm lift and within 60 mm of TCP",
             "reported_errors": "both success-only means and all-defined-attempt means are saved",
             "provenance": provenance(), "conditions": []}
    write_json(folder / "study.json", study)


def mean_metric(records, key):
    values = [r[key] for r in records if key in r and r[key] is not None and np.isfinite(r[key])]
    return {"mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None, "count": len(values)}


def run_condition(name, **settings):
    trial_folder = folder / name
    records = []
    for seed in seeds:
        try:
            ok, info, _ = tb.run_pick(seed=seed, save_dir=trial_folder, **settings)
        except Exception as error:
            # A broken dependency/renderer is not a physical failed grasp.
            study["status"] = "execution_error"
            study["execution_error"] = {"condition": name, "seed": seed,
                                        "type": type(error).__name__, "message": str(error)}
            write_json(folder / "study.json", study)
            raise
        records.append(info)
        print(name, seed, "PASS" if ok else "FAIL", info["reason"], flush=True)
    successful = [r for r in records if r["success"]]
    summary = {"name": name, "settings": settings, "trials": len(records),
               "successes": len(successful), "success_rate": len(successful)/len(records),
               "success_wilson_95": wilson(len(successful), len(records)),
               "failure_reasons": dict(Counter(r["reason"] for r in records if not r["success"])),
               "success_only": {}, "all_defined_attempts": {}, "records": records}
    for key in ["grasp_error_mm", "object_error_mm", "tracking_rms_mrad", "time_s",
                "kf_qd_rmse_mrad_s", "nees_mean", "torque_rms", "min_hold_lift_m"]:
        summary["success_only"][key] = mean_metric(successful, key)
        summary["all_defined_attempts"][key] = mean_metric(records, key)
    study["conditions"].append(summary)
    write_json(folder / "study.json", study)
    return summary


if not args.replot:
    if args.confidence_study:
        # These seeds are kept separate from the normal 0..14 development figures.
        for condition, settings in [("nominal", {}), ("nudge", {"nudge": (.04, -.03)}),
                                    ("noise6", {"noise_scale": 6.0})]:
            for policy in ["baseline", "confidence"]:
                run_condition(condition + "_" + policy, grasp_policy=policy, **settings)
    else:
        run_condition("nominal")
        for scale in [3, 6, 10]:
            run_condition(f"noise_{scale}", noise_scale=scale)
        for force in [40, 120]:
            run_condition(f"forearm_push_{force}N", push=(force, 0, 0))
        for dx in [.03, .06]:
            run_condition(f"nudge_dx_{int(dx*100)}cm", nudge=(dx, -dx*.7))
        for mass in [.5, 2.0]:
            run_condition(f"payload_{mass}kg", payload=mass)
        run_condition("mount_error_2mm", mount_error_mm=2.0)
        run_condition("depth_scale_1p02", depth_scale=1.02)
        run_condition("parked_open_loop", open_loop=True, block_speed=0.)
        run_condition("moving_open_loop", open_loop=True)
        run_condition("no_velocity_feedforward", use_v_feedforward=False)
        run_condition("nudge_closed_loop", nudge=(.04, -.03))
        run_condition("nudge_open_loop", nudge=(.04, -.03), open_loop=True)

names = [r["name"] for r in study["conditions"]]
rates = [100*r["success_rate"] for r in study["conditions"]]
# Clamp harmless floating-point roundoff at 0% and 100% for Matplotlib.
low = [max(0., 100*(r["success_rate"]-r["success_wilson_95"][0])) for r in study["conditions"]]
high = [max(0., 100*(r["success_wilson_95"][1]-r["success_rate"])) for r in study["conditions"]]
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(np.arange(len(names)), rates, color=["tab:green" if r==100 else "tab:orange" if r>0 else "tab:red" for r in rates])
ax.errorbar(np.arange(len(names)), rates, yerr=[low, high], fmt="none", ecolor="black", capsize=3)
ax.set_xticks(np.arange(len(names)), names, rotation=35, ha="right", fontsize=8)
ax.set_ylim(0, 112)
ax.set_ylabel("successful sustained lifts [%]")
ax.set_title(f"{study['split']}: {count} trials per condition; 95% Wilson intervals")
ax.grid(axis="y", alpha=.25)
fig.tight_layout()
fig.savefig(folder / "success.png", dpi=150)
plt.close(fig)

study["status"] = "complete"
write_json(folder / "study.json", study)
# Only a completed study can replace the published pointer.
pointer = "latest_confidence_study.json" if args.confidence_study else "latest_study.json"
write_json(RESULTS / pointer, {"study": (folder / "study.json").relative_to(RESULTS).as_posix(),
                               "status": "complete"})
print("Saved reproducible study:", folder)
