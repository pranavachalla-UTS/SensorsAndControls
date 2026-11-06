# Refresh only generated sections. Manual submission details remain untouched.
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PAGE = HERE / "README.md"


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def replace_section(text, name, body):
    start, end = f"<!-- {name}:START -->", f"<!-- {name}:END -->"
    if start not in text or end not in text:
        raise ValueError(f"Missing generated-section markers: {name}")
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    return before + start + "\n" + body.rstrip() + "\n" + end + after


validation = read(RESULTS / "validation.json")
if validation is None:
    status = "Revised code has not completed the validation runner yet. Historical figures are archived and are not current results."
else:
    status = f"Latest runner status: **{validation['status']}**."
    if validation.get("blocked_stage"):
        status += f" Stopped at `{validation['blocked_stage']}`; dependent studies were not run."
    if validation.get("message"):
        status += "\n\n" + validation["message"]
    passed = [s["script"] for s in validation["stages"] if s["returncode"] == 0]
    if passed:
        status += "\n\nPassed stages: " + ", ".join(f"`{s}`" for s in passed) + "."
    status += "\n\n[Machine-readable validation record](results/validation.json). Full stage logs are under `results/execution/`."

lines = []
for pointer_name, title in [("latest_study.json", "Repeated-trial study"),
                             ("latest_confidence_study.json", "Held-out confidence-policy comparison")]:
    pointer = read(RESULTS / pointer_name)
    study = read(RESULTS / pointer["study"]) if pointer else None
    if study is None or study.get("status") != "complete":
        lines += [f"**{title}: pending a completed graphics-enabled run.**", ""]
        continue
    path = Path(pointer["study"])
    lines += [f"### {title}", "", f"Split: `{study['split']}`. Seeds: {study['seeds'][0]}–{study['seeds'][-1]}. Errors below use all attempts with a defined measurement, including failures.", "",
              "| Condition | Sustained lifts | 95% Wilson interval | Grasp error, all defined attempts [mm] | Cycle time, all attempts [s] |",
              "| --- | ---: | --- | ---: | ---: |"]
    for row in study["conditions"]:
        lower, upper = row["success_wilson_95"]
        def metric(key):
            value = row["all_defined_attempts"][key]["mean"]
            return "—" if value is None else f"{value:.2f}"
        lines.append(f"| {row['name']} | {row['successes']}/{row['trials']} | {100*lower:.1f}–{100*upper:.1f}% | {metric('grasp_error_mm')} | {metric('time_s')} |")
    lines += ["", f"[Raw records, settings and provenance](results/{path.as_posix()}). NPZ time histories regenerate with the same command and are retained locally.", "", f"![Success and sample uncertainty](results/{path.parent.as_posix()}/success.png)", ""]
    failures = [f"{row['name']}: {row['failure_reasons']}" for row in study["conditions"] if row["failure_reasons"]]
    if failures:
        lines += ["Failure counts: " + "; ".join(failures) + ".", ""]

cal = read(RESULTS / "calibration_report.json")
noise = read(RESULTS / "noise_report.json")
if cal and noise:
    lines += ["### Held-out calibration", "",
              f"Fit/solver selection used {cal['training_views']} views; {cal['validation_views']} separate views were held out.",
              f"Held-out reprojection RMS: **{cal['validation_reprojection_rms_px']:.3f} px**. Fixed-board scatter RMS: **{cal['validation_board_scatter_rms_mm']:.3f} mm**.",
              "[Calibration report](results/calibration_report.json) · [Noise measurements](results/noise_report.json)", ""]
else:
    lines += ["Calibration and noise fitting: **pending a successful Part B run**. Picks require the resulting files and cannot silently use nominal parameters.", ""]

model = read(RESULTS / "model_report.json")
est = read(RESULTS / "estimation_report.json")
control = read(RESULTS / "control_report.json")
if model and est and control:
    trial = est["trial"]
    lines += ["### Model, observer and controller checks", "",
              f"At the reported operating point, controllability and observability ranks are **{model['controllability_rank']}/12** and **{model['observability_rank']}/12**. This is a local smooth-model result; contacts and saturation are outside that proof.", "",
              f"On the Part C seed-{trial['seed']} development trial, velocity RMSE was **{1000*est['velocity_kf_rmse_rad_s']:.2f} mrad/s** for the observer versus **{1000*est['velocity_fd_rmse_rad_s']:.2f} mrad/s** for encoder differencing. Mean time-aligned joint NEES was **{trial['nees_mean']:.3f}** for the three-state per-joint filter. A mean below 3 suggests conservative uncertainty on this trial; it is not a multi-trial consistency proof.", "",
              f"The Part D trial recorded peak joint torque **{control['peak_torque_nm']:.2f} Nm**. LQR gains are kp={control['lqr_kp']:.2f}, kd={control['lqr_kd']:.2f}. The weight sweep is a compensated double-integrator analysis, not repeated nonlinear robot trials.", "",
              "[Model report](results/model_report.json) · [Observer report](results/estimation_report.json) · [Control report](results/control_report.json)", ""]

page = PAGE.read_text()
page = replace_section(page, "VALIDATION", status)
page = replace_section(page, "RESULTS", "\n".join(lines))
PAGE.write_text(page)
print("Updated generated results/status only; video, contributions and AI sections were preserved.")
