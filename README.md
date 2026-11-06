# UR3 vision-guided pick - UTS 41014 Assessment 4, Track B

Start here. You normally only need **this README and `main.py`**. The other folders hold supporting code, robot files and evidence.

## Folder guide

```text
main.py           One place to run the project
requirements.txt  Python dependencies
src/              Sensing, observer, controller and experiment code
models/           Current MuJoCo scene
assets/           Original robot meshes, descriptions and licences
data/            Saved camera calibration
results/          Summary, combined trial records and figures
docs/            Explanations, assessment guide and video plan
tests/           Automated checks
archive/          Old models, captures, results and preparation scripts
```

You do not need to read or edit every file. Keeping sensing, estimation and control in separate small files makes them easier to explain than one very long script. The large `assets/Robots/` folder is third-party robot material; its original files and licences are preserved. Old work lives in `archive/` and is not needed for a normal pick.

## Install and run

Tested with Python 3.12 on macOS ARM64. From the repository folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py pick --seed 0
```

Other commands use the same launcher:

```sh
python main.py test
python main.py calibrate
python main.py model
python main.py trials --trials 10
python main.py plot --condition nominal
python main.py plot --condition torque_pulse
```

For a viewer on macOS: `.venv/bin/mjpython main.py pick --seed 0 --viewer`. Elsewhere: `python main.py pick --viewer`. The viewer needs a working graphics connection and was not validated in this environment.

Optional inspection commands are `python main.py preview`, `python main.py detect`, then `python main.py approach`. These create a paired camera capture and derived points in `results/`. A full pick always takes fresh images.

## What the project does

The wrist camera detects a red tabletop cube from RGB and optical-depth pixels. Estimated encoder angles provide the camera-to-base transform. A twelve-state Kalman observer estimates six joint angles and six velocities. Torque feedback follows smooth joint references, with two fresh visual corrections before descent. The arm moves through MuJoCo dynamics, not animated joint-position assignments.

A spherical cup must contact the cube before an ideal suction weld engages. A pick succeeds only if the actual cube centre stays above 0.70 m throughout the final 0.5-second hold. Ground truth is used for evaluation; the controller uses estimated state. This version assumes a stationary, axis-aligned cube and an unlimited-strength suction attachment.

The default camera casts rays through MuJoCo geometry on the CPU to produce flat RGB and metric depth images. Localisation receives pixels, not object IDs or poses. It does not model lighting or RGB noise. The OpenGL camera option `--camera raster` exists but was not validated: CoreGraphics failed in this environment. Supplied calibration belongs to the CPU camera; using it for another backend is rejected.

## Where the important work is

| Topic | File |
| --- | --- |
| Joint model, KF and feedback | `src/joint_control.py` |
| Frames, joint names and IK | `src/robot_common.py` |
| RGB-D localisation | `src/vision.py` |
| Complete pick sequence | `src/run_pick.py` |
| Calibration and model analysis | `src/calibrate_camera.py`, `src/analyse_model.py` |
| Repeated experiments | `src/run_experiments.py` |
| Student explanations and equations | [docs/EXPLANATION.md](docs/EXPLANATION.md) |
| Measured results and limitations | [docs/RESULTS.md](docs/RESULTS.md) |
| Five-minute recording plan | [docs/DEMO_PLAN.md](docs/DEMO_PLAN.md) |

The saved forty-trial evidence is grouped in `results/trials.json`; its summary is `results/experiment_summary.json`. Figures are in `results/figures/`. Re-running trials creates individual working logs in `results/raw/`, which Git ignores. Earlier failed development summaries remain in `archive/development/`.

The original checkpoint is commit `88e3d8884449d822e9ec49f1145ab752934dc2a3`. Historical scripts and original captures are preserved in `archive/`. Optional URDF regeneration helpers in `archive/preparation/` require `xacro` and were not rerun; they are not part of the normal workflow.

## Before submission

Review and understand the code, validate the viewer on your machine, record the five-minute video and add its real link, and fill in actual team contributions. Larger held-out placement/noise studies, model mismatch, mount errors and more realistic suction mechanics would strengthen the evidence. The tested simulation does not establish real-world reliability or guarantee a mark. Confirm with the teaching team whether ideal suction meets their expected pick demonstration.

**Contribution statement:** record each actual member and the work they performed and can explain. This remains unfinished until the team supplies its real contributions.

**Generative AI declaration:** OpenAI Codex helped inspect and organise the repository, implement calibration/estimation/control, create tests, run simulated experiments, and draft explanations. Team members must review the submitted work and update this declaration to reflect actual use. Retain the original licences in `assets/Robots/` and acknowledge any weekly tutorial code actually used.
