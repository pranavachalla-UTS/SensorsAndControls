# UTS 41014 Assessment 4 - Track B: UR3 vision-guided pick

This branch adds a simulation pipeline with a wrist RGB-D sensor, simulated camera calibration, a joint Kalman observer, torque feedback, a sampled pose-based visual approach, and a contact-gated ideal suction pick. Python uses explanatory `#` comments. The original robot assets and captured measurements remain in the repository.

The original working checkpoint is commit `88e3d8884449d822e9ec49f1145ab752934dc2a3` on `main`. The old motion and scene-building preview scripts are also preserved in `legacy/` for reading. They refer to the old root layout and are historical examples, not the supported run commands.

## Install and run

Tested with Python 3.12 on macOS ARM64 and the versions in `requirements.txt`. From the repository folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python calibrate_camera.py
python analyse_model.py
python -m unittest discover -s tests -v
python run_pick.py --seed 0
python run_experiments.py --trials 10
python plot_results.py --condition nominal
python plot_results.py --condition torque_pulse
```

The default `ray` camera casts rays through MuJoCo geometry to produce flat RGB and metric optical-depth images on the CPU. Localisation receives only these pixels, not object IDs or object poses. This backend makes the experiment reproducible without an OpenGL context. It does not model lighting, textures, rolling shutter, or RGB sensor noise.

For the interactive viewer on macOS, use `.venv/bin/mjpython run_pick.py --seed 0 --viewer`. Elsewhere use `python run_pick.py --viewer`. The viewer still needs a working graphics connection. OpenGL RGB-D capture exists behind `--camera raster`, but was not validated here because the environment could not create a CoreGraphics connection. The supplied calibration is for `ray`; the code rejects using it for another backend.

Optional inspection steps: `python view_project.py`, `python detect_target.py`, and `python approach_target.py`. These save a paired capture and derived points under `results/`. Previewing no longer rewrites a scene. The complete pick always captures fresh images and does not rely on old `.npy` files.

## Where to look

| File | Purpose |
| --- | --- |
| `dynamic_scene.xml` | Existing UR3/table plus six torque motors, camera, cup, and initially inactive suction constraint |
| `robot_common.py` | Joint names, portable paths, transforms, and IK on a planning copy |
| `vision.py` | Pinhole RGB-D sensor, red-object segmentation, depth localisation |
| `calibrate_camera.py` | Offline simulated reference fixture, fitted intrinsics and held-out residuals |
| `joint_control.py` | Discrete state model, KF prediction/correction, estimated-state feedback |
| `analyse_model.py` | Controllability/observability and actual MuJoCo joint linearisation |
| `run_pick.py` | Fresh sensing, visual corrections, approach, contact, lift, and logs |
| `run_experiments.py` | Repeated seeded trials and uncertainty on success rates |
| `tests/` | Frame, observer, sensing, IK isolation, and dynamic motion checks |
| `docs/EXPLANATION.md` | Equations, design choices, limitations, and viva preparation |
| `docs/RESULTS.md` | Measured results and development failures |

## Sensing, estimation, and control

The camera is fixed to `tool0`. A paired RGB/depth image is back-projected using calibrated intrinsics. The optical axes are right/down/forward; a sign conversion maps them into MuJoCo's camera axes. Forward kinematics uses estimated encoder angles to compute the camera-to-base transform. See the frame diagram and equations in `docs/EXPLANATION.md`.

The twelve states are six angles and six velocities. Six noisy angle encoders are measured; velocities are estimated by a Kalman filter. Feedback uses these estimates and a separate model for dynamics compensation. The simulated arm advances with `mj_step` and motor torques. Position assignments occur only at initialisation or in a separate IK/kinematics copy.

The camera is sampled at home and during two approach corrections. Estimated object position determines a world-frame cup goal, IK finds a joint reference with the tool pointing down, and joint feedback follows a smooth trajectory. This is a slow sampled position-based visual loop, not continuous image-based servoing. The scene assumes a stationary, axis-aligned red cube on the known table.

A spherical cup must physically contact the cube before a weld constraint can engage. The weld represents ideal suction, not a realistic vacuum seal or finger gripper. It is configured from the vision/encoder estimates. A trial succeeds only when the actual cube centre remains above 0.70 m for the final 0.5-second hold. Ground-truth object position and joint velocity are used only for scoring; contact is the simulated cup's contact sensor.

## Evidence and limitations

Measured results are in `docs/RESULTS.md`, `results/experiment_summary.json`, per-trial JSON, and the calibration/model-analysis files. Full time-series CSV files are generated by the commands above and ignored by Git to keep the repository manageable. Each row includes truth for evaluation, estimates, covariance, reference, and torque.

These tests cover a limited workspace and ideal simulated geometry. They do not establish real-world grasp reliability, arbitrary-object performance, or a guaranteed mark. Before submission, review the explanations, rerun the commands on your own machine, verify the viewer, and record the actual five-minute demonstration. A simulation-only submission needs its own rigorous analysis; this branch does not claim hardware evidence.

## Contribution statement - complete before submission

Record each actual team member and the work they performed and can explain. Do not assign the generated implementation or experiments to a student without review. This section is intentionally unfinished until the team supplies its real contributions.

## Generative AI declaration

OpenAI Codex was used to inspect the repository, suggest and implement organisation, calibration, estimation and control code, develop tests, run simulated experiments, and draft explanations. Team members must review and understand the submitted work, verify the results, and update this declaration to reflect their actual use. Original UR3 assets are from the bundled `Robots/` description; retain its licence and attribution. Acknowledge any weekly tutorial code actually used by the team.

## Submission work still required

- Team review and contribution statement.
- A five-minute video showing a nominal pick, a varied/disturbed trial, and estimates/uncertainty against evaluation truth.
- A working link to that video and the final project page in Canvas.
- Stronger studies for higher-band evidence: larger held-out placement/noise sweeps, model mismatch and mount errors, estimator/control tuning comparisons, and quantified failure limits.
- Confirm with the teaching team whether this ideal suction model meets their expected pick demonstration. A physical gripper or realistic suction/contact model would provide stronger evidence.
