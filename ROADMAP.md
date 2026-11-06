# Build-it-yourself roadmap — Track B

A build order for the vision-guided pick, in ten stages.

The order is **not** Part A → E. The marking criteria are a way of presenting
the finished thing; this is the order you can actually build it in, where each
stage depends only on the ones before it.

**The rule that makes this work:** every stage ends with a test you can run on
its own. Never move on from a stage you cannot demonstrate. If you build the
whole pipeline and only then press go, a 2 mm error could be coming from any of
six places and you will lose days.

Reference implementation is in `project/`. Use it when you are stuck, not
before — you learn this by getting it wrong first.

---

## Stage 0 — A robot in a world

**Goal.** A UR3 standing on a table in MuJoCo, and a Python script that loads
it and prints something.

**Build.**
- Install: `mujoco`, `roboticstoolbox-python`, `control`, `opencv-python` (**4.x — OpenCV 5 removed `cv2.calibrateHandEye`**), `matplotlib`.
- Write the scene XML: floor, table, the UR3 links. Steal the link/inertia
  block from `ur3.xml` that `load_ur3.py` generated in your earlier work — it
  came from the official `ur_description`, so the numbers are right.
- Add torque motors (`<motor>`), one per joint, with the UR3's real limits:
  56/56/28/12/12/12 N m.

**You'll know it works when.** `mujoco.viewer` opens, the arm is on the table,
and with zero torque it flops over under gravity. That flop is the point — it
proves gravity is on and the motors are doing nothing yet.

**Time.** 1–2 hours.

---

## Stage 1 — Kinematics you trust

**Goal.** Given joint angles, know exactly where the tool is. Given a desired
tool pose, find joint angles.

**Build.**
- `rtb.models.UR3()` gives you `fkine`, `jacob0`, `ikine_LM`.
- Decide where your TCP is (mine: 105 mm out along tool z) and apply that
  offset consistently.

**You'll know it works when.** Three tests pass:
1. `rtb` FK and MuJoCo's own tool position agree to under a micrometre at
   several random joint angles. (Mine matched to 1e-12 m. If yours is out by
   centimetres, your frames disagree — find it now, not in Stage 7.)
2. IK round-trips: pick random angles → FK → IK → you get the same pose back.
3. The Jacobian matches a finite difference of FK.

**Trap.** Seed `ikine_LM` with the current joint angles. Unseeded it will
happily hand you an elbow-down, wrist-flipped solution that is correct and
useless.

**Time.** 2–3 hours.

---

## Stage 2 — Make the arm hold itself up

**Goal.** The arm tracks a joint trajectory, against gravity. No vision, no
estimation — use the true state from the simulator for now.

This is the biggest conceptual stage. Do not rush it.

**Build.**
- **Computed torque.** `tau = M(q)·v + c(q,qdot) + g(q)`. Get `M` and the bias
  from a *separate* model instance (I used a second MuJoCo model; `rtb` works
  too but its `coriolis()` has a bug on the URDF UR3 — use `rne(q, qd, 0)`
  instead, which returns `c + g` in one call).
- This turns the arm into six independent double integrators, `qddot = v`.
- **LQR** on that double integrator. Weights by Bryson's rule:
  `Q = diag(1/e_max², 1/v_max²)`, `R = 1/a_max²`. Use `control.dlqr`.
- A quintic trajectory to move between two joint configurations.

**You'll know it works when.**
1. Commanding the home position, the arm holds still and the torque equals the
   gravity torque. Sub-millimetre droop.
2. It tracks a quintic from home to another pose with tracking error under a
   few mrad.
3. Deliberately break it: feed the controller a wrong mass (say ×1.5). Tracking
   should get worse but not blow up. That tells you feedback is doing work, not
   just the feedforward.

**Trap.** If you get the sign of the bias torque wrong the arm accelerates into
the table convincingly. Check gravity-hold first, before any motion.

**Time.** 4–6 hours.

---

## Stage 3 — Stop using the true state

**Goal.** Replace the simulator's true `q` and `qdot` with estimates. This is
Part C and it is where the subject's content actually lives.

**Build.**
- **Encoder model.** True angle + white noise, then rounded to the encoder's
  resolution (14-bit → 0.38 mrad). Position only — no velocity output.
- **Kalman filter per joint**, state `[q, qdot, d]`:
  - constant-acceleration model,
  - `d` is an acceleration disturbance (friction, unmodelled payload),
  - `R` = noise variance + `resolution²/12` (quantisation is uniform noise).
- Feed `d̂` back into the controller: `v = qdd_ref - kp·e - kd·ė - d̂`.

**You'll know it works when.**
1. The velocity estimate beats differencing the encoder by roughly 10×.
   Plot both against truth — that plot is one of your best figures.
2. `d̂` settles on a non-zero value and cancelling it improves tracking
   several-fold. (Mine: 4.74 → 0.32 mrad, a 15× gain.)
3. **NEES** averages somewhere near 2. If it is 300, you are over-confident
   and your `Q` is too small.

**Trap.** When the torque saturates you did not apply the `v` you asked for.
Tell the observer what you *actually* applied, or it reads your own saturation
as an external disturbance and fights it.

**Time.** 3–4 hours.

---

## Stage 4 — A camera you can trust

**Goal.** A wrist camera, calibrated the way you would on a real robot.

**Build.**
- Add the camera to the scene. **MuJoCo cameras look down their own −z with y
  up**; OpenCV wants x right, y down, z forward. One `diag(1,−1,−1)` converts,
  and it should live in exactly one place in your code.
- A checkerboard you can place on the table.
- Move the arm to ~20 varied poses, grab images, `cv2.findChessboardCorners`,
  `cv2.cornerSubPix`, `cv2.calibrateCamera`.
- Hand-eye: `cv2.calibrateHandEye` using FK of the **encoder-measured** angles.

**You'll know it works when.**
- Reprojection error well under a pixel.
- Hand-eye result within a millimetre of where you mounted the camera.
- **The check that needs no ground truth:** the board never moved, so computing
  its pose through `FK(q)·T_tool_cam·T_cam_board` from every view should give
  the same answer each time. Low scatter means your calibration is consistent.
  Put this in your report — it is the one that would work on real hardware.

**Traps.** Vary distance *and* tilt *and* roll; photos from one spot cannot
separate focal length from distance. Use an odd × even corner count (9×6) so
the board is not symmetric under 180°.

**Time.** 3–4 hours.

---

## Stage 5 — Find the block and say where it is

**Goal.** One RGB-D frame → the block's position and orientation in base
coordinates.

**Build.**
- Segment in **HSV**, not RGB — hue separates "is it red" from "how bright is
  it", which survives the arm shadowing the table.
- Largest connected blob; reject it if it touches the image edge (you are only
  seeing part of the block, so its centre is wrong).
- Back-project every masked pixel with its own depth, transform to base frame
  with `FK(q̂) · T_tool_cam`.
- Top face = highest few mm of points. Mean → x, y. PCA of those points → yaw.
- A ring of pixels just outside the blob is the table → block height → centre z.

**You'll know it works when.** You localise a block of known pose to within a
couple of millimetres, and adding green and blue decoys does not fool it.

**Traps.**
- **Your depth noise model decides your results.** Independent per-pixel noise
  averages away over a 5000-pixel blob and you will "achieve" 0.02 mm, which no
  real camera does. Add a smooth per-frame bias field that neighbouring pixels
  share. That is what sets the real accuracy floor.
- Take the 90th percentile for the top face, not the max — one bad pixel
  should not define the top of your block.
- A block's long axis is its local **y** if it is 30 × 50 mm. Getting this
  wrong gives a confident 90° error.

**Time.** 3–4 hours.

---

## Stage 6 — Fuse the frames

**Goal.** A Kalman filter on the block's pose, updated by every camera frame.

**Build.**
- State `[x, y, z, yaw]`, no motion model (the block sits still), small process
  noise so it *can* move if the evidence says so.
- **Measure your own `R`.** Hold the arm still, take 40 frames at six
  different heights, and record both the spread and the error from truth.
- Fit `sigma(Z) = a + b·Z²` and recompute `R` per frame from that frame's range.
- A chi-squared gate to reject outliers; after ~4 consecutive rejections,
  conclude the block really moved and re-initialise there.

**You'll know it works when.** The estimate is better than any single frame,
and if you teleport the block mid-run the filter rejects a few frames and then
jumps to the new position.

**Trap — this one is worth a mark or two.** Your measurement will be *precise
but not accurate*: spread a few tenths of a millimetre, error over a
millimetre. Fit `R` to the **total RMS error**, not the spread. A filter tuned
on repeatability claims ±0.3 mm while being 1.5 mm out, and its gate then
throws away perfectly good measurements.

**Time.** 2–3 hours.

---

## Stage 7 — Visual servoing

**Goal.** The arm drives itself to the block using the filter's estimate,
updated continuously.

**Build.**
- **PBVS**: `error = goal pose − current TCP pose` (6-vector: position, plus
  `log(R_goal · R_nowᵀ)` for rotation).
- `twist = gain · error`, saturated to a sane speed.
- `qdot = J⁺ · twist` — **damped** least squares, or it will ask for absurd
  joint speeds near a singularity.
- Integrate that into a smooth reference with an acceleration limit, and hand
  it to your Stage 2 controller. Do not command velocity directly.
- **Recompute the goal from the filter every control step.** This is the
  difference between visual servoing and "photograph, then move blind".

**You'll know it works when.** The tool converges on the block exponentially,
and the error plot is a clean decay.

**Trap.** A parallel gripper is unchanged by a 180° turn, so the block's yaw
only means anything modulo π. Wrap it, and pick whichever of the two equivalent
wrist angles is closer to where the wrist already is — otherwise it will
sometimes take the long way round for no reason.

**Time.** 4–6 hours.

---

## Stage 8 — Actually pick it up

**Goal.** Gripper, and the phase sequence that completes the task.

**Build.**
- Two sliding fingers, position servo with a **force limit** (mine: 15 N) so it
  squeezes rather than crushes or shoves. An equality constraint keeps them
  symmetric.
- Phases: `settle → to_observe → look → align → descend → grasp → lift`.
- Move on from a phase when the error has been inside tolerance for a short
  settling time — not the instant it first touches it, or noise will trigger it.

**You'll know it works when.** It picks the block up and holds it, from several
different starting positions.

**Trap.** You will discover the block **leaves the camera's view** on the way
down — a wrist camera cannot see what the gripper is closing on. Do not treat
this as a bug to hide. Let the filter coast and show its uncertainty growing.
It is an honest limitation and a good thing to be asked about in the viva.

**Time.** 2–3 hours.

---

## Stage 9 — Prove it, then break it

**Goal.** Part E. This is where Credit becomes Distinction.

**Build.**
- Randomise the block's x, y and yaw every trial, plus the decoys.
- Run 15 trials. Report success rate and error spread, not one lucky run.
- **Reuse the same seeds across conditions** so comparisons are paired — a
  difference between rows is then the condition, not luck about placement.
- Conditions: more sensor noise, a shove to the arm, the block knocked
  mid-approach, an unmodelled payload.
- **Push each one until it breaks**, then diagnose *why* from your logs. A
  found failure is worth more than another 100% row.
- **The ablation that matters:** closed-loop vs open-loop (camera off after the
  first look) *with the block knocked mid-approach*. With nothing disturbed,
  open loop works fine — which is exactly why the comparison needs a
  disturbance to mean anything. Mine: 15/15 vs 0/15.

**You'll know it works when.** You can say "it fails at 10× noise, and here is
the number that explains why" instead of "it works".

**Time.** 3–4 hours.

---

## Stage 10 — Write it up

**Goal.** The deliverables. Do not leave this to the last night; the marking
guide weights the project page and video heavily.

**Checklist from the brief:**
- [ ] GitHub project page: track, model, **frame diagram**, estimator and
      controller design *with parameter values*, quantitative results,
      install and run instructions that actually work from a fresh clone
- [ ] Code organised and runnable from a fresh environment
- [ ] **Ground truth used only for evaluation, and visibly so** — name every
      such function `true_*` so a marker can grep for it
- [ ] 5-minute video: the task completing, at least one disturbed trial, and
      the estimate *with its uncertainty* next to the true value
- [ ] Contribution statement per member
- [ ] Generative AI declaration

**Write down what went wrong and what you changed.** The Distinction band asks
for failures traced to causes. The debugging you did in Stages 2–7 *is* that
content — keep notes as you go, because you will not remember in three weeks.

**Time.** 4–6 hours.

---

## Rough plan

| Stage | | Hours |
|---|---|---|
| 0–1 | Scene and kinematics | 4 |
| 2 | Controller | 5 |
| 3 | Observer | 4 |
| 4–5 | Camera, calibration, detection | 7 |
| 6 | Object filter | 3 |
| 7–8 | Servoing and grasping | 8 |
| 9 | Experiments | 4 |
| 10 | Write-up and video | 5 |
| | **Total** | **~40** |

Due 3 November. Stages 0–3 (no vision at all) make a complete, demonstrable
milestone — get there first and you have something to show your group and the
teaching staff even if the rest runs late.

---

## The five mistakes that cost me the most time

1. **`rank(ctrb(A,B))` lied.** It reported 9/12 for a system that is fully
   controllable, because the matrix is appallingly conditioned. Use the **PBH
   test** instead — check `[A − λI, B]` one eigenvalue at a time.
2. **My first depth noise model was far too kind.** Independent per-pixel noise
   averaged away, giving 0.02 mm localisation. Correlated bias is what real
   stereo cameras suffer from.
3. **I nearly tuned the object filter on repeatability instead of accuracy.**
   They differ by 10× here.
4. **Frame conventions.** MuJoCo's camera axes and the gripper's 180° symmetry
   each cost an hour of confusion. Write both down once, in one place.
5. **Chasing NEES = 2 when the error was a bias.** Friction flipping sign is a
   step, not a random walk. No amount of process noise models a step well —
   at some point the honest move is to say so in the report rather than
   inflate `Q` until the number looks right.
