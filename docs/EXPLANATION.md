# How the pipeline works

## 1. Joint model and analysis

A rigid manipulator obeys `M(q) q_ddot + h(q, q_dot) = tau + passive + contact`. MuJoCo provides the mass matrix `M`, the gravity/Coriolis term `h`, and passive damping. We evaluate them in a separate model built from estimated state, not the actual simulator state.

Choose `tau = M(q_hat) u + h(q_hat, v_hat) - passive(q_hat, v_hat)`. With accurate parameters in free motion this gives approximately `q_ddot = u`. Contacts, payload, state-estimation error, torque clipping and numerical integration make this approximation imperfect. It is not six independent physical motors with constant inertia.

Define `x = [q1 ... q6, v1 ... v6]`. At sampling interval `dt = 0.002 s`, constant acceleration gives:

```text
A = [ I  dt*I ]     B = [ 0.5*dt^2*I ]     C = [ I  0 ]
    [ 0     I ]         [       dt*I ]
x_next = A*x + B*u + process_noise
y = C*x + encoder_noise
```

The controllability matrix is `[B, AB, ..., A^11 B]`. The observability matrix stacks `C, CA, ..., CA^11`. Both ranks are 12, so all states are controllable and observable in this ideal compensated model. One instantaneous angle measurement cannot determine velocity, but a sequence of angles can.

`analyse_model.py` also finite-differences the actual MuJoCo torque dynamics at home with equilibrium torques, then selects the UR3 states from the full robot-plus-cube model. Its controllability and observability ranks are also 12. The numerical matrices and home mass matrix are saved. This is local evidence, not proof of global reachability or controllability during contact.

## 2. Frames and calibration

```mermaid
flowchart LR
  P[RGB pixels u,v and optical depth Z] --> C[Optical camera coordinates X,Y,Z]
  C -->|T_BC from estimated encoders and known mount| B[Robot base coordinates]
  B -->|T_WB fixed base pose| W[World-frame cup goal]
  W --> IK[IK planning copy]
  IK --> J[Joint reference]
```

Back-projection is `X = (u-cx)*Z/fx`, `Y = (v-cy)*Z/fy`. Depth means optical `Z`, not radial distance. The CPU sensor explicitly converts ray range into optical depth.

`T_BC = inverse(T_WB) * T_WC`. `T_WC` includes the MuJoCo-to-optical axis conversion `diag(1,-1,-1)`. Base orientation matters: this robot's base is rotated, so adding a base-frame point directly to the base's world position would be wrong.

Calibration moves a known reference sphere to 36 known optical positions in a separate simulation. Image centroids fit `u = fx*(X/Z)+cx` and `v = fy*(Y/Z)+cy`. Twenty-four positions fit the parameters; twelve positions evaluate reprojection residuals. This checks the simulated pinhole pipeline with rendered sensor observations. Known fixture positions and camera poses are legitimate calibration references. The simulated mount is known; we have not estimated real hand-eye parameters, lens distortion or depth scale from real data.

The red target is found using red dominance, positive finite depth, and the largest connected region. A 5x5 erosion removes mixed boundary pixels. Top-face pixels give a lateral centre, while median interior height avoids selecting positive depth noise. The horizontal-top-face assumption is deliberate and limits generalisation.

## 3. Observer

The KF first predicts `x_minus = A*x_hat + B*u` and `P_minus = A*P*A_transpose + Q`. With an encoder sample it calculates an innovation and Kalman gain, then corrects the state. With a missing encoder packet it keeps the prediction. Velocity never comes directly from MuJoCo in the controller.

`R = encoder_std^2 * I` uses the simulated encoder noise variance. Nominal encoder standard deviation is 0.001 rad. `Q = 4*B*B_transpose + 1e-10*I` models a 2 rad/s^2 acceleration uncertainty plus a numerical covariance floor. These are initial engineering choices, not fitted optimal values. A larger Q makes the filter react faster to modelling error but trust noisy measurements more. A smaller R makes it trust encoders more. The Joseph covariance update preserves numerical stability.

The prediction input is the acceleration implied by the clipped commanded torque in the estimated model. This accounts approximately for saturation, but not unknown contact force or the attached cube. Logs contain covariance and empirical error so that this mismatch is visible. Aggregate +/-2-sigma coverage is a diagnostic; it is not a full statistical consistency test across correlated samples.

## 4. Feedback and pick

The acceleration command is `u = q_ref_ddot + 64*(q_ref-q_hat) + 16*(q_ref_dot-v_hat)`. Gains correspond to a nominal natural frequency of 8 rad/s and damping ratio 1. The compensated discrete closed-loop poles are approximately 0.98537 and 0.98250, repeated six times. Both are inside the unit circle. That prediction assumes exact compensation and no saturation/contact.

Acceleration is clipped to +/-30 rad/s^2. Motor torque limits are 56, 56, 28, 12, 12, 12 Nm. A quintic blend provides zero velocity and acceleration at trajectory endpoints. The KF/controller runs at 500 Hz in simulation time. This does not imply real-time CPU performance.

The visual loop is slow and sampled: observe the target, move above it, reobserve and correct twice, then descend. During each motor move, the joint loop continues to use estimated states. The target must remain stationary while descending and lifting. The tool orientation remains the initial downward orientation; arbitrary target orientation is outside this version.

Suction engages only when the cup and target touch. Its relative pose uses estimated tool pose and the detected surface, with a known 2 cm half-height for this cube. The weld is an idealised, unlimited-strength attachment. No cube position is teleported during the pick. The only direct target-position assignments randomise the initial trial setup.

The imported stationary base mesh overlaps the adjoining shoulder mesh at their joint. A specific base/shoulder collision exclusion prevents this intended assembly overlap from resisting the motor. Other robot, table, target and cup collisions remain enabled.

## 5. Reading the evidence and explaining it in the viva

Angle RMSE compares estimated and true joint angles. Velocity RMSE tests the unmeasured states. Tracking RMSE compares actual angles with references, aggregated across all six joints and the entire trial. Torque RMS quantifies effort. Localisation error compares the first measured surface with the true cube top for evaluation only.

Success is a sustained object lift, not IK convergence or reaching a tool waypoint. Ten successes out of ten gives a 95% Wilson interval of roughly 72.2% to 100%, not proof of universal reliability. Reusing seeds across conditions pairs placements, but these are development seeds, not a held-out final benchmark.

Be ready to explain why the KF predicts during dropout, why ignoring the camera axis flip breaks localisation, why a local rank result does not ensure a successful grasp, why a disturbance increases velocity-estimation error, and why high depth noise can prevent contact even when joint tracking is excellent.
