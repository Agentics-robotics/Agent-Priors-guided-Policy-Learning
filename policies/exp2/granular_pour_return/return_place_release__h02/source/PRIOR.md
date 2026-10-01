# Fresh-TCP body-increment prior

## Decision and targeted failure

This independent policy targets predecessor handoff variation and feasible intermediate states that differ from the nominal absolute path. Demonstrations enter with a moderate tilt, become upright at different times, travel rapidly toward the target, descend, open, and retreat. An absolute Cartesian sequence can be brittle if the previous skill switches at a different point or execution lags the demonstrated state.

## Mechanism

For every supervised slot, `encode_targets` computes the demonstrated commanded TCP with public FK and encodes

`T_measured_tcp_before_action^-1 * T_commanded_tcp`

as body-frame translation in metres, a local rotation vector in radians, and the demonstrated gripper scalar. `decode_action` composes the predicted local transform with the **fresh current measured TCP for that executed slot**, then calls `panda_kinematics.solve_ik` from fresh measured joints. It never integrates against a predicted pose. This is a feedback-like action coordinate, not a scripted controller.

The two causal input frames contain source, destination, and bowl poses relative to the TCP; source relative to destination; world TCP pose; and measured arm/finger configuration and velocity. These relations expose error and phase while preserving world reachability and robot posture. Fixed physical scales are applied in preprocessing; the framework fits the representation normalizer only on valid 7D labels. No augmentation or auxiliary objective is used.

## Model and gradient path

A standard `DiffusionBackbone` receives a flattened 126D condition and predicts epsilon for 16 slots of 7D local actions. The masked diffusion objective is fully active and trains all model parameters. No backbone modification is needed because robustness comes from the action frame and causal relational inputs.

## Evidence and overlap

All twelve complete frozen `[1460, stop)` slices are bound. The 280 demonstrated actions in `[1460,1740)` overlap `controlled_pour_and_right` and remain action-supervised. In demo12300, indices 1460-1640 show gradual righting, 1680-1740 show rapid outbound motion, 1820-1840 show setdown, 1850 shows opening, and 1880-1920 retreat. Demo12303 is already outbound by 1680 and opens at 1840; demo12308 also opens at 1840. This supports phase variation, but successful demonstrations do not prove recovery.

## Calling and handoff

The caller supplies symbolic identities `source_object="container"` and `destination="target_pose"`; adapters resolve current observed poses. Prefer this model for a still-secure tilted entry or an unusual but reachable intermediate state. Replan every eight actions. Switch to the destination-frame model after righting when destination displacement is the main concern, or to the phase-regularized world model near nominal support when release timing is ambiguous.

## Assumptions and limitations

TCP, source, destination, and joints are observed accurately; the source remains grasped until release; and local IK targets are feasible. Local coordinates can drift or oscillate and only require the network to extrapolate target error for large destination shifts, so they do not provide the destination policy's constructed equivariance. There is no contact truth or supervision for lost grasp, collision, failed support, regrasp, repouring, or premature-release recovery. IK diagnostics are always reported.
