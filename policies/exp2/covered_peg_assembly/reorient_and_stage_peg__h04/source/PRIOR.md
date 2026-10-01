# Destination-frame desired-peg prior

## Decision and targeted failures

This independent policy targets two related deployment changes at the retrieval-to-reorientation handoff: a reachable displacement of the square-hole fixture and a different but still rigid peg-to-TCP grasp transform. It learns the desired **peg pose in the newest observed hole frame**, rather than learning a TCP pose directly in the hole frame (h01), a TCP pose in the held-peg frame (h02), or a local TCP transform (h03).

The destination displacement invariant applies by construction to the represented peg route: if the observed hole and intended route are rigidly displaced together, the same hole-relative actions decode to correspondingly displaced world targets. This is strongest for transport into the fixture neighborhood, final reorientation, hole-relative y/z centering, descent, and the retained early insertion. The full lift/turn/transport segment remains learned, but changing only the destination can still change the source-to-destination relation, collision geometry, and reachability.

The grasp factorization separately adapts a desired peg target to the causal grasp geometry. It is intended for a valid rigid grasp offset, not for a dropped or slipping peg.

## Evidence and motivation

The original-policy validation is in distribution only. h01 passed the final rule in 5/12 runs, h02 in 8/12, and h03 in 0/12. All 36 runs recorded a transient first pass before final scoring. h01 had no IK-unconverged steps but often ended outside the complete placement rule. All four h02 failures had 78-101 IK-unconverged steps. Every h03 run ended with `peg_head_inserted=true` and `peg_axis_aligned=false`. These are observed outcomes, not causal diagnoses.

The demonstrations use essentially one hole pose. At index 1120, peg x is approximately -0.610 to -0.622 m and peg z is 0.028 to 0.040 m. The route lifts well above the 0.05 m box wall, turns the peg toward identity orientation, and descends to hole height. At index 1380, demo12205 still has peg y about 0.160 m while demo12208 has y about 0.180 m. End x varies approximately -0.252 to -0.185 m. These facts support learning from geometric state rather than a single endpoint, but they do not establish recovery.

The expected benefit is falsifiable: direct desired-peg supervision may make the object alignment variable easier to learn and may reduce sensitivity to a valid changed grasp offset. There is no h04 validation result yet, and no out-of-distribution result exists for any policy.

## Causal inputs

`build_inputs` constructs `[2,76]` fixed-scale features. Each causal frame contains:

- measured Panda arm/finger `qpos` and `qvel`;
- peg, TCP, target, and lid poses expressed in that frame's observed hole frame, each as translation plus continuous 6D rotation;
- that frame's hole pose relative to the newest hole anchor;
- the frame's peg pose relative to its measured TCP, exposing the candidate grasp transform;
- lid angle and velocity;
- peg clearance above the box wall; and
- TCP-to-peg coupling distance.

This retains robot configuration, kinematic branch, velocity, finger state, cover/obstacle state, grasp geometry, destination geometry, and clearance dependencies. No image, contact truth, future state, or action history enters inference. The framework supplies two frames oldest-first and duplicates only the initial episode frame.

At each replan, `chunk_context` stores the newest observed `hole_pose` and the causal rigid transform

`G = T_tcp^-1 * T_peg`.

Both remain fixed for the eight executed actions and refresh at the next replan.

## Cartesian target conversion

For each demonstrated native command, `encode_targets` first obtains the exact command TCP pose `C_i` with `panda_kinematics.commanded_tcp_poses`. It converts that command to a desired held-object pose using only the causal chunk grasp transform:

`P_i* = C_i * G`.

It then emits

`[translation(T_hole^-1 * P_i*) in metres, rotation_6d(T_hole^-1 * P_i*), native gripper scalar]`.

At deployment, `decode_action` projects the sampled 6D orientation to SO(3), forms

`P_world* = T_hole * P_hole*`

and reconstructs the TCP target by

`C_world* = P_world* * G^-1`.

For encoded demonstration labels these operations algebraically recover the original commanded TCP pose before IK. The decoder calls only `panda_kinematics.solve_ik(C_world*, current_observation['qpos'], robot)` from freshly measured joints, appends the unchanged gripper scalar, and reports convergence, position/rotation residuals, iterations, limit contacts, maximum joint change, posture diagnostics, and target world TCP/peg poses. It never hides or replaces an unconverged result. A deterministic orthogonal fallback is used only for a degenerate sampled 6D vector.

## Model and gradient path

`policy.py` uses the published `DiffusionBackbone` unchanged. The two feature frames are flattened into its global condition, and it predicts epsilon for the 10-dimensional desired-peg action sequence. The framework's masked diffusion loss is the only active objective. `prior_loss` is a differentiable zero; there is no auxiliary head. Every learned parameter is registered in the backbone and is therefore included in optimization, EMA, and saved state.

The standard backbone is retained because the desired invariance is completely implemented by causal inputs and an invertible Cartesian label/decoder transform; no architecture change is required.

## Normalization and temporal semantics

Inputs use fixed physical scales rather than evaluation-fitted statistics: general relative translations use 0.6 m; grasp translation uses 0.2 m; arm position/velocity use 3; finger position uses 0.04 m and finger velocity 0.2 m/s; lid angle uses 1.85 rad and lid velocity 4 rad/s; clearance uses 0.5 m; coupling distance uses 0.2 m. Rotations use 6D matrix columns. The framework fits the action representation midpoint/half-range normalizer only on valid authorized represented labels.

Prediction slots are fixed at t-1 through t+14. The host executes slots 1-8, beginning at current t. All actions in every `[1120,1500)` binding are supervised with framework masks. Incoming `[1120,1270)` and outgoing `[1380,1500)` are substantial action-supervised overlaps, not observation-only context.

## Calling, selection, and handoff

The call argument is the empty closed-schema object. Prefer h04 when the cover is open, fingers are closed, peg and hole poses are reliable, and peg-to-TCP geometry is stable across both causal frames. A moved but reachable hole or a changed rigid grasp offset is the intended selection cue.

Use h01 when peg pose or grasp-offset estimation is unreliable but hole/TCP observations are reliable. h02 remains appropriate when displacement of the held source dominates with the nominal destination. Preserve causal history across switches. Hand off to `align_and_insert_peg` when coupling remains stable, orientation is near the hole axis, hole-relative y/z error is small, descent is near hole height, and IK diagnostics remain healthy. Since all original policies could leave a transient pass before final scoring, timely overlap handoff is preferred over untrained indefinite continuation.

## Assumptions and limitations

Closed fingers and stable geometry are only grasp proxies. The causal `G` must remain valid for one eight-action chunk; slip invalidates the reconstruction. Hole-frame covariance cannot preserve box/lid collision clearance, arm reachability, or insertion contact under arbitrary fixture motion. Moving only the hole changes the required source-to-destination route, which still must be inferred from narrow training support. The policy cannot recover a dropped peg, reacquire a bad grasp, undo collision, reopen a collapsed cover, make an unreachable target feasible, or guarantee stability after the demonstrated early-insertion endpoint.
