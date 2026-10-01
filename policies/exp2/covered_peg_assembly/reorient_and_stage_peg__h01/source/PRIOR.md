# Destination-frame SE(3) prior

## Decision and targeted failure

This independent policy targets the most likely fixture-displacement failure: all twelve demonstrations use essentially one `hole_pose`, while deployment may move the square-hole fixture. The policy learns commanded TCP poses in the **current observed hole frame**, not at demonstration world coordinates. This construction covers lift/transport globally and is strongest for the approach, final reorientation, y/z centering, descent, and early +x insertion retained in this skill.

Observed support is not a recovery claim. In `demo12200`, the peg rises from about z=0.032 m at 1120 to z=0.428 m at 1240, reaches the hole neighborhood around 1380, and descends to about z=0.10 m before insertion. At 1380, `demo12205` remains laterally offset near y=0.160 m whereas `demo12208` is near y=0.180 m. This supports different successful handoff timing, but no demonstration varies the hole position.

## Implemented mechanism

`build_inputs` uses both causal frames. In each frame it expresses TCP, peg, target, and lid poses relative to that frame's observed hole, using translation plus a continuous 6D rotation representation. It appends scaled measured `qpos`, `qvel`, lid position/velocity, and peg clearance above the box wall. Thus destination-relative geometry is emphasized, while robot configuration, finger state, obstacle state, velocity, and reachability dependence remain visible.

At each replan, `chunk_context.anchor_pose` stores the newest causal `hole_pose`. `encode_targets` obtains the exact demonstrated commanded TCP poses by Panda FK and encodes every slot as:

`[T_hole^-1 * T_command translation (m), relative rotation 6D, native gripper scalar]`.

`decode_action` projects 6D orientation to SO(3), composes the represented target with the same chunk hole anchor, and calls the supplied `panda_kinematics.solve_ik` from freshly measured joints. It does not substitute another controller or alter an unconverged result. Diagnostics expose convergence, position/rotation residuals, iterations, joint-limit contacts, maximum joint change, and the decoded world TCP target. The gripper scalar is unchanged.

A rigid transform applied to the observed hole and desired route is therefore applied to the decoded Cartesian target by construction. This is an expected benefit under reachable fixture displacement, not evidence of task success under that change. Because measured joints are retained, the learned policy may still respond differently when the transformed route changes arm feasibility.

## Model and gradient path

`policy.py` uses the published `DiffusionBackbone` unchanged. The two `[2,57]` feature frames are flattened to the global condition. The 10-dimensional action diffusion objective is the only loss; `prior_loss` is a differentiable zero and there is no auxiliary head. All backbone parameters are registered, optimized, EMA-tracked, and checkpointed by the framework. Future states never enter inference.

## Normalization and temporal alignment

Input scaling is fixed physical scaling: relative positions by 0.6 m, arm positions/velocities by 3, finger positions by 0.04 m, finger velocities by 0.2 m/s, lid angle by 1.85 rad, lid velocity by 4 rad/s, and clearance by 0.5 m. No evaluation statistics are fitted. The framework fits the action representation normalizer only from valid authorized labels, including constant/narrow channels.

The framework supplies two frames and duplicates only the initial episode frame. Prediction slots are t-1 through t+14; every `[1120,1500)` action remains supervised with the framework mask. Execution uses slots 1-8, beginning at t. The anchor remains fixed for one sampled chunk and is refreshed from observation at the next replan.

## Calling and handoff

The call argument is the empty closed-schema object. The high-level agent should prefer this policy when `hole_pose` is reliable, the peg remains apparently coupled to the TCP, and the fixture is displaced but reachable. Incoming overlap `[1120,1270)` supports takeover from retrieval; outgoing overlap `[1380,1500)` supports transfer to alignment/insertion over multiple chunks.

Hand off once stable grasp cues, near hole-axis alignment, lateral centering and descent are observed. Stop or switch on peg/TCP drift, gripper opening, lost clearance, growing alignment error, or poor IK diagnostics.

## Assumptions and limitations

The policy has no contact truth. It assumes correct semantic hole observations and a stable held peg. Hole-frame equivariance does not make collision geometry, box/lid placement, arm reachability, or contact dynamics invariant. It does not recover a dropped peg, collision, bad grasp, or unreachable target. Prefer the held-peg anchor policy for a displaced source under a stable grasp, and the local-twist policy for valid but off-timing intermediate states.
