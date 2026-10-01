# Adaptive interaction-anchor Cartesian prior

## Decision and targeted failure

`close_drawer__h04` targets translation of the block/TCP release interaction at the placement-to-closure handoff. It is not an in-distribution repair: each of the three immutable original policies passed all 12 replay-entry validation runs, and no out-of-distribution comparison is available.

The targeted uncertainty is supported by variation at the common start. Across the 12 demonstrations at index 900, the block height ranges from about 0.063 m to 0.124 m and total finger width from about 0.0365 m to 0.0765 m. In inspected `demo12100`, the block settles by index 920, the TCP clears the cavity through indices 940-960, approaches the handle by indices 1080-1140, closes the drawer after index 1160, opens near index 1260, and is high and clear by index 1329/1330. These observations motivate release-state conditioning but do not demonstrate recovery from a shifted or failed placement.

## Analytic interaction frame and useful invariant

For every state, the adapter resolves the observed world poses of `target_pose`, `tcp_pose`, and `object_pose`. It computes

- `g_open = clip((drawer_position - 0.24 m) / 0.04 m, 0, 1)`;
- `g_near = clip((0.16 m - ||p_tcp - p_object||) / 0.08 m, 0, 1)`;
- `g = g_open * g_near`.

The action anchor uses the orientation of `target_pose` and world translation

`p_anchor = p_target + g * (p_object - p_target)`.

When the drawer is fully open and TCP-object distance is at most 0.08 m, the anchor translation is exactly the observed block translation. Release completion and initial clearance are therefore covariant by construction to a common translation of the block and nearby TCP. Between 0.08 m and 0.16 m the anchor blends smoothly back toward the drawer frame. At a distance of at least 0.16 m, or after the drawer closes below 0.24 m, the anchor is exactly `target_pose`; normal block-position variation then cannot translate the handle approach, closure, release, or retreat.

The invariant is deliberately limited. The block orientation is not used as the anchor orientation because it changes while settling. Robot joints, fixed-base reachability, collisions, contact feasibility, and changed relative drawer geometry do not transform away.

## Causal inputs

`build_inputs` returns two causal frames of 42 values each. Each frame contains:

- target-frame TCP translation and rotation 6D;
- target-frame object translation and rotation 6D;
- scaled nine-component Panda qpos and qvel, retaining both arm configuration and finger state;
- scaled drawer position and velocity;
- target-frame TCP-object displacement;
- the analytic interaction gate.

The scene features are relational under a coherent rigid displacement, while qpos/qvel retain configuration, reachability, velocity, gripper, and joint-limit dependencies. No images, contact truth, future state, action history, or recurrent phase memory are used. The framework preserves two real causal frames across skill switches and duplicates the first observation only at an episode start. There are no high-level call arguments.

## Cartesian labels and decoding

For each prediction slot, `encode_targets` obtains the demonstrated commanded world TCP pose by exact FK with `panda_kinematics.commanded_tcp_poses`. It recomputes the interaction anchor from that slot's pre-action observation and encodes

`T_relative = inverse(T_anchor) @ T_command`.

The 10 learned values are relative xyz in metres, relative orientation as the first two rotation-matrix columns (continuous 6D), and the native gripper scalar. The framework fits the representation normalizer only to valid labels in these coordinates; padded slots remain masked.

At each executed slot, `decode_action` recomputes the anchor from the fresh current observation, projects rotation 6D to SO(3) by Gram-Schmidt, composes the represented pose with the anchor, and sends the resulting world TCP target to `panda_kinematics.solve_ik` from freshly measured qpos. It appends the learned gripper scalar unchanged; the executor maps values greater than or equal to zero to open and negative values to close. The decoder reports the gate components, TCP-object distance, anchor and world target, 6D fallback status, and IK convergence, residuals, iterations, limit contacts, joint change, and posture diagnostics. An unconverged result is returned and reported rather than hidden or replaced.

Prediction slots are t-1 through t+14. The host executes slots 1 through 8, corresponding to current t through t+7. There is no action accumulation, decoder script, or state mutation during denoising.

## Model, objective, and saved modules

The model uses the supplied mature `DiffusionBackbone` unchanged, with the flattened 84-value two-frame condition and a 10D action sequence. Masked epsilon diffusion loss is the only training objective. `prior_loss` is a differentiable zero because the scientific prior is implemented by causal inputs and the Cartesian reference transformation; no artificial auxiliary prediction is added. Every trainable parameter belongs to the registered backbone and is therefore included in optimization, EMA, and checkpoint state.

## Supervision and handoff

All 12 bindings retain every action in `[900,1330)`. The entire `[900,1160)` interval is action-supervised overlap with `place_block_in_drawer`: 260 actions, or 13.0 s, per trajectory. It includes release completion, clearance, retreat and reorientation, handle approach, and handle acquisition. This overlap supports switching at different demonstrated release states but is not evidence of recovery from arbitrary errors.

Enter with the open drawer loaded or while a valid placement is finishing. Prefer h04 specifically when object_pose is reliable, the TCP is near the block, and translation of that early interaction is the principal uncertainty. Later valid invocation is supported, but the gate will normally be zero. Exit guidance remains the unchanged physical condition: drawer position below 0.025 m and nearly stationary, block inside near floor height, gripper open, and TCP high and clear. This is a terminal skill; the gate is neither contact truth nor a success predicate.

## Expected failure signature and limitations

The distinctive failure is incorrect anchor selection. A noisy, occluded, bouncing, or handle-adjacent block can keep the gate active and make the TCP chase or re-contact the block, fail to clear the cavity, or bias handle approach. Conversely, if an early displaced block is already more than 0.16 m from the TCP, the gate is zero and the targeted block-translation covariance is absent. The construction assumes that valid placement keeps the block away from the later handle corridor and that `target_pose` remains a coherent drawer-attached frame.

The policy has no contact truth and no demonstrated recovery from a missed release, block outside the cavity, missed handle, jam, obstruction, collision, changed handle geometry, or unreachable target. Translation covariance does not guarantee IK feasibility under a fixed robot base. Prefer the existing moving-drawer-frame policy for coherent fixture displacement, the TCP-local policy for realized-hand mismatch, or the fixed-destination policy for partial-closure and stopping ambiguity. Those preferences and the expected benefit of h04 are unconfirmed deployment hypotheses, not conclusions from the in-distribution validation.
