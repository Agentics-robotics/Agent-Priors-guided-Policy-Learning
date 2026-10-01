# Hybrid drawer-frame / fresh-TCP Cartesian prior

## Decision and targeted failure

This policy targets two deployment changes with one consistent representation: displacement of the drawer reference during approach and pulling, and small intermediate-state or continuation mismatch near the open stop. It redundantly encodes the demonstrated command position in the current `target_pose` frame and as a correction from the measured TCP. A causal progress blend uses the drawer-relative position through approach and most of the pull, then changes smoothly to the fresh-TCP position near the open stop. Command orientation is always a fresh-TCP body correction.

The displacement claim is structural and scoped. Up to 90% drawer progress, composing the learned target-frame position with a freshly observed translated or rotated `target_pose` moves the commanded position with that reference by construction. Between 90% and 98% progress, the decoder smoothly interpolates the two position reconstructions; at and above 98%, it uses the fresh-TCP reconstruction. Orientation is not made fully drawer-frame equivariant. Relative coordinates also do not make reachability, collision geometry or contact invariant.

## Evidence and expected benefit

In `demo12100`, `target_pose.x` moves from 0.125 m at index 0 to -0.04172 m at 220 and about -0.175 m at 250. Commanded TCP x changes from about -0.1478 m at handle contact to -0.3222 m at 220 and -0.4476 m at 250, retaining an approximately fixed drawer-relative handle relation during pull. The gripper command changes to close at 165 and to open at 265; indices 300:440 provide the departure overlap.

The in-distribution validation is also relevant but narrower than deployment. The existing fresh-TCP residual policy h02 passed all 12 demonstrated entries and ended near the open stop in every run. The fully drawer-relative h01 passed 6/12; several failures first reached or nearly reached the threshold and later ended below it. These facts do not establish a cause or an out-of-distribution advantage. They motivate the unconfirmed hypothesis that drawer anchoring is most useful before the stop, while a local correction is useful for near-stop continuation and departure.

## Inputs and argument

The required `open_stop_m` argument is drawer travel in metres and is bound to 0.300 for every training occurrence. It normalizes progress and defines the fixed blend interval. Each of two causal frames contains 48 deterministic features:

- normalized drawer progress and velocity, plus the causal smooth blend value;
- world `target_pose` position and 6D rotation;
- measured TCP pose relative to `target_pose`;
- block pose relative to `target_pose`;
- all nine joint/finger positions and all nine velocities with fixed physical scales.

These retain anchor placement, block relation, current arm configuration, finger state and two-frame motion. The framework duplicates only the initial episode observation and retains causal history across policy switches. No future state, image, action history, contact truth or simulator-only signal is used.

## Action labels and decoder

Each 10D slot is:

1. commanded TCP position in fresh per-slot `target_pose` coordinates, 3 metres;
2. translation of `inverse(measured_tcp_pose) * commanded_pose`, 3 metres;
3. rotation vector of that same measured-TCP body correction, 3 radians;
4. demonstrated native gripper scalar, 1 value.

`encode_targets` obtains commanded poses only from verified FK of demonstrated native joint commands. Both position branches reconstruct the identical demonstrated world command. `decode_action` recomputes the blend from fresh observed drawer progress for every executed slot. It composes the target-frame position with fresh `target_pose`, composes the body correction with fresh measured TCP, blends the two world positions using smoothstep over progress 0.90 to 0.98, and takes orientation from the body reconstruction. Predictions are not accumulated across slots or denoising calls.

The resulting world TCP target is passed only to `panda_kinematics.solve_ik` from freshly measured joints. Diagnostics report blend value, branch disagreement, world target, convergence, position and rotation residuals, iterations, limit contacts, maximum joint change and posture status. The gripper sign is preserved. Framework representation normalization is fitted only on valid transformed labels; padded labels and diffusion loss use the supplied mask.

## Model and gradients

The model uses the unchanged published `DiffusionBackbone` with a flattened 96-value causal condition and action dimension 10. Every trainable parameter is registered in that backbone and receives gradient from the masked epsilon diffusion loss. There is no auxiliary head, augmentation or hidden state. `prior_loss` is a connected zero used only for reporting; it does not replace diffusion learning. The fixed blend belongs to the invertible/redundant Cartesian converter rather than a task-completing script.

## Coverage and handoff

All twelve bindings supervise actions 0:440. Actions 300:440 remain the complete 140-action, 7.0-second overlap with `retrieve_tunnel_block`; the decoder is fully on the fresh-TCP position branch there under demonstrated progress. Hand off after the drawer is approximately 0.300 m open and stable, fingers are open, the TCP has cleared the drawer, and the block remains undisturbed. A switch shortly after release or later above the tunnel is action-supervised.

## Applicability and limitations

Prefer h04 when `target_pose` is reliable, drawer displacement is a concern, and near-stop continuation should be fresh-state-local. h02 remains the evidence-backed in-distribution choice for small state mismatch without a displacement requirement. h03 is preferable when independent block displacement during the departure overlap is central.

The blend has no hysteresis and can be ambiguous with noisy or nonmonotone progress. The target-frame invariant applies to early/pull position, not to orientation or the block-relative tail. Large drawer transformations can be unreachable or collide. Closed fingers do not prove handle contact. No demonstration or validation establishes recovery from a missed handle, premature release, disturbed block, severe tracking error, unseen phase order, collision or IK failure. The expected failure signature is a drawer that never reaches 0.29 m or reaches it and later regresses, possibly with extra gripper sign changes; under large displacement, IK residuals or unsafe geometry may instead dominate.
