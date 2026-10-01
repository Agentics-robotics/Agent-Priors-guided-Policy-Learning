# Drawer-frame absolute Cartesian prior

## Decision and target failure

World waypoints can overfit the single demonstrated drawer placement. This policy learns every commanded TCP pose in the freshly observed `target_pose` frame. It targets displacement or rotation of the drawer reference during handle approach, pulling, release and initial clearance (approximately actions 0:340). This gives translation/rotation equivariance by construction while retaining current joints and the anchor's world pose for reachability.

Observed evidence is relational, not a displaced-scene trial. In `demo12100`, `target_pose.x` moves from 0.125 m at index 0 to -0.04172 m at 220 and about -0.175 m at 260. The corresponding commanded TCP is near -0.1478 m at handle index 160, -0.3222 m at 220 and -0.4452 m at 260, maintaining a drawer-frame pulling relation. Indices 280:340 show release and clearance. Expected robustness to unseen drawer displacement is a consequence of the converter, not an observed success result.

## Inputs and high-level argument

The required argument `open_stop_m` is in metres and is 0.300 for all bindings. It normalizes drawer progress. Each of two causal frames produces 47 deterministic features:

- drawer progress and velocity;
- world `target_pose` position/6D rotation;
- measured TCP pose relative to `target_pose`;
- block pose relative to `target_pose`;
- all nine measured joint positions and velocities, with fixed physical scales.

These retain robot configuration, finger state, world placement and block relation. Initial episode history is framework-duplicated; the same builder is used after switches. No image, contact truth, future state or action history is used.

## Action and converters

`encode_targets` applies verified FK to each demonstrated seven-joint command. It returns ten values per slot: target-frame position in metres, target-frame rotation as the first two matrix columns (6D), and the demonstrated gripper scalar. The frame is resolved separately from each slot's pre-action observation, including during drawer motion.

At execution, `decode_action` Gram-Schmidt projects 6D rotation to SO(3), composes it with the fresh current `target_pose`, and passes that world TCP pose to `panda_kinematics.solve_ik` from fresh measured joints. IK convergence, residuals, iterations and joint-limit contacts are reported. The gripper sign is preserved. Representation normalization is fitted by the framework only on valid transformed training labels. Padded labels remain masked.

## Model and gradients

The model is the unchanged published `DiffusionBackbone` with a flattened 94-value two-frame condition and action dimension 10. All trainable parameters are registered in that backbone and receive gradients from masked epsilon diffusion loss. There is no auxiliary head or auxiliary loss (`prior_loss` is a reported zero connected to diffusion loss). No augmentation is applied.

## Coverage and handoff

Every authorized occurrence supervises actions 0:440. Actions 300:440 are deliberately retained action-supervised overlap with `retrieve_tunnel_block`, covering retreat, turn and initial descent. Hand off when the drawer is about 0.300 m open and stationary, fingers are open, and the TCP is clear; a later switch over the tunnel is also supported.

## Applicability and limitations

Prefer this policy when `target_pose` is reliable and the drawer placement is the principal variation. It can fail when the independently placed block moves because its entire tail remains drawer-relative; the dual-anchor policy covers that case. The fresh-TCP residual policy is preferable for small intermediate tracking mismatches. Assumptions are a rigid drawer reference, valid causal observations and reachable collision-feasible composed poses. Closed fingers are not proof of contact, and the data do not demonstrate recovery from a missed handle, premature release, phase error, collision or IK failure.
