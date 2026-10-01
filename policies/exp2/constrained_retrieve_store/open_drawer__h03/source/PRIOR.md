# Phase-routed dual-anchor Cartesian prior

## Decision and target failures

Opening and departure have different spatial anchors. The drawer contact path should follow the rigid drawer reference, but the overlap should follow the independently placed tunnel block. This policy therefore learns two coordinate versions of every commanded TCP pose and causally chooses between them at execution.

Observed evidence supports the relationship. At index 420, commanded TCP minus block position is about (-0.1400, 0, +0.2500) m in `demo12100`, `demo12101` and `demo12102`, despite different block x/y. Earlier actions through index 340 are shared while `target_pose` translates with the drawer pull. In `demo12100`, index 265 has a fully open drawer but measured finger width about 0.014 m; by 270 width is about 0.0746 m. This supports using drawer openness, stability and released fingers as causal routing cues. It does not prove contact or recovery.

## Inputs, argument and route

Required `open_stop_m` is 0.300 m in every binding. Each of two causal frames has 58 features: drawer progress/velocity, measured finger width, causal route flag, world drawer-reference pose, object relative to drawer, TCP relative to each anchor, and all joint positions/velocities. This keeps anchor geometry, robot configuration and reachability dependencies.

The object route is selected only when fresh observations satisfy all three conditions:

1. `drawer_position >= 0.98 * open_stop_m`;
2. `abs(drawer_velocity) <= 0.02 m/s`;
3. measured finger width is at least 0.06 m.

Otherwise the drawer route is selected. Routing is recomputed for each executed slot and never advanced during denoising. No future state or contact truth is used.

## Actions and conversion

Each 19D slot contains:

- commanded pose relative to current slot `target_pose`: position 3 + rotation 6D;
- the same commanded pose relative to current slot `object_pose`: position 3 + rotation 6D;
- demonstrated gripper scalar.

`encode_targets` gets the command pose through verified FK and encodes both exact branches for every valid slot. Thus the model receives diffusion gradients for both branches even when one is not selected in that state. This avoids using an unavailable predicted future phase to interpret a single branch.

`decode_action` uses the fresh causal route, Gram-Schmidt projects the corresponding 6D rotation, composes it with the selected current world anchor and invokes `panda_kinematics.solve_ik` from fresh measured joints. It reports the selected anchor, route values, world target, convergence, residuals, iterations and joint-limit contacts. The gripper sign is unchanged. Normalization is fitted only on valid dual-anchor labels; padding is masked.

## Model and gradients

The unchanged standard `DiffusionBackbone` conditions on 116 flattened values and diffuses 16 by 19 actions. All trainable modules are registered and optimized by masked epsilon loss. There is no auxiliary head or augmentation; the reported prior loss is a connected zero.

## Displacement benefit, coverage and handoff

The early target branch makes handle approach, pull, release and clearance transform with drawer displacement. The late object branch makes post-release retreat/turn/descent, especially the complete supervised 300:440 overlap, transform with block displacement by construction. These are expected benefits outside the narrow demonstrated positions, not observed deployment results.

Prefer this policy when both anchors are reliable and block displacement or late handoff is important. Switch to `retrieve_tunnel_block` after object routing is active, the drawer remains open/stable, fingers are open and arm clearance is sufficient.

## Limitations

The hard route can be incorrect for unseen phase orders, such as an already-open drawer with open fingers while the TCP is still at the handle; the fresh-TCP residual policy is preferable for small intermediate inconsistencies. Nineteen-dimensional redundant output is harder than a single frame. Anchor noise, large displacement, collision geometry and IK reachability can still cause failure. Finger width is not contact truth, and demonstrations do not establish recovery from a missed handle, premature release or disturbed block.
