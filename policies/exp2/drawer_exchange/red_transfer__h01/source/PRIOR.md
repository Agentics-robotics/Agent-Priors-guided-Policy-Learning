# Red-object equivariant Cartesian prior

## Decision and target failure

This policy targets displacement of the manipulated red block beyond the narrow demonstrated pickup range. Every commanded TCP target is learned in the SE(3) frame of the red pose from the newest causal observation at chunk construction. Translation or rotation of that observed red frame therefore produces the corresponding world-target transformation before IK. The direct invariant is most relevant to red approach, contact, grasp, lift, and carried-red motion; it is not claimed for the later blue pickup.

Observed support is distinct from the expected benefit. In `demo1000`, the TCP is above red by index 400, closes at 445, and red/TCP co-motion is visible at 470 and 500 before transport near the goal at 560 and release by 610. `demo1001` begins from another red XY and follows the same relationship. The demonstrations do not show large red displacement or failed-grasp recovery, so performance there is an expected consequence of the representation rather than observed recovery evidence.

## Inputs and calling contract

The required identifiers are `manipulated_object="red"`, `placement_target="red_goal"`, and `handoff_object="blue"`. They select current fields rather than supplying poses or code. At every replan, `build_inputs` resolves them against both causal observations and freezes the newest red pose only for the resulting chunk.

Each of the two 44-D feature rows contains scaled measured Panda joints, fingers and velocities; TCP pose relative to red; blue pose relative to red; red-goal and blue-goal points relative to red; and drawer position/velocity. Joint state intentionally preserves configuration and reachability information even though scene geometry is relational. The 3-D intent vector records the closed typed selections. Initial episode history may be duplicated by the framework; across skill switches the real two-frame causal history is retained.

## Action representation and converters

The 10-D action is red-frame TCP translation in metres, relative TCP rotation as the first two rotation-matrix columns, and the demonstrated gripper scalar. `encode_targets` computes TCP poses by FK of the demonstrated seven-joint commands, then applies `inverse(chunk_red) * commanded_tcp`. It never uses future observations. The framework fits action normalization on these valid labels.

`decode_action` safely orthonormalizes the sampled 6D rotation, composes the relative target with the same causal red anchor, and calls `panda_kinematics.solve_ik` from freshly measured arm joints. It appends the represented gripper scalar unchanged; the common executor interprets nonnegative as open and negative as close. Diagnostics publish convergence, position/rotation residuals, iterations, limit contacts, and maximum joint change. Gram-Schmidt projection is exact on encoded valid rotations; arbitrary degenerate samples receive a deterministic finite fallback.

## Model and gradients

The model is the unchanged standard `DiffusionBackbone` with flattened two-frame features and intent as global conditioning. There are no auxiliary modules or labels. Masked epsilon diffusion loss is the total loss, and `prior_loss` is a differentiable zero. Thus all trainable parameters receive their intended action-diffusion gradient and are covered by optimizer, EMA, and checkpoint handling. Training keeps the fixed H2/16/8, 20 Hz, DDPM, batch, optimizer, seed, and 20,000-update recipe.

## Supervision and handoff

All twelve authorized `[230,850)` bindings are retained. Consequently `[230,470)` remains action-supervised overlap with `open_drawer`, and `[600,850)` remains overlap with `blue_insert`. The latter deliberately includes held-red descent/release before retreat and blue acquisition. Nominal successor readiness is red left on the pad plus blue/TCP co-motion on initial lift; release and closed fingers are transition cues, not additions to the completion contract or proof of contact.

## Applicability and limitations

Prefer this model when the drawer is open, red is exposed and reliably localized, and red displacement is the dominant variation. Exact equivariance is frozen for one chunk; it does not cover red motion within the chunk, stale pose estimates, changed furniture, transformed targets outside Panda reach, collision, or altered grasp physics. After red release the blue transition remains expressed relative to red and is not blue-displacement invariant. Use the destination-centered policy for moved-pad placement and the phase-conditioned local policy for uncertain intermediate or post-release entries. No successful demonstration establishes recovery from an unseen miss or off-pad release.
