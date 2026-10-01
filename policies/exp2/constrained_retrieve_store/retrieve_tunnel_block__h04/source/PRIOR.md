# Dual-reference clearance-blend Cartesian prior

## Decision and targeted failure

This independent policy targets displacement of the approached block beyond the narrow demonstrated initial x/y range without using a moving object SE(3) frame for the entire skill. It predicts every commanded TCP target in two redundant forms:

1. **Low-motion candidate:** world-axis TCP translation relative to the fresh observed block position, plus absolute world TCP orientation.
2. **Post-clear candidate:** the full TCP pose relative to the fresh observed `target_pose` destination.

The decoder uses the first candidate while the block bottom is below the 0.16 m tunnel roof. Once the bottom is above the roof, it continuously blends to the destination candidate over 0.04 m. Thus translating the low block translates approach, grasp alignment, horizontal extraction and early-lift TCP positions by construction, while block rotation does not rotate those tunnel-constrained motions. Reorientation and transfer become destination anchored after clearance.

This design is motivated by an observed validation contrast, not a causal diagnosis: h01 passed 5/12 in-distribution runs and reported nonzero IK-unconverged steps in 10 runs, whereas h02 and h03 each passed 12/12 and reported none. The available validation has no out-of-distribution trials and cannot establish why h01 failed. Restricting block anchoring to the phases that need displacement equivariance is an unconfirmed, falsifiable expected benefit.

## Demonstration support

Initial block support is narrow: x is approximately -0.44785 to -0.43650 m, y is -0.40763 to -0.39511 m, and z is about 0.08 m. Entry TCP and joints are effectively fixed. In demo12100, the TCP aligns near the block at 540, fingers are closed by 560, the block has moved to x about -0.613 m while still low at 600, its centre rises from about 0.151 m at 640 to 0.210 m at 660, and transfer begins after 680. Demo12101 shows the same relationship with a different initial block position and somewhat different lift timing. These observations support the chosen phase relationship but do not demonstrate recovery or large displacement.

## Causal inputs and learned modules

`build_inputs` emits two causal 72-channel frames. Each contains measured qpos/qvel, finger width, world TCP and block poses, destination-to-block and block-to-TCP transforms, drawer position/velocity, object centre and bottom clearance, four tunnel-boundary margins, the analytic destination blend weight, and two-frame object/TCP displacement and coupling features. Physical fixed scales are used only for conditioning. Measured joints and world/tunnel margins deliberately retain Panda configuration, reachability and collision dependencies rather than treating the whole problem as translation invariant.

`DualReferenceDiffusion` flattens the two frames into a 144-D global condition and uses the unchanged public `DiffusionBackbone` with 19 action channels. All trainable parameters are registered in that backbone and are therefore optimized, EMA-tracked and saved by the framework. There is no auxiliary head. The masked epsilon diffusion loss is the complete learned objective; `prior_loss` is a differentiable zero. Future states never enter deployment inputs, and no augmentation is used.

## Action labels

For each valid slot, `encode_targets` obtains the demonstrated native joint command's verified FK TCP pose. It emits:

- channels 0:3: commanded world TCP position minus the contemporaneously observed block world position, in metres;
- channels 3:9: commanded absolute world TCP rotation as its first two rotation-matrix columns;
- channels 9:12: commanded TCP translation in the contemporaneous destination frame, in metres;
- channels 12:18: commanded destination-relative rotation in 6D;
- channel 18: the demonstrated native gripper scalar.

Both pose branches encode the same command on every supervised slot. This avoids a future-dependent branch mask: even if actual execution crosses clearance earlier or later than the demonstrated prediction, both sampled candidate channels are available. The framework fits representation normalization only from these valid 19-D labels; the original native-joint action scales are not reused.

## Decode and clearance blend

At each executed slot, `decode_action` resolves fresh `object_pose`, `target_pose` and measured joints. It reconstructs the object candidate by adding channels 0:3 to the block position and projects channels 3:9 to a rotation. It reconstructs the destination candidate by composing channels 9:18 with `target_pose`.

Let

`c = object_center_z - object_half_m - tunnel_clearance_m`.

The destination weight is `clip(c / 0.04, 0, 1)`. Candidate positions are linearly blended. Candidate rotations are blended along the SO(3) relative rotation from the object candidate to the destination candidate. On exact demonstration labels the two world candidates are identical, so the blend preserves the commanded pose at every height. The blend is a reference conversion, not a waypoint, contact detector or task-completing script.

The resulting world TCP pose is passed exactly once to `panda_kinematics.solve_ik` from fresh measured qpos. The returned seven absolute Panda joint targets are followed by channel 18; nonnegative opens and negative closes. Diagnostics report both candidate poses, position/rotation disagreement, bottom clearance, blend weight, IK convergence, task residuals, iterations, limit contacts, joint change and posture status. A safe finite 6D projection handles degenerate sampled rotations but does not alter valid labels.

## Calling and temporal contract

HL calls with `{object_pose_field: "object_pose", destination_pose_field: "target_pose", tunnel_clearance_m: 0.16}`. All arguments enter conditioning or decoding. History is two causal frames, with episode-start duplication owned by the framework and history retained across policy switches. Prediction slots are t-1 through t+14; execution uses slots 1 through 8, beginning at current t, at 20 Hz.

All twelve bindings supervise [300,840). This retains 140 predecessor-overlap actions [300,440) and 160 placement-overlap actions [680,840) per trajectory.

## Applicability and handoff

Prefer h04 when reliable block translation is the principal deployment change and the destination remains observed. Enter with the drawer open/stable, a low block pose, and an open gripper free of the handle; a mid-skill start also requires plausible object/TCP/finger geometry. Hand off after the block is outside the tunnel, its bottom is above the roof, closed-finger object/TCP motion remains coupled, the destination branch dominates, and transfer toward the drawer has begun.

Existing h02 is preferable when a valid unusual predecessor TCP or redundant configuration is the main variation with a near-nominal block. Existing h03 is preferable when its explicit conservative low-clearance transfer guard is needed. Closed fingers or blend weight alone do not prove grasp or successor readiness.

## Limitations and expected failures

No available result verifies out-of-distribution displacement. Fresh block translation may produce unreachable or colliding targets when a displaced block is incompatible with the fixed tunnel and Panda base. The world-orientation low branch does not adapt to a rotated block. Object-height noise can select the wrong blend, and the 19-D diffusion can produce disagreeing candidates, especially near the transition. The destination branch depends on a correct `target_pose`.

Expected failure signatures are large candidate disagreement, premature destination-directed motion below clearance, an abrupt target in the blend band, fingers closing without object/TCP coupling, failure to extract or lift, a drop, drawer disturbance, or persistent IK nonconvergence/limit contact. No contact truth or failed-grasp demonstrations support retry. Physical reachability, collision freedom and stable grasping remain assumptions, and the supplied success definition is unchanged.
