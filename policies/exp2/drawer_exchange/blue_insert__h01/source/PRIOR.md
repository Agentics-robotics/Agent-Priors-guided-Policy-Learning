# Blue-frame equivariant Cartesian targets

## Decision and targeted failure

This policy targets source-block displacement and rotation beyond the narrow demonstrated pickup range. Across the twelve starts, blue is only observed around x=-0.414..-0.387 m and y=0.287..0.315 m. The learned target is therefore a TCP pose in the **causal chunk-start blue pose frame**, not an absolute world pose. Translating or rotating the observed blue pose transforms the decoded pickup, close, lift and early-carry target by construction.

This is an expected generalization benefit, not evidence of recovery. The demonstrations show only successful nominal grasps and insertions.

## Inputs and calling contract

The required typed arguments are `source_object="blue"` and `destination_field="blue_goal"`. They select `blue_pose` and the calibrated drawer goal; arbitrary coordinates and actuator commands are not accepted.

For each of two causal observations, the 44 fixed-scale features contain Panda qpos/qvel (including fingers), TCP pose relative to blue, blue-goal displacement in the blue frame, blue world pose/orientation, red-to-red_goal residual, and drawer position/velocity. World blue pose and joint state deliberately retain robot reachability/configuration dependence. At an episode start only, the framework duplicates the first observation; history otherwise crosses policy switches.

## Action representation and conversion

Each of 16 slots is `[position_m(3), rotation_6d(6), gripper(1)]` in the blue frame frozen when the chunk is sampled. `encode_targets` obtains the Cartesian TCP pose reached by each demonstrated native joint command, applies `inverse(T_world_blue) @ T_world_command`, and retains the native gripper scalar. Slot alignment remains t-1 through t+14; slots 1..8 execute at t..t+7.

`decode_action` projects rotation 6-D to SO(3), computes `T_world_blue @ T_blue_target`, and calls only `panda_kinematics.solve_ik` from freshly measured joints. It reports convergence, position/rotation residuals, iterations, joint-limit contacts and maximum joint change. The source-frame roundtrip is exact before numerical IK; nonunique IK is judged in task space.

## Model and gradients

The unchanged standard `DiffusionBackbone` receives the flattened 2x44 condition and diffuses 10-D actions. There is no auxiliary loss: `prior_loss` is differentiable zero and the masked epsilon diffusion loss is the full objective. All input scaling is fixed in physical units; representation normalization is fitted by the framework only from valid transformed labels. No augmentation is used.

## Evidence, coverage and handoff

In demo1000 the free blue remains near (-0.401,0.291) through approach, the TCP descends by 800, closure appears by 820, and blue/TCP co-move through 840-850. The package preserves all twelve [600,stop) cuts and the 3,000 shared [600,850) actions with `red_transfer`, rather than treating overlap only as context.

Prefer this model when the drawer is open and source displacement/rotation dominates. Once a held blue block approaches the drawer and precise goal anchoring dominates, prefer the complementary destination-frame model. Completion is unchanged: drawer open AND red on pad AND blue inside at one observation.

## Limitations

Blue-relative coordinates do not make the arm, route or contact invariant. A shifted source may be unreachable or collision-prone. The reference is frozen for an eight-action execution chunk, so slip is incorporated only at the next replan. Proximity and finger closure do not prove grasp. There are no demonstrated drops, re-grasps, drawer-wall impacts or recoveries.
