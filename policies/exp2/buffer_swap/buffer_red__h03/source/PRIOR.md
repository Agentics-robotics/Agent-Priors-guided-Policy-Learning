# Buffer-Destination-Frame Diffusion Policy

## Decision and target failure

This independent policy targets buffer-location change and precise red lowering/release. Every commanded Cartesian TCP target is represented in the high-level supplied buffer pose. Carry endpoint, lowering, opening and retreat therefore move with the destination by construction. Acquisition is not declared invariant to red displacement; red-to-buffer and TCP-to-red relations remain learned.

Prefer this model when the free buffer is intentionally moved or when red is already held/near the center and placement/release accuracy dominates. Use `buffer_red__h01` when red displacement dominates, and `buffer_red__h02` when intermediate phase/TCP entry is ambiguous.

## Evidence and expected benefit

In inspected successful runs demo10100, demo10103 and demo10105, differently sampled red starts converge near (-0.181,0,0.020). Around 260-280 the commanded TCP is low near (-0.190,0,0.023), opens, then retreats around 300-340. Red remains fixed while the tail acquires and lifts blue around 420-479. This is observed placement and overlap support, not evidence for a relocated buffer or failure recovery.

Expected benefit is equivariance of the destination-local command sequence to a changed supplied buffer pose, particularly carry endpoint through retreat. It is falsifiable under reachable destination changes. A moved destination still changes Panda posture, collision geometry and target-to-destination travel; world and robot features preserve those dependencies.

## Inputs and arguments

The required arguments are `target_object="red"`, `successor_object="blue"`, and `buffer_pose_world=[x,y,z,qw,qx,qy,qz]` in world metres/wxyz. The authorized demonstrations bind `[-0.181,0,0.02,1,0,0,0]`. The object names select current structured observations; the destination pose directly defines all action conversion and relational features.

Each of two causal frames contains 57 values: scaled qpos/qvel/fingers; world TCP and destination positions; destination-relative full poses for TCP, red and blue; and direct TCP-to-red/TCP-to-blue vectors. Position scales are 0.5 m, arm qpos 3 rad, qvel 2.5 rad/s and fingers 0.04 m. Rotational input uses continuous 6D columns. These are fixed unit scales, not evaluation-fitted statistics. qpos/qvel and world positions intentionally retain reachability, joint-limit and posture information.

## Action representation and conversion

The seven learned values are destination-local translation in metres, local SO(3) rotation vector in radians, and the native gripper scalar. `encode_targets` uses verified FK command TCP poses and computes `inverse(T_buffer) @ T_command`, then applies the SO(3) logarithm. `decode_action` uses the exponential map, composes `T_buffer @ T_local`, and calls only `panda_kinematics.solve_ik` from freshly measured qpos. Convergence, position/rotation errors, iterations, joint-limit contacts and maximum joint change are returned. No alternate controller or destination-completing script exists. The framework fits representation normalization only on valid destination-frame labels.

The standard DiffusionBackbone is unchanged. Masked epsilon diffusion is the full objective and `prior_loss` is differentiable zero; there are no extra learned modules or auxiliary labels.

## Temporal scope and handoff

History 2, horizon 16, execution 8 and 20 Hz remain fixed. Slots are t-1..t+14 and execution starts at slot 1. Every assigned action in all 12 occurrences is retained. The substantial 210-230 action overlap includes red lowering/release, opening, retreat, transit, blue closure and initial lift. Switch readiness requires red visibly stable at the actual supplied destination and open/clear fingers; a later blue-lift switch is also supervised. Preserve causal history.

## Limitations

A moved buffer can be occupied, unreachable, near a joint limit or create unsupported collision geometry; the representation cannot make those conditions safe. Red acquisition may fail when red is displaced beyond demonstrated target-to-buffer relations. Rotation-vector labels assume the demonstrated orientation branch remains suitable. There is no image, contact truth, collision model, or demonstrated recovery from missed grasp, drop or bad release. IK diagnostics indicate numerical feasibility only. The unchanged full-task success definition is not modified by this skill policy.
