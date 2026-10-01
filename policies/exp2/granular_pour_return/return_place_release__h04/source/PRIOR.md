# Fresh source-object-frame SE(3) prior

## Decision and targeted variation

This independent policy targets a securely held source container that reaches the return skill translated or rotated relative to the tightly clustered demonstrated handoff poses, or with a changed but still valid observed source/TCP relation. This is an expected deployment variation, not a measured validation failure: each of `return_place_release__h01` through `__h03` passed all 12 in-distribution validation runs, and no out-of-distribution result is available.

The expected benefit is for source-coupled late righting, carry, descent, and post-release clearance geometry. It does not make the entire fixed-destination placement problem invariant. Destination progress still depends on the observed destination-in-source relation and learned successful behavior.

## Cartesian action mechanism

For every supervised slot, `encode_targets` obtains the demonstrated commanded world TCP pose from the native arm command using verified public FK. It then uses the observed container pose immediately before that action and encodes

`T_source_before_action^-1 * T_commanded_tcp`

as source-frame translation in metres, continuous 6D rotation (the first two columns of the relative rotation matrix), and the demonstrated native gripper scalar. The framework fits the representation normalizer on valid 10D labels in these actual coordinates.

For every executed slot, `decode_action` resolves the **fresh current** observed container pose, orthonormalizes the represented 6D rotation, and reconstructs

`T_world_command = T_world_source_fresh * T_source_command`.

It then calls `panda_kinematics.solve_ik` from freshly measured joints and appends the learned gripper scalar. There is no integration against a predicted pose, scripted return path, contact gate, or release rule. IK convergence, position and rotation residuals, iterations, limit contacts, maximum joint change, and posture diagnostics are returned explicitly.

This converter makes a fixed represented source-relative TCP target transform with displacement of the observed source by construction. The intended invariant is the local grasp/righting/carry relation and source-relative clearance, not global reachability or fixed-goal placement under arbitrary shifts.

## Causal inputs and retained dependencies

Each of the two causal rows has 63 deterministic features:

- destination pose expressed in the source frame (9: scaled translation and 6D rotation),
- TCP pose expressed in the source frame (9),
- bowl pose expressed in the source frame (9),
- source world pose (9),
- destination world pose (9),
- seven arm joints plus finger sum/difference (9), and
- seven arm velocities plus finger velocity sum/difference (9).

Relative translations use fixed 0.5 m scales, except the source-relative TCP translation uses 0.2 m; world translations use a 1 m scale. Arm angles use 3 rad, arm velocities 2.5 rad/s, finger width 0.08 m, and finger differential velocity 0.4 m/s. These are fixed unit conversions, not data fitting. Source-relative relations provide the intended geometry, while world source/destination poses and joint state retain robot-base reachability, redundant-posture, and support dependencies. At episode initialization the framework duplicates the first observation; causal history remains available across policy switches.

The required call arguments are `source_object="container"` and `destination="target_pose"`. They select `container_pose` as both the action reference and source feature anchor, and `target_pose` as the goal feature. The same resolution is used in training and deployment.

## Model, loss, and saved modules

The standard `DiffusionBackbone` is used unchanged with a flattened 126D condition and 10D action. The only trainable module is this registered backbone. Masked epsilon diffusion loss is the full training objective; `prior_loss` is a differentiable zero for reporting. Thus all trainable parameters receive the main action-supervised gradient and are included in optimization, EMA, and saved state. No auxiliary objective or future label enters deployment inference.

The standard backbone is sufficient because the prior is implemented in the causal geometry and action converter, not through a new temporal architecture. The fixed numerical recipe remains history 2, horizon 16, execution 8, 20 Hz, 100-step DDPM, 20,000 updates, seed 0, batch 128, and last EMA selection.

## Evidence, responsibility, and overlap

All twelve frozen `[1460, stop)` segments are bound without trimming. This preserves every assigned late-righting, return, descent, release, retreat, and terminal-hold action. In particular, all 280 actions in `[1460,1740)` remain action-supervised in every occurrence, preserving the substantial overlap with `controlled_pour_and_right`.

In demo12300, indices 1460-1680 show the grasped source being righted; indices 1720-1740 show rapid target-directed travel while the TCP remains offset from the source; indices 1800-1840 show descent and low support geometry; index 1850 changes the gripper command to open; and 1880-1920 show retreat. Demo12303 and demo12308 command opening at 1840, showing release-time variation. Final observations place an upright stationary source near `(-0.470, -0.282, 0)` m and a clear TCP near `(-0.460, -0.280, 0.280)` m. Contact is inferred from low source height and stationarity because contact truth is unavailable.

## Handoff and library role

Prefer this policy only when the source is still securely held, its pose is reliable, and the off-cluster source pose or source/TCP relation is the main concern. Accept the demonstrated moderate tilt or an upright outbound state in the shared interval. Continue through observed support, opening, and retreat; terminate from physical observations of upright in-bounds support, source stationarity, open fingers, and TCP clearance.

This policy supplements rather than replaces the validated originals. Prefer h01 when destination displacement is dominant after righting, h02 when fresh TCP increments are safer than trusting a source anchor, and h03 near nominal support when release phase ambiguity dominates.

## Assumptions and limitations

The source remains securely grasped until demonstrated release; source, TCP, destination, and joint observations are accurate; the source is static after release; and IK targets are feasible. Fresh source anchoring can chase a free or noisy container, so increasing source/TCP separation or motion inconsistent with a grasp is a stop/switch signature. Source-frame reconstruction does not guarantee collision avoidance, destination reachability, contact, support, or success.

The demonstrations and available validation contain only successful nominal entries. They do not establish recovery from slip, lost grasp, collision, failed placement, premature opening, recontact, or insufficient pouring. There is no regrasp, corrective repour, placement-recovery, or release-recovery supervision. Expected displacement benefit is therefore falsifiable but unconfirmed until deployment evaluation; no final-test feedback informed this design.
