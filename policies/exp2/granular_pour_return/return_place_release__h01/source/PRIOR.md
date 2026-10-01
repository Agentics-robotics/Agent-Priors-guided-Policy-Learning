# Destination-frame SE(3) return prior

## Decision and targeted failure

This independent policy targets displacement of the return destination relative to the single demonstrated `target_pose`. The demonstrations use one destination near `(-0.46, -0.28, 0)` m, so a world-absolute policy can memorize that location. The expected benefit is limited to the upright, target-directed transport, descent, placement, release, and retreat portion. The full assigned slice is nevertheless supervised so responsibility and predecessor overlap are unchanged.

## Mechanism

`encode_targets` obtains each demonstrated commanded TCP pose from the native joint command with the verified public FK and encodes

`T_destination^-1 * T_commanded_tcp`

as destination-relative translation (metres), 6D rotation (the first two rotation columns), and the demonstrated gripper scalar. `decode_action` orthonormalizes the 6D rotation, computes

`T_world_command = T_world_destination * T_destination_command`,

then calls `panda_kinematics.solve_ik` from freshly measured joints. This construction makes the Cartesian target move with translation or rotation of the observed destination. The destination pose is resolved from the latest causal observation at every chunk replan and fixed only for that eight-action chunk.

The two-frame input contains destination-relative source, TCP, and bowl poses; source-relative TCP pose; measured arm/finger configuration and velocity; and the destination world pose. Relative features provide the invariant, while joint state and world goal retain reachability, redundancy, and robot-base dependencies. Position channels receive fixed unit conversions only; fitted normalization is performed by the framework on the actual 10D labels. There is no augmentation and no auxiliary loss.

## Model and gradients

The standard `DiffusionBackbone` is used unchanged with a flattened 126D causal condition and a 10D action. The masked epsilon diffusion loss is the entire objective, so all backbone parameters receive action-supervised gradients. The standard architecture is sufficient because the intended prior is in the coordinates and converter, not a new temporal model.

## Calling contract and evidence

The required symbolic arguments are `source_object="container"` and `destination="target_pose"`; adapters use them to select `container_pose` and `target_pose`. All twelve frozen `[1460, stop)` bindings are included. Thus every assigned action is labeled, including the 280-action `[1460,1740)` overlap with `controlled_pour_and_right` and terminal stationary actions.

Observed support: demo12300 is tilted at 1460, upright near 1680, near the destination by 1740-1760, descends through 1820-1840, opens by 1850, and retreats by 1920. Demo12303 and demo12308 open at 1840, showing release timing variation. These are successful examples, not evidence of recovery.

## Handoff and selection

Prefer this policy for a securely held, upright or nearly upright source and a reachable destination displacement. Continue until observed upright support in the return region, open fingers, and TCP clearance. Use the fresh-TCP incremental policy for a substantially tilted or unusual intermediate entry. Use the phase-regularized world policy near nominal setdown when release timing is the main ambiguity.

## Assumptions and limitations

The source remains grasped until demonstrated release; object and target observations are accurate; the target is static during each chunk; and IK is feasible. Only one nominal target and successful paths are available. Destination equivariance does not prove collision avoidance or reachability under large shifts. It also transforms early righting commands, so no displacement benefit is claimed for strongly tilted entry. There is no contact truth, corrective repouring, regrasp, placement-failure, or release-recovery supervision. IK convergence, residuals, and limit contacts are reported rather than hidden.
