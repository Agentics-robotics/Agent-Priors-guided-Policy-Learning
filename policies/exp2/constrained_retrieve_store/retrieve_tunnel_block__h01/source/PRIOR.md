# Object-frame Cartesian target prior

## Decision and target failure
This independent policy targets displacement of the manipulated block beyond the narrow demonstrated initial range. It learns every commanded TCP target in the **contemporaneously observed block frame**. At execution, each represented slot is composed with the fresh `object_pose`, so the approach, grasp alignment, low pull, clearance lift and early held-object transfer move with a displaced block by construction. This is a local equivariance; the tunnel, drawer and Panda base do not move with the block.

Observed support: demo12100 and demo12101 place the initial block near (-0.4368,-0.4063,0.0800) and (-0.4478,-0.4076,0.0800). Their commanded TCP at index 540 moves correspondingly from about (-0.4367,-0.4019,0.0889) to (-0.4480,-0.4032,0.0888). Both then pull in negative world x at low height, lift above the 0.16 m roof, reorient and start transfer. Expected benefit outside this range is not observed policy success.

## Inputs and modules
`build_inputs` produces two causal 60-channel frames. They contain measured qpos/qvel, drawer state, world object and TCP poses, object-to-TCP and object-to-destination transforms, block position relative to fixed tunnel geometry, and signed roof clearance. Positions use declared fixed physical scales; rotations use 6D columns. The measured joints and world/tunnel features deliberately retain configuration, reachability and collision dependencies.

`ObjectFrameDiffusion` flattens the two frames and supplies the resulting 120-D causal condition to the unchanged public `DiffusionBackbone`. All trainable parameters are in that registered backbone and are optimized and saved by the framework. There is no auxiliary head or auxiliary loss; `prior_loss` is differentiable zero and the masked epsilon diffusion loss remains the complete objective.

## Targets and conversion
`encode_targets` converts each native demonstrated arm command to its verified FK TCP pose, computes `inverse(object_pose_before_action) @ commanded_tcp_pose`, and stores translation (metres), rotation 6D and the demonstrated gripper scalar. This produces 10-D unnormalized labels in the representation actually fitted by the framework normalizer. Futures are used only to prepare supervised slots, never as deployment inputs.

`decode_action` Gram-Schmidt projects rotation 6D, composes the local target with the freshly observed block pose, and calls `panda_kinematics.solve_ik` from freshly measured qpos. The returned absolute seven joint targets are followed by the learned gripper scalar; nonnegative opens and negative closes. IK convergence, position/rotation residuals, iterations, limit contacts and target pose are reported. A finite fallback handles degenerate sampled 6D vectors but does not script task motion.

## Calling and temporal contract
HL supplies `{object_pose_field:"object_pose", destination_pose_field:"target_pose", tunnel_clearance_m:0.16}`. These values select both observed transforms and the clearance feature. History is two causal frames, with episode-start duplication owned by the framework. Prediction slots are t-1..t+14; only slots 1..8 execute at 20 Hz. All twelve bindings supervise [300,840), including predecessor overlap [300,440) and placement overlap [680,840).

## Handoff and use
Enter with the drawer open/stable, a reliable low block pose and an open gripper free of the handle. Prefer this policy when block displacement is the principal variation. Hand off after observed closed-finger object/TCP coupling, exit from the tunnel, z above 0.16 m, and motion toward `target_pose`. Closed fingers are only a proxy.

## Assumptions and limitations
The translated targets must remain physically valid, collision-free and IK-reachable. Object-frame equivariance does not address tunnel or drawer displacement and can follow a noisy or dropped object. No contact truth and no failed-grasp examples exist. For an unusual predecessor TCP/configuration with a near-nominal block, h02 has a fresh-TCP correction representation. For ambiguous extraction/lift/transfer ordering, h03 retains destination geometry and a narrow clearance guard.
