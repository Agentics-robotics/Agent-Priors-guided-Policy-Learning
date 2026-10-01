# Prior: fresh-TCP incremental correction

## Decision and targeted failure

This policy targets entry states and replans whose measured TCP and Panda configuration differ from the nearly identical nominal trajectories. Rather than predict an absolute world or object-frame pose, it predicts a local SE(3) correction from each measured TCP. Each executed slot is re-anchored to the fresh physical TCP, so the policy can continue from intermediate states instead of assuming exact replay.

Observed support: all demonstrations begin from the same qpos and follow essentially the same approach. In inspected trajectories, indices 0--200 are free-space approach, closure and lid motion start near 220, the long pull continues through 740--750, release is commanded at 760, and retreat occupies 780--949. This demonstrates labels for nominal local corrections, not recovery from unseen errors. Tolerance to shifted entries is an expected consequence of fresh anchoring and relational conditioning.

## Causal inputs and argument use

Each of two causal frames has 36 dimensionless values: lid pose in the TCP frame, local TCP-to-handle vector computed from public handle geometry, lid angle divided by the caller's `max_open_angle_rad`, scaled lid velocity, measured qpos/qvel including finger joints, peg position in the TCP frame, and the supplied travel scale. `lid_pose_field` selects the observed moving lid pose. Thus both arguments affect actual computation in training and deployment. Measured qpos/qvel preserve reachability, gripper state and robot dynamics that a task-space correction alone cannot encode.

## Action conversion

For every training slot,

`Delta_tcp = inverse(T_world_tcp_measured_before_action) * T_world_tcp_command`,

where the target comes from public FK of the demonstrated seven-joint command. The unnormalized learned action is local translation in metres, the relative rotation logarithm in radians, and the demonstrated gripper scalar. Masks cover padded slots and the framework fits normalization only on valid labels in this representation.

For each physical execution step, `decode_action` constructs the correction exponential and right-multiplies it onto the **fresh current** observed TCP pose. Corrections are not accumulated from chunk start and denoising calls never advance physical history. The resulting world target goes to `panda_kinematics.solve_ik` from fresh measured qpos. Diagnostics expose convergence, task residuals, iterations, joint-limit contacts, maximum joint change and correction norms. The gripper scalar is passed unchanged for the standard sign rule. Matching-observation encode/decode recovers the demonstration command in task space; nonunique IK is accepted only through the declared FK consistency check.

## Model and optimization

The standard `DiffusionBackbone` is unchanged, with a 72-D flattened condition and 7-D action. Masked epsilon diffusion loss remains fully active. There is no auxiliary head, augmentation, privileged inference input, image encoder, recurrent state or decoder script. All fixed temporal and numerical settings are retained: two observations, horizon 16 (t-1 through t+14), execute slots 1--8 at 20 Hz, and the specified 20,000-update DDPM/EMA training.

## Handoff, expected benefit and limitations

Prefer this model for a collision-free, locally reachable off-path entry or partly open lid when TCP-to-handle geometry still indicates the phase. It is expected to correct from measured state more naturally than absolute replay. It can nevertheless accumulate bias, issue ineffective tiny corrections or oscillate in orientation. Re-anchoring the future local labels against later current TCP states is responsive but does not prove stability.

The package retains all `[0,950)` actions and therefore all 2,520 outgoing overlap actions in `[740,950)`. Handoff requires observed open fingers, lid exposure, clear TCP, low lid velocity and unchanged peg. No contact truth is available, and successful demonstrations do not establish missed-grasp or lost-contact recovery. Use the moving-lid body-frame policy for large assembly displacement and the phase-predictive world policy when nominal release timing is the main ambiguity.
