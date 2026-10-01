# Prior: fresh-TCP body-frame incremental targets

## Decision and targeted failure

This independent policy targets nonnominal intermediate TCP states and early/late entry into the assigned skill. Instead of predicting an absolute replay pose, each action is the demonstrated commanded TCP target relative to the measured TCP immediately before that action. At execution, every slot is composed with the fresh measured TCP. Thus the same learned local correction can be applied around a nearby changed robot state.

The motivation is observed timing variation, not demonstrated recovery. Demo12300 remains near the table-supported grasp around indices 148-160 and has source z about 0.029 m at 180. Demo12303 closes earlier and has source z about 0.056 m at 180. Both then execute the same lift/carry responsibility. No trial deliberately perturbs the robot or shows a missed grasp.

## Inputs and arguments

`source_object="container"` and `destination_object="bowl"` select observed poses. Each of two causal frames has 55 deterministic features: TCP-relative source and bowl poses, source-relative bowl pose, world source/TCP positions, qpos/qvel, finger width, and particle centroid relative to the source. Fixed physical scales keep channels comparable. Measured joints and world positions retain robot configuration and reachability dependencies; relative geometry alone is not treated as making the problem fully invariant. Initial history duplication and cross-skill history are framework-owned.

## Actions and conversion

The learned dimension is 7: body-frame translation xyz in metres, body-frame rotation vector xyz in radians, and the unchanged gripper scalar. For slot i, `encode_targets` computes `inverse(action_observation[i].tcp_pose) @ commanded_tcp_pose[i]`. This uses only training labels for the command; no future state enters model inputs. `decode_action` computes `current_tcp @ Exp(local_action)` and invokes `panda_kinematics.solve_ik` from freshly measured qpos. Solver convergence, task residuals, iterations, limit contacts, and maximum joint change are reported without hiding failure.

The standard DiffusionBackbone is unchanged; coordinate design supplies the prior. Two 55-channel frames form a 110-dimensional condition. The only optimization objective is masked epsilon diffusion, with differentiable zero `prior_loss`. Representation normalization is fitted by the host on valid 7D labels, not inherited from native joint actions. Temporal and numerical settings remain fixed at H2/16/8, 20 Hz, and the declared 20,000-update DDPM/EMA recipe.

All twelve authorized [0,360) segments are supervised. The entire [180,360) action interval overlaps `controlled_pour_and_right`, preserving lift, carry, stage, and first-tilt handoff behavior.

## Expected benefit and limitations

Expected benefit: for a locally shifted TCP or an invocation partway through a demonstrated phase, decoded targets move with the actual TCP reference and avoid forcing an absolute nominal joint path. This is a falsifiable expected benefit, not evidence of recovery.

Local actions can compound prediction error and do not guarantee convergence from large handle or source displacement. The source-frame policy covers the latter more directly. Body-frame increments do not certify contact, resolve severe phase ambiguity, or recover unseen misses and slips; the phase-aware policy explicitly shapes causal phase features. All decoded targets remain constrained by Panda IK, limits, collision clearance, and the observed object geometry.
