# Current-TCP incremental Cartesian prior

## Decision and targeted failure

This independently trained policy targets modest hand-pose and robot-configuration mismatch at an incoming or intermediate switch. The 530-action skill traverses very different configurations: lid release near TCP `(-0.390,-0.199,0.228)` around index 770, clearance and approach, grasp near z=0.029 m around 1080-1100, and lift above z=0.43 m around 1240. Deployment may enter between those nominal states.

Each action is the local transform

`inverse(T_world_tcp_measured_before_action) @ T_world_commanded_tcp`

encoded as local translation (metres), local rotation vector (radians), and native gripper scalar. Decode composes the represented transform with the fresh measured TCP, then solves IK from fresh qpos. This conversion cancels current hand-pose offset in the target construction. It differs from peg anchoring: it addresses local execution state but does not make motion invariant to peg displacement.

## Inputs and normalization

Two causal frames each contain 56 deterministic features: arm qpos/qvel, finger position/velocity, absolute TCP, peg, and lid poses relative to the robot base translation, TCP-in-peg pose, and lid angle/velocity. These preserve robot configuration, world reachability, active-object geometry, and recent motion through the two-frame difference available to the backbone. No future observation, action history, contact truth, image, or recurrent phase is used.

Fixed physical scaling is applied to joints, velocities, and lid state. The framework fits action normalization only on valid local-transform labels from the authorized bindings. Padding remains masked.

## Model and loss

The standard `DiffusionBackbone` is unchanged. The `[2,56]` causal tensor is flattened as global condition for epsilon prediction over `[16,7]` represented actions. Masked epsilon loss is the complete training objective; `prior_loss` is a differentiable zero because the prior is the action geometry. All learned parameters are registered in the backbone and included in optimizer, EMA, and checkpoint state.

## Cartesian conversion

Labels use FK of demonstrated native joint commands, not measured lagging TCP. Rotation-vector exponential and logarithm maps provide the local SO(3) conversion. At execution, the represented SE(3) target is left-composed with fresh measured TCP, and only `panda_kinematics.solve_ik` converts it to seven Panda joint targets. Diagnostics report convergence, residuals, iterations, limit contacts, maximum joint change, and world TCP target. The original gripper sign is retained.

## Expected benefit and limitations

The expected, falsifiable benefit is lower sensitivity to modest TCP/qpos deviations at early or late handoffs than a world-absolute target policy. This is not observed rollout evidence. Local increments can drift or oscillate across repeated chunks and do not create a stability guarantee. A displaced peg still requires learned conditioning rather than constructed equivariance, so the peg-frame policy covers that case better. Large target transforms can be unreachable. Demonstrations contain no misses, drops, blocked-lid cases, regrasp, or collision recovery.

## Calling and handoff

Call arguments are `{}`. All twelve `[740,1270)` occurrences are supervised, preserving `[740,950)` incoming and `[1120,1270)` outgoing overlaps. Prefer this policy when the scene remains reachable but the hand state is locally off nominal. Handoff only after closed fingers, peg clearance above the wall, and causal peg/TCP co-motion jointly indicate successor readiness; none alone proves contact.
