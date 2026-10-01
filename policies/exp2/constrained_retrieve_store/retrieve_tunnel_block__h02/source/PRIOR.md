# Fresh-TCP body-delta prior

## Decision and target failure
This independently trained policy targets valid entry states or mid-skill restarts whose Panda TCP and redundant joint configuration differ from the nominal predecessor handoff. Every demonstrated command is represented as a body-frame SE(3) correction from the **measured TCP immediately before that action**. At deployment each selected slot is composed with that slot's fresh measured TCP, not accumulated from older predictions.

The demonstrations do not vary the index-300 entry TCP substantially, so robustness to a new handoff is an expected benefit. Within the authorized segment, however, supervised states span high approach (420), low alignment (520:540), closed-gripper pull (560:600), lift (640:700), and reorientation/transfer (780:820). Those phases supply action labels for local corrections at many configurations.

## Causal inputs and model
`build_inputs` emits two 52-channel causal frames. Each includes measured qpos/qvel, explicit finger width, object-to-TCP and object-to-destination transforms, world TCP pose, drawer state, signed object clearance and object world position. The historical pair exposes current motion without future states or unprovided action history. Joints preserve redundancy, limits and reachability dependencies.

`FreshTcpDeltaDiffusion` flattens the 2x52 input to a 104-D global condition and uses the unchanged public `DiffusionBackbone` with 7 action channels. All parameters are registered, optimized, EMA-tracked and checkpointed through that backbone. There is no auxiliary head; the masked epsilon diffusion loss is active on every valid supervised slot and `prior_loss` is differentiable zero.

## Action labels and execution conversion
For each slot, `encode_targets` obtains the Cartesian FK pose of the native joint command, forms `inverse(measured_TCP_before_action) @ commanded_TCP`, and encodes body translation in metres, relative rotation as a three-component rotation vector in radians, and the native gripper scalar. This is a 7-D representation fitted only on authorized labels.

`decode_action` constructs the relative pose with the exponential map and computes `fresh_measured_world_TCP @ relative_pose`. It then invokes `panda_kinematics.solve_ik` from fresh qpos and appends the represented gripper value. IK convergence, residuals, iterations and joint-limit contacts are reported rather than hidden. Nonnegative gripper values open and negative values close.

Because each slot is a feedback correction, there is no open-loop summation: labels use each slot's own pre-action observation and execution resolves each selected slot from its current observation. The check compares reconstructed task-space pose and gripper, allowing nonunique IK.

## Calling, overlap and handoff
The typed call selects `object_pose`, `target_pose`, and the 0.16 m clearance feature. Two causal frames persist across skill switches; initial duplication is framework-owned. Prediction remains 16 slots (t-1..t+14), and slots 1..8 execute at 20 Hz. Every action [300,840) in all 12 trajectories is bound, preserving 140 predecessor-overlap and 160 successor-overlap actions.

Prefer this policy when the block remains near demonstrated tunnel support and TCP/configuration mismatch is the main variation. Enter with the drawer open and a collision-free observed arm state. Hand off after coupled closed-finger block/TCP motion, tunnel exit, clearance above 0.16 m and destination-directed motion.

## Assumptions, expected failures and complementarity
Local corrections do not guarantee a globally safe path and can drift. They are not invariant by construction to a large block displacement; h01 covers reliable object displacement with object-frame actions. Phase inference still relies on two observed frames and grasp proxies; h03 is preferable when preventing premature transfer below the tunnel roof is more important. IK feasibility and collision freedom remain external physical constraints. Contact truth and failed-grasp recovery are absent from the demonstrations.
