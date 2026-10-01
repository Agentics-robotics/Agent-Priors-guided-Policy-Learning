# Single prior: adaptive relational Cartesian diffusion

## Decision and objective

This package learns the entire task with one diffusion policy. It does not expose a stage, segment, object selector, policy selector, or caller parameter. The chosen prior is **translation-relative Cartesian control with a causal adaptive reference**:

- Near the tunnel block, the learned TCP position is an offset from the currently observed block position.
- Near the drawer, it is an offset from a drawer-handle anchor computed from the currently observed `target_pose` as `target.xyz + [-0.270, 0, 0.065]` metres.
- Between those regions, the reference is a smooth blend. The block weight is
  `sigmoid((distance(TCP, drawer_anchor) - distance(TCP, object)) / 0.040 m)`.

This is a geometric computation, not a stage label. It uses only the current causal observation and has no phase memory. The same rule creates every training label and is recomputed from the fresh state for every executed action.

The falsifiable expected benefit is that, while the block branch is active, translating the observed block translates an otherwise unchanged approach, pull, lift, and carry target by the same amount. While the drawer branch is active, translating the observed destination/fixture translates opening and closing targets. The expected benefit is limited by reachability, collision geometry, and correct branch selection; relative coordinates do not make the full control problem invariant.

## Evidence and learning difficulty

All twelve complete trajectories were inspected. Initial block positions span approximately x = -0.448 to -0.437 m and y = -0.408 to -0.395 m. This is small but genuine position variation and motivates sharing motion as an offset rather than memorizing one world target.

The demonstrations also show why one global world-frame target prior is weak:

- In `demo12100`, indices 150--275 approach, grasp, and pull the target-derived drawer anchor while the block remains near `[-0.437,-0.406,0.080]` m.
- Indices 430--530 move from above the tunnel to a low TCP pose near the block. Around indices 550--600 the fingers are closed and the block follows the TCP from the tunnel to x near -0.613 m.
- Index 700 has the held block lifted to about z = 0.313 m; index 800 transfers it toward the drawer; indices 880--920 lower and release it near the open-drawer target.
- Indices 950--1120 retreat and return to the drawer handle. Indices 1150--1240 close the drawer, which moves both `target_pose` and the stored block by roughly 0.298 m. Index 1330 is the final open-gripper retreat with the drawer below 0.025 m.
- The other eleven overviews repeat this strategy with different initial and placed block positions. For example, `demo12103` indices 572, 763, 954, and 1145 show low grasp/pull, raised transport, placement, and handle return; `demo12111` indices 578, 771, 964, and 1157 show the same dependencies.

The main difficulties are the roughly 67-second behavior length versus a 16-slot prediction horizon, narrow low-tunnel motion, binary grasp/release decisions without contact truth, changing robot reachability, and the fact that closing the drawer moves the destination and already placed block. Two causal frames expose motion, fingers, drawer velocity, and whether the block is actually following the TCP, but they cannot prove contact.

## Causal inputs

`build_inputs` returns `features[2,60]`, oldest frame first. At episode start the framework duplicates the first observation. Each frame contains:

1. seven arm joints scaled by 3 rad and two finger joints scaled by 0.04 m;
2. seven joint velocities scaled by 2.5 rad/s and two finger velocities scaled by 0.30 m/s;
3. world TCP position scaled by 0.70 m and its world rotation in 6D form;
4. object-minus-TCP, target-minus-TCP, object-minus-target, handle-minus-TCP, and adaptive-reference-minus-TCP vectors;
5. world target position, object rotation, and target rotation;
6. drawer position and velocity, plus the causal block-reference weight.

World TCP, joint configuration, and joint velocity deliberately remain present to retain robot reachability and contact-dependent posture. Relative channels expose transferable geometry. Rotations remain in world coordinates because the fixed tunnel and drawer impose world-oriented clearance. Inputs use fixed physical scales rather than fitting any evaluation data. No image, future observation, action history, simulator contact, or privileged phase enters inference.

## Learned action and converters

The learned action has ten channels per slot:

`[reference-relative position xyz (m), world rotation 6D, native gripper scalar]`.

`encode_targets` obtains the exact demonstrated command's TCP pose using `panda_kinematics.commanded_tcp_poses`, subtracts the adaptive reference derived from that slot's pre-action observation, encodes the world rotation as its first two columns, and copies the demonstrated gripper scalar. Invalid edge padding remains governed by the framework mask.

At execution, `decode_action` recomputes the reference from the fresh measured observation, adds the predicted offset, projects the two rotation columns to SO(3) by robust Gram--Schmidt, and calls `panda_kinematics.solve_ik` from the fresh measured arm joints. The returned joint solution is not replaced or hidden when unconverged. Diagnostics report convergence, position and rotation residuals, iterations, limit contacts, maximum joint change, reference weight, reference and handle positions, and the requested world TCP pose. The gripper scalar is appended unchanged; the executor applies its standard nonnegative-open/negative-close rule.

Prediction slot 0 corresponds to t-1. The host executes slots 1 through 8, starting at current t, at 20 Hz. Although each chunk predicts 16 relational targets, fresh observations are used by every decode, making the executed positions closed-loop with respect to object and drawer movement.

## Model and gradient path

`RelationalCartesianDiffusion` uses the published `DiffusionBackbone` unchanged with a 120-dimensional flattened two-frame condition and ten action channels. This retains the mature temporal U-Net because the required bias is fully expressed in inputs and action coordinates; no backbone modification is necessary. The masked epsilon diffusion loss is the total loss. There is no auxiliary head and `prior_loss` is differentiable zero, so no unsupported labels or ineffective auxiliary gradients are introduced. All trainable parameters are in the registered backbone and therefore in AdamW, EMA, and checkpoints.

The assignment-declared numerical recipe is retained: history 2, horizon 16, execute 8, DDPM 100 training/inference steps, batch 128, seed 0, AdamW at 1e-4 with the declared cosine schedule, and 60,000 updates with last EMA selection. Representation normalization is fitted by the framework only from valid ten-dimensional training labels, not from native-joint scales or evaluation data.

## Complete bindings and calling contract

Every assigned trajectory has exactly one `[0,stop)` binding with `context_start=0` and `call_args={}`. The closed schema accepts only `{}`. No action is removed, duplicated, or delegated. The binding includes drawer opening, tunnel approach, grasp, constrained pull, lift, transfer, placement, release, handle return, closure, and terminal retreat.

## Assumptions and limitations

The target-to-handle offset and fixed scene geometry must retain their declared meaning. The block must remain visible in `object_pose`, and displaced approach and transfer targets must remain reachable without a new collision topology. Translation is handled by construction only when the smooth reference remains on the semantically appropriate anchor. Large displacement can move the nearest-anchor transition, cause the initial state to choose the block too early, or require a different tunnel exit path. The action orientation is not equivariant to arbitrary object rotation. The data contains only one strategy and a narrow initial displacement range, so extrapolation far outside it is unsupported.

Expected observable failures are: fingers close while the block pose stays fixed; TCP motion stalls near the tunnel while measured joint motion/pose fails to progress; the block does not follow the TCP during pull or lift; the gripper opens while the block is outside the moving target region; the drawer remains at or above 0.025 m; or IK reports nonconvergence/limit contact with elevated residual. These signatures are evidence of failure, not hidden stage transitions.

`HANDOFF.json` documents full-task entry and successful exit for monitoring only. It is not a runtime high-level controller and does not alter the supplied success definition.
