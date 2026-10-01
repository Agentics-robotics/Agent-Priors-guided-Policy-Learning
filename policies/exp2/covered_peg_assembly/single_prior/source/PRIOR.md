# Single full-task prior: causal relational SE(3) action reference

## Decision

This package contains one policy for the entire covered-peg task. It has no segments, stage labels, caller-selected object, high-level policy, policy switch, or scripted task completion. Every one of the 12 demonstrations has one binding covering its complete `[0, stop)` action interval with `call_args={}`.

The inductive bias is a **causal adaptive Cartesian reference**. The reference is resolved from the newest observation once per 8-control replan and is held fixed for that chunk:

1. the observed lid frame is emphasized while the lid is closed, or while a closed gripper remains geometrically close to the observed lid handle;
2. the observed peg frame is emphasized after release of the opened lid and during peg approach, grasp and early lift;
3. the observed `target_pose` frame is emphasized after the peg is lifted or has made source-to-destination progress.

These are continuous representation weights, not a predicted or externally supplied stage. The single diffusion network always receives the same input and produces the same 10-dimensional action type.

The policy learns the commanded TCP pose relative to that reference. Therefore translating the lid during lid interaction, the peg during source approach/retrieval, or the destination during elevated transport/insertion translates the decoded TCP command directly by construction. Relative observation features also expose the new geometry to the network. Absolute Panda joints and world TCP pose remain inputs so that object equivariance does not erase reachability, joint-limit and robot-configuration dependencies.

## Demonstration evidence and learning difficulty

All complete trajectories contain roughly 1,539--1,559 actions. They show the same long sequence but with physical state feedback throughout:

- Initial pegs vary from approximately `x=-0.6038` to `-0.6157 m` and `y=-0.1958` to `-0.2073 m`, about 12 mm in each horizontal direction. The lid begins at angle zero and the robot begins with an open 0.08 m finger width.
- In `demo12200`, the TCP reaches the lid handle region near indices 200--250; the fingers close to about 0.049 m total width and the observed lid angle rises along the handle arc. At indices 740--760 the lid is near 1.71--1.74 rad. By index 780 the fingers are open again and the TCP retracts.
- The open-gripper peg approach is visible around indices 980--1080. At index 1100 the fingers total about 0.0253 m and the peg pose begins to follow the TCP. Peg z then rises from about 0.026 m to 0.151 m at index 1180 and about 0.375 m at index 1220.
- Transfer both changes position and turns the peg from its initial quarter-turn quaternion toward the identity orientation. Around index 1400 the peg is near `(-0.300, 0.180, 0.420) m` with identity-like orientation; by index 1504 it is near the insertion line. All terminal states place the peg center near `(-0.18224, 0.17678, 0.09825) m`, identity-like, while the TCP is offset near x=-0.213 m.
- The other eleven complete overviews show the same interaction order with different initial peg positions, lid dynamics, pickup joint configurations and final lid settling angles (roughly 1.25--1.31 rad).

The main difficulties are the very long behavior, only 12 closely related demonstrations, two separate grasps/releases, unobserved contact, moving lid geometry, held-object transport, a large orientation change, and millimetre-scale insertion. A gripper command alone does not establish contact. The policy therefore conditions on two frames of object/TCP relationships, lid position/velocity, finger state, joint state and velocity rather than using elapsed time or a stage label.

## Reference mechanism

`adapters._reference` uses only the newest causal state. It constructs nonnegative weights summing to one:

- Lid relevance combines a smooth lid-closed gate over 0.55--0.95 rad with a closed-gripper/handle-proximity gate. Handle proximity is computed from the public local handle point `[-0.17, 0, 0.028] m` transformed by the observed lid pose. During demonstrated opening the proximity term keeps the lid frame active after the angle gate falls, while post-release pickup states with the lid settled near 1.1 rad use the peg frame.
- Once lid relevance falls, peg-to-destination relevance increases smoothly with peg height above the lid origin (0.06--0.16 m) or with horizontal progress from the lid/source region toward `target_pose` (0.48--0.72 normalized progress).
- Reference position is the weighted lid/peg/target position. Reference orientation is a sign-aligned normalized quaternion blend. In the demonstrated interaction interiors one weight is near one; blending smooths only the transition.

The thresholds encode demonstrated geometry, not success. They neither issue an action nor claim contact. They can fail on unusual intermediate states, as documented below.

## Causal input

`build_inputs` returns `features[2,80]`, oldest frame first. At episode start only, the framework duplicates the first frame. Each frame contains:

- nine Panda positions, with arm joints divided by 3 and finger joints by 0.04;
- nine velocities divided by 2.5;
- world TCP position divided by 0.5 and its 6D rotation;
- SE(3) TCP pose relative to lid, peg, and target;
- peg pose relative to target and hole pose relative to target;
- TCP error from the known lid-handle point in the lid frame;
- lid angle and velocity with fixed physical scales; and
- the three causal reference weights.

Every relative pose uses translation divided by 0.25 m and the first two rotation-matrix columns. These are fixed, dimensionless physical scalings; no evaluation statistics or future observations are used. World TCP, qpos and qvel intentionally retain robot reachability and configuration information. The declared dependencies exactly match the fields read by the adapter.

## Learned action and converters

The learned action dimension is 10. For every prediction slot it is:

`[reference-frame TCP translation (3 m), reference-frame TCP rotation 6D (6), gripper scalar (1)]`.

The prediction slots are fixed at t-1 through t+14; slots 1--8 execute t through t+7 at 20 Hz. `encode_targets` obtains the demonstrated world TCP pose by exact public FK of the seven native commanded joints. It computes `inverse(chunk_reference) @ commanded_TCP`, stores the local translation and 6D rotation, and copies the native gripper scalar. It does not use measured TCP as the action label.

`decode_action` robustly Gram--Schmidt projects the 6D columns, composes the local pose with the same chunk reference, and calls only:

`panda_kinematics.solve_ik(world_target, freshly_measured_qpos, public_robot)`.

It returns the seven absolute IK joint targets plus the unchanged gripper scalar. It reports convergence, position and rotation residuals, iteration count, joint-limit contacts, maximum joint change, world target pose and reference weights. It does not replace or conceal an unconverged result. Gram--Schmidt is exact for valid encoded labels; the fallback only makes degenerate sampled columns a finite proper rotation. The declared independent roundtrip tolerance is 0.005 across metres, radians and gripper scalar.

The framework fits midpoint/half-range normalization only from valid 10D represented training labels. Native-action normalization is not reused. No data augmentation is used, avoiding invalid contact or reachability labels.

## Model, losses, and gradient path

`RelationalDiffusionPolicy` contains the published `DiffusionBackbone` unchanged. The two 80D frames are flattened to the 160D global condition. The backbone predicts epsilon for `[B,16,10]` noisy represented actions. There is no auxiliary head: the useful prior is in the causal relational input and action coordinates, so adding a weak phase surrogate would risk reproducing elapsed-time segmentation.

`compute_loss` applies the framework's masked epsilon loss. `prior_loss` is a differentiable zero and diffusion learning remains fully active. All trainable parameters are registered under the backbone and therefore enter AdamW, EMA and checkpoints. The assignment's declared numerical recipe is retained: history 2, horizon 16, execution 8, DDPM 100 train/100 inference steps, batch 128, seed 0, AdamW 1e-4, weight decay 1e-6, gradient clipping 1, EMA 0.999, cosine schedule with 500-step warmup, published widths 128/256/512, and 60,000 updates.

## Expected benefit and falsifiable scope

The expected benefit is lower extrapolation burden under object displacement. For example, with equal represented output, moving the peg at source by a vector moves the early retrieval TCP target by that vector; moving `target_pose` moves elevated transfer and insertion targets with it. During lid contact, changing lid pose changes both handle-relative conditioning and the decoded target frame. This can be falsified by comparing decoded world TCP targets before and after a reachable reference displacement while holding represented action fixed: the active-reference component should transform with that displacement. This construction covers lid approach/opening, peg approach/early lift, and destination transport/insertion. Transition regions are only partially equivariant because their reference is blended.

The construction assumes object observations are accurate, target_pose moves consistently with the physical hole, scene topology and object dimensions are unchanged, and displaced interactions remain within Panda reach and collision-free. It does not make dynamics, IK, collision geometry or contact invariant.

## Full-task entry, exit, and limitations

Entry can be reset or a causally observed intermediate state from the same task; there is no handoff controller. Exit remains the supplied success definition, including head location and at least 0.985 axis alignment. `HANDOFF.json` is documentation only.

Expected failures and observable signatures include:

- **Unreachable displaced target:** IK reports `converged=false`, large pose residual, limit contact or large joint change.
- **Missed lid grasp:** fingers close but lid_position does not rise while TCP remains near the handle.
- **Missed peg grasp/drop:** fingers close without stable peg-to-TCP relative motion, peg z does not rise, or peg returns toward the box.
- **Out-of-support reference transition:** weights change quickly or remain mixed in a state unlike demonstrated lift/transport; decoded chunks may be discontinuous across replans.
- **Inconsistent destination observation:** target-relative motion heads away from hole_pose.
- **Insertion/contact error:** peg-target relative translation stops shrinking, orientation is below required alignment, or final y/z remains outside 0.009 m despite a closed gripper.
- **Large displacement/collision:** Cartesian equivariance requests a geometrically corresponding target but cannot reason about unseen walls, self-collision or altered hinge/box topology.

Two observations provide velocity cues but no contact truth or force sensing. Demonstrations have narrow variation, so object rotation changes, large independent lid/peg/hole displacement, unusual partial opening, regrasp, jamming, or recovery after a drop are not established. The decoder deliberately contains no scripted recovery.
