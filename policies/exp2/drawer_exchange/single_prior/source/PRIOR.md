# Single full-task relational-anchor diffusion prior

## Decision and objective

This package uses one full-task diffusion policy. It does not expose a stage, an object argument, a high-level selector, or persistent phase state. Its generalization target is translation of the drawer/blue destination, the red and blue source objects, and the red pad relative to the demonstrations.

The learned Cartesian action contains a per-slot reference-frame score. The selected observed origin is one of:

1. the dynamic `blue_goal` origin, used as the drawer frame while opening and as the blue destination while inserting;
2. the current red-block origin;
3. the observed red-pad origin; or
4. the current blue-block origin.

The represented position is a world-axis offset from that origin. `decode_action` resolves the origin again from the **fresh current observation** for every executed slot. Therefore, conditional on selecting the same origin, translating that source object or destination by a vector d translates the decoded TCP target by exactly d while leaving the represented offset unchanged. This construction covers drawer approach/pull, red approach, red transport to the pad, blue approach, and blue transport into the moving drawer. It also makes intervening motion relative to the next relevant source or destination. It does not make reachability, collision clearance, contact, grasp success, or the learned origin choice invariant.

## Evidence and learning difficulties

All twelve original demonstrations were inspected and are bound completely and exactly once. They are long (1063--1077 actions), and each combines qualitatively different geometry and gripper/contact behavior. In `demo1000`, the TCP approaches the drawer at indices 120--137, closes and pulls while `blue_goal.x` moves from about 0.125 m to -0.175 m at indices 151--240, approaches and grasps red around 377--446, transports red around 480--584, places it by 618, approaches and grasps blue around 755--824, transports blue around 858--961, and inserts it by 996. The other inspected demonstrations preserve this order but vary the initial objects: across the assignment, initial red x is about 0.093--0.108 m and y about -0.077-- -0.055 m, while blue x is about -0.414-- -0.387 m and y about 0.287--0.315 m.

The principal difficulties are the approximately thousand-step multimodal sequence, source-versus-destination ambiguity while carrying, discrete gripper changes, moving drawer coordinates, and the lack of contact truth. The two-frame state must also distinguish a stable grasp from merely being nearby. Absolute robot configuration still matters because the same Cartesian relation may be reachable through different or infeasible joint configurations.

## Causal inputs

`build_inputs` constructs two 71-channel frames, oldest first. Initial-episode padding remains framework-owned duplication. Each frame contains:

- measured seven arm joints, two finger joints, and all corresponding velocities;
- absolute world TCP position and TCP 6D rotation;
- red-minus-TCP and blue-minus-TCP positions and both object 6D rotations;
- drawer/blue-goal-minus-TCP, red-pad-minus-TCP, red-pad-minus-red, blue-goal-minus-TCP, and blue-goal-minus-blue relations;
- absolute red, blue, and dynamic blue-goal positions; and
- drawer position and velocity.

Fixed physical scaling is used: arm positions /3, fingers /0.04, arm velocities /2.5, finger velocities /0.5, Cartesian values /0.5 m, drawer position and velocity /0.3. Rotations are 6D matrix-column values. There is no fitted evaluation normalization and no clipping. Relative channels expose the transformations of interest, while absolute TCP/object locations, measured joints, velocities, finger aperture, and drawer state preserve world, reachability, and contact-dependent information.

## Cartesian action and labels

Each of 16 slots is 14-dimensional:

- 0:3: TCP position offset in metres from the selected observed origin, with axes aligned to world;
- 3:9: world TCP rotation as the first two rotation-matrix columns;
- 9: native gripper sign, +1 open and -1 close;
- 10:14: one-hot reference-frame supervision / learned scores in the order listed above.

Slots represent t-1 through t+14. The framework executes slots 1 through 8, beginning at current t.

`encode_targets` computes the exact demonstrated command TCP pose using `panda_kinematics.commanded_tcp_poses`, not the lagging measured TCP. A causal pre-action state assigns a useful origin label: the dynamic drawer origin before it is open; red object versus red pad according to observed red completion and grasp/elevation evidence; then blue object versus dynamic blue destination according to observed blue grasp/elevation evidence. This rule only constructs authorized action labels from each slot's pre-action observation. At deployment no such rule selects a stage: the four scores are generated jointly by the diffusion model and decoded by argmax. Future slot labels are training targets only.

The framework fits midpoint/half-range representation normalization from valid 14D training labels. Native-joint action scales are not reused. Invalid edge padding is ignored by the framework diffusion mask and by the auxiliary `anchor_mask`. No data augmentation is used, so no unsupported contact or reachability labels are synthesized.

At decode, the selected origin is read from the fresh state, the offset is added, and the 6D world rotation is projected to SO(3) by Gram--Schmidt. Valid encoded labels are unchanged. A degenerate sampled 6D pair receives a documented finite fallback based on current TCP orientation; `rotation_projected` reports it. The resulting world 4x4 TCP target is passed only to `panda_kinematics.solve_ik` from freshly measured joints. The solver's convergence, position and rotation residuals, iterations, joint-limit contacts, and maximum joint change are reported. The best unconverged solution is returned without substitution. The gripper score is thresholded to the executor's exact binary convention.

## Learned modules and losses

A registered 142-to-256 MLP encodes the two causal frames. Its output conditions the unchanged standard `DiffusionBackbone` with the assigned widths, DDPM steps, and temporal setup. A registered linear auxiliary head predicts four reference logits for all 16 slots from that same causal embedding.

The main loss is the masked epsilon diffusion loss over every valid 14D action component. The auxiliary target is the causal-origin label for each valid future slot; masked cross entropy is added with weight 0.05. Thus the auxiliary loss has a real gradient through the head and shared condition encoder, while the diffusion loss remains active through the encoder and standard backbone. All modules are registered, optimized, checkpointed, and EMA-averaged. The assigned optimizer, DDPM, seed, update budget, and last-EMA selection are otherwise unchanged.

The auxiliary head is not queried by the adapter during execution; it shapes the shared causal embedding. The executable origin scores remain part of the sampled action, so denoising has no physical side effect and no hidden history is advanced.

## Expected benefit and assumptions

The falsifiable benefit is exact target translation equivariance after a correct origin prediction. For example, a displaced red source changes the red-origin world target while preserving the demonstrated grasp offset; a displaced red pad changes carrying/place targets; a displaced blue source changes its approach; and the observed dynamic `blue_goal` changes both drawer-relative and blue-insertion targets. The two-frame relations should make origin selection and intermediate-state replanning easier than learning all world coordinates directly.

This assumes the scene semantics and drawer geometry are unchanged, observations are accurate, displacements remain in collision-free reachable workspace, and the demonstrated order remains appropriate. World TCP orientation is deliberately retained rather than made object-rotation invariant because the demonstrations contain narrow object yaw support and the down-facing gripper orientation is a robot/world dependency.

## Full-task handoff and limitations

Entry may be the normal closed-drawer start or a recoverable intermediate state with all required observations. There is no runtime handoff inside the task and no successor policy. Exit is only the supplied simultaneous conjunction: drawer position strictly greater than 0.26 m, red fully on its pad at the required z interval, and blue fully in the translated drawer cavity at the required z interval. No release, velocity, clearance, or sustained-time requirement is added.

Expected failures are: a low-margin/wrong origin prediction causing a target jump; a closed gripper with increasing TCP--object separation indicating a lost grasp; repeated object poses with oscillating TCP targets; red or blue settling outside containment/height bounds; the drawer remaining below threshold; repeated SO(3) fallback; or reported IK nonconvergence/limit contact. Large shifts, severe rotations, obstacles, drawer re-closing, dropped-object recovery, changed geometry, and unreachable targets are outside demonstrated support. Relative coordinates do not prove contact or task success.
