# Single full-task selected-anchor Cartesian diffusion prior

## Decision and objective

This package contains one policy for the complete buffer swap. It has no caller-selected object, stage argument, segment, runtime policy choice, or hidden phase memory. Its generalization objective is **translation of the blocks being approached/manipulated and of their observed destination regions**.

The learned Cartesian action is a TCP pose relative to one learned translation anchor. Four action channels encode anchor scores for `red_object`, `blue_object`, `red_destination`, and `blue_destination`; decoding uses their argmax. If the selected anchor moves by a world translation `d` and the learned relative position is unchanged, the decoded TCP target moves by exactly `d`. This is the construction-level invariant. Object anchors cover approaches and the first red transfer to the unmarked temporary area. Destination anchors cover the blue transfer and, after blue is placed, the final red transfer. The world orientation, gripper command, robot configuration, reachability, contact dynamics, and choice of anchor are deliberately not assumed translation invariant.

The expected benefit is smaller label variation for contact-critical offsets and direct update of a sampled command when a block or goal has moved. It is falsifiable: compared with a world-position output, moderate object/goal translations should preserve approach and placement offsets when the correct anchor is predicted. This does not claim that relative coordinates identify the whole control problem.

## Evidence from all original demonstrations

All twelve complete trajectories were inspected from start through terminal state. They consistently implement one long order:

1. Red begins near the bottom/blue goal and is approached, closed on, lifted, moved to a free central location around `x=-0.18, y=0`, opened, and left there.
2. Blue begins near the top/red goal and is approached, closed on, lifted, transported to the observed bottom/blue goal, lowered and opened.
3. Red is re-approached in the central area, lifted, transported to the observed top/red goal, lowered and opened; the arm retreats to about 0.30 m TCP height.

For example, `demo10100` shows red contact/lift around indices 102--205 and central release by 274, blue pickup around 446 and bottom placement around 617--652, then red pickup around 823 and top placement/retreat around 961--1029. The same sequence is visible at the corresponding inspected indices in `demo10101` through `demo10111`. Initial red and blue XY positions vary by roughly centimetres across demonstrations, while the final blocks lie within a few millimetres of their goals. The policy therefore needs both fine object/TCP geometry and long-horizon progress disambiguation. The central buffer is not an observed fixture, and grasp truth is unavailable; these are the main difficulties.

## Causal input representation

`build_inputs` constructs a `[2,72]` tensor from the two causal observations, oldest first. Initial episode padding is framework duplication of the first observation. Each 72-channel frame contains:

- all 9 measured `qpos` and 9 `qvel` channels, retaining arm posture, finger width and dynamic state;
- world TCP position and world rotation 6D;
- red and blue world positions and rotation 6D;
- both observed world goal positions;
- TCP-to-red, TCP-to-blue, TCP-to-each-goal, red-to-red-goal, blue-to-blue-goal, and red-to-blue displacement vectors.

The two frames expose object/TCP motion and finger changes without contact truth or future state. Fixed physical scaling is used rather than fitting another normalizer: arm positions are divided by 3 rad, finger positions by 0.04 m, arm velocities by 2.5 rad/s, finger velocities by 0.5 m/s, and all world/relative positions are divided by 0.5 m (implemented as multiplication by 2). Rotations remain unit-scale matrix columns. Absolute robot and world features are retained because equal relative geometry can still differ in Panda reachability or joint-limit feasibility.

Only `qpos`, `qvel`, `tcp_pose`, `red_pose`, `blue_pose`, `red_goal`, and `blue_goal` are used. Drawer compatibility channels are intentionally excluded.

## Learned action and label construction

The unnormalized action has 14 dimensions:

1. 3 metres: commanded TCP position minus the selected anchor translation;
2. 6 values: first two columns of the commanded **world-frame** TCP rotation;
3. 1 native gripper scalar;
4. 4 one-hot anchor labels during training, interpreted as scores at inference.

Targets use `panda_kinematics.commanded_tcp_poses`, so they describe the Cartesian pose reached by the demonstrated native joint command, not lagging measured TCP. Every slot is referenced to its own pre-action `action_observations` state. This matches fresh-state decoding during physical execution.

The training-only anchor labeller is deterministic and label preserving. A lifted object close to the TCP is object anchored while red is being moved to the temporary site; lifted blue is blue-destination anchored; lifted red becomes red-destination anchored after blue is observably near its goal. Lowered completed placements near a destination retain the destination anchor. Remaining approach/transit commands use the nearest object or destination in target XY. The label changes coordinates only: subtracting the selected anchor and adding that same anchor reconstructs every original Cartesian command. No action is added, removed, scripted, or assigned a caller phase. At deployment the diffusion model predicts all four scores from causal state; the labelling rule is not a runtime controller.

The framework fits action normalization on these actual 14-dimensional valid labels. Native-joint min/max scales are not reused.

## Decoder and physical action

For each represented slot, `decode_action`:

1. selects the maximum learned anchor score;
2. resolves that object or observed goal translation from the **fresh current observation**;
3. adds the learned relative position;
4. projects the two learned rotation columns to SO(3) by finite Gram--Schmidt, with a deterministic finite fallback only for degenerate sampled columns;
5. calls `panda_kinematics.solve_ik(target_pose, current_observation['qpos'], public_context['robot'])`;
6. appends the unchanged learned gripper scalar.

No alternate controller, position clipping, or silent IK substitution is present. Diagnostics report the selected anchor, world target, IK convergence, position/rotation residuals, iterations, joint-limit contacts and maximum joint change. Demonstrated valid 6D rotations are unchanged by the orientation projection. The declared task-space conversion check permits 0.02 maximum metres/radians/gripper error; the actual shared IK uses its tighter default tolerances when a target is reachable.

The host temporal convention is unchanged: history 2, horizon 16 representing `t-1` through `t+14`, execute slots 1--8 beginning at current `t`, at 20 Hz.

## Model and gradient path

A registered trainable condition encoder maps flattened `[2,72]` features through `144 -> 256 -> 256` with Mish and LayerNorm. Its 256-dimensional output conditions the supplied standard `DiffusionBackbone` with the declared 14 action dimensions. The mature DP backbone is otherwise unchanged because the prior is fully expressed in causal relational inputs and action coordinates.

Masked DDPM epsilon loss is the only objective. Its gradient flows through the backbone and condition encoder; both are registered, optimized, EMA-tracked and checkpointed. `prior_loss` is a differentiable zero reporting term. No auxiliary head is used because no extra prediction target is needed for the construction-level anchor prior, and an artificial stage objective could encourage brittle temporal classification.

Training keeps all complete bindings and the assignment's numerical recipe, including 60,000 declared updates, batch 128, seed 0, DDPM 100 training/inference steps, AdamW at `1e-4`, cosine schedule with 500 warmup steps, gradient norm 1, EMA 0.999, and last-EMA selection.

## Assumptions, expected failures, and signatures

The useful assumptions are that structured object/goal poses are accurate, blocks remain on a reachable tabletop, the central temporary area remains free, and deployment states still admit the demonstrated transfer order. Moderate translations are targeted; large changes can alter collision constraints and IK branches even when local offsets transfer.

Expected failures include:

- **Missed grasp or slip:** fingers narrow/close but the intended block does not rise or co-move with TCP across the causal frames; finger closure alone is not proof of contact.
- **Wrong anchor/ambiguous progress:** decoded anchor diagnostics repeatedly alternate while TCP and object-to-goal vectors show little progress.
- **Unreachable displaced target:** `ik_converged=false`, non-small residuals, joint-limit contacts, or repeated large joint changes.
- **Buffer failure:** red remains near an occupied goal, is dropped outside the central free area, or obstructs the later blue path.
- **Placement failure:** a carried object returns to table height but its object-to-own-goal XY vector remains outside full-containment margin or its z is outside tolerance.
- **Out-of-support state:** block rotation/displacement, obstruction, drop, or changed order is unlike every demonstration, producing repeated motion without decreasing relational errors.

There is no failed-grasp recovery, obstacle demonstration, collision model, contact sensor, explicit buffer marker, or guarantee under large block yaw. Goal/object anchoring cannot make the robot base, joint limits, world obstacles, or grasp dynamics invariant.

## Full-task entry and exit

Normal entry is the start of a buffer-swap episode with both blocks/goals observed, an open gripper, reachable objects, and free central table space. Intermediate demonstrated-style entry is possible because the policy conditions on physical progress rather than a caller phase, but recovery is not demonstrated.

Exit is only the supplied success expression `red_at_goal AND blue_at_goal` with full rotated XY containment and the supplied z tolerance. Anchor choice, open gripper, arm retreat, or low velocity is not a substitute. `HANDOFF.json` records these semantic conditions and limitations; it is not a runtime controller.
