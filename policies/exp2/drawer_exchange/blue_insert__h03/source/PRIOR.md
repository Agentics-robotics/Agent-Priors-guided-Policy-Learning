# Fresh-TCP increments with causal phase regularization

## Decision and targeted failure

This policy targets valid intermediate entry states produced by the predecessor, phase ambiguity, and small TCP tracking deviations. It learns the demonstrated command as an SE(3) transform from each slot's measured pre-action TCP, then composes every executed slot with the freshly measured TCP. This avoids tying corrections to one absolute world trajectory.

The cut changes object identity: demo1000 opens over red by 620, retreats by 650, crosses toward blue, descends by 800, closes by 820 and lifts through 840-850. Timing differs substantially: at index 850 demo1000 blue is already near z=0.191 m while demo1001 is only near z=0.061 m. These are observed nominal variations. Better handling of unseen intermediate states is expected, while failed-grasp/drop recovery is not established.

## Inputs and typed intent

Required arguments `source_object="blue"` and `destination_field="blue_goal"` select the causal source and destination fields. Each of two frames contributes 38 fixed-scale values: qpos/qvel, TCP pose relative to blue, TCP-red displacement, blue-goal displacement, red-pad residual and drawer position/velocity. Two frames expose motion and finger changes without simulator contact truth. History crosses policy switches and is duplicated only at initial episode padding.

## Action conversion

The 7-D label is `[local_translation_m(3), local_rotation_vector_rad(3), gripper(1)]`, where the local transform is `inverse(T_world_tcp_before_action) @ T_world_commanded_tcp`. The commanded target is FK of the demonstrated native joint command. Training uses each slot's `action_observations`; no future state is an inference input.

At execution, `decode_action` receives fresh physical state for each represented slot, computes `T_world_tcp_fresh @ T_tcp_delta`, and calls only `panda_kinematics.solve_ik` from fresh joints. Diagnostics report IK residuals, limits and increment norms. Slot timing remains t-1..t+14 with execution of slots 1..8.

## Trainable phase mechanism and gradients

A registered 76->128->128 MLP encodes causal features. The unchanged standard DiffusionBackbone consumes the 128-D condition, and a registered five-class linear head predicts:

0. predecessor red release,
1. blue approach/free transit,
2. blue-held transport,
3. insertion/release,
4. post-release retreat.

`encode_targets` derives the label only from causal current finger width, TCP-red distance, TCP-blue distance, blue-goal distance and height. It supplies an explicit valid mask. Masked cross entropy is weighted 0.05 and backpropagates through both phase head and shared encoder; `prior_loss` reports this weighted term. The masked epsilon diffusion loss remains primary and trains encoder plus backbone. All modules are registered, optimized, EMA-tracked and checkpointed. The head does not script or terminate actions.

## Coverage, use and handoff

All twelve [600,stop) occurrences and all 3,000 [600,850) overlap actions are retained. Prefer this model when switch timing or current configuration differs but remains within the demonstrated responsibility. For large blue displacement, use the blue-frame model; for changed drawer placement during stable carry, use the destination-frame model. Success remains the supplied simultaneous evaluator condition, never the phase prediction.

## Limitations

Local increments can drift over an eight-step chunk and do not provide a global collision-free route. The phase label is a heuristic partition of observed nominal states; proximity/co-motion and finger closure are not contact truth. Large source or destination displacement is conditioned rather than invariant. Demonstrations do not cover drops, re-grasps, drawer-wall impacts or recovery.
