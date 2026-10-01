# Phase-aware local Cartesian corrections

## Decision and targeted failure

This policy targets entry at a perturbed intermediate state and ambiguity between descending while attached, opening/settling, retreat after deposit, and presentation at the handle. It uses fresh-current-TCP-frame Cartesian targets as local corrections and an auxiliary causal phase classifier. It does not target large rigid displacement; the object- and destination-frame policies cover that case.

Observed evidence motivates the modes. demo12100 indices 680-900 show closed carry/descent, 910-920 opening and settling, 940-960 upward retreat, and 1080-1159 handle approach/re-closing. Settling timing differs: demo12102 opens around 900 and reaches floor by 905, while demo12104 opens around 895 and is settled by 900. This observed variation supports phase regularization but does not establish recovery from errors.

## Inputs and calling contract

Required arguments select `object_pose`, `target_pose`, and the fixed `carry_release_retreat_handle` vocabulary. The vocabulary is validated and encoded as an input channel; pose fields select actual live observations.

Each of two causal frames has 48 features:

- destination-frame TCP pose: translation divided by 0.75 m plus rotation 6D;
- destination-frame object pose: translation divided by 0.75 m plus rotation 6D;
- TCP-frame object pose: translation divided by 0.5 m plus rotation 6D, exposing following/separation;
- seven arm positions / 3 rad, two finger positions / 0.04 m;
- seven arm velocities / 2.5 rad/s, two finger velocities / 0.25 m/s;
- drawer position / 0.3 m and velocity / 0.3 m/s;
- one vocabulary code.

These are causal and deterministic. Robot posture and velocities preserve reachability and gripper dependencies. Two-frame object/TCP relative change provides motion evidence without claiming contact. The framework alone pads the initial observation and preserves history across skill switches.

## Local action conversion

A 7D action is `[delta_position_current_tcp(3 m), delta_rotation_vector_current_tcp(3 rad), gripper(1)]`. For every training slot, `encode_targets` converts native joint commands to their commanded world TCP poses and computes `inverse(T_world_measured_tcp_pre_action) @ T_world_commanded_tcp`. Its local rotation is mapped with the SO(3) logarithm.

For each executed slot, `decode_action` reads the new measured TCP, exponentiates the sampled rotation vector, and computes `T_world_tcp_fresh @ T_tcp_increment`. Thus slots 1..8 are closed around fresh physical observations rather than accumulated from chunk start. The world target enters only `panda_kinematics.solve_ik`, initialized from fresh measured qpos. The learned gripper scalar is appended unchanged. IK convergence, position/rotation errors, iterations, limit contacts and maximum joint movement are explicit diagnostics.

Task-space roundtrip tolerance is 0.005 over FK position metres, rotation radians and gripper error. Rotation log/exp is exact over demonstrated local labels and joint nonuniqueness is allowed.

## Phase target, mask, and gradient path

The auxiliary target corresponds to current action slot 1 (time t), matching the causal history's current state. It is masked by that slot's framework action mask.

The deterministic four-class label is:

- **0 carry/closed descent:** object is above the destination floor band and current demonstrated gripper command is negative;
- **1 opening/settling:** object is above the floor band and gripper command is nonnegative;
- **2 deposited retreat:** object is within 0.015 m of destination z and TCP is not in the handle region;
- **3 handle-ready:** deposited object plus TCP more than 0.20 m behind destination x and below z=0.20 m.

These labels use successful demonstration action/geometry and are not contact truth or an inference-time oracle. Deployment receives only causal features. A registered MLP maps the flattened 96D input to a 128D latent. That latent directly conditions the unchanged published diffusion backbone and feeds a registered 4-logit phase head. Masked cross entropy is `prior_loss`; total loss is diffusion loss plus 0.1 times phase loss. Therefore phase gradients update both the head and the shared condition encoder, while action diffusion gradients update the condition encoder and backbone. Every learned module is in the model, optimizer, EMA and checkpoint. Repeated denoising calls do not mutate phase state.

## Bindings and handoff

All twelve [680,1160) slices, context 679, remain bound. This keeps the 680:840 predecessor overlap and the 900:1160 successor overlap fully action-supervised.

Prefer this policy for an unusual local lag, height or release timing while geometry remains within the demonstrated corridor. Use `place_block_in_drawer__h01` when a plausibly attached block is rigidly displaced and `place_block_in_drawer__h02` when the drawer/goal is displaced or fixed-handle anchoring is important. Transfer to close_drawer only after actual two-frame object observations support floor-height stationarity and detachment; the auxiliary class alone is insufficient.

## Expected benefit and limitations

Expected benefit: fresh-TCP decoding reduces open-loop replay after local perturbations, and the auxiliary head encourages a condition latent that separates action modes with different gripper/vertical commands. This is a falsifiable expectation, not rollout evidence.

Local increments can drift and provide no construction-level global displacement invariance. Coarse labels can be wrong for unseen contact failures; two frames may not resolve noisy settling. The policy cannot recover an unseen drop, guarantee collision clearance, or guarantee IK feasibility. It preserves the supplied task success definition and uses no augmentation.
