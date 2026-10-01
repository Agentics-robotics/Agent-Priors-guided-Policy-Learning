# Rigid-grasp source-motion prior

## Decision and target failure

This policy targets a secure but atypical container-to-TCP grasp and modest source/TCP displacement at entry. Pour success depends on the container and its lip, not on replaying one nominal TCP path. For every demonstrated action, the observed rigid container-to-TCP transform converts the commanded TCP pose into a desired container pose. The policy diffuses the corresponding **container-frame source increment**. At execution it applies that source motion to the fresh container pose and then reattaches the fresh observed container-to-TCP transform to obtain the required world TCP target.

Prefer h03 only when the source is observably rigid with the TCP and pose estimates are reliable. Use h01 for major bowl displacement and h02 for ambiguous partial discharge. An unusual transform is not evidence that a grasp exists, and rapid transform change is a failure signal rather than a reason to compensate harder.

## Evidence and assigned responsibility

The inspected trajectories keep the gripper closed through actions 180, 320, 360, 1460, 1600, 1680 and 1740 while the source follows the TCP from early lift to staged pour, righting and outbound carry. Start source heights range roughly 0.024-0.060 m, and outbound timing differs (for example demo12303 is already away from the bowl near 1680). These observations support nominal rigid transport and the need to retain source pose; they do not demonstrate slip recovery or large grasp-transform variation.

All twelve complete `[180,1740)` intervals are supervised. Incoming overlap `[180,360)` and outgoing overlap `[1460,1740)` therefore remain substantial action supervision under the new source-motion representation.

## Causal input

`build_inputs` uses two causal observations and produces 130 features per frame:

- scaled measured Panda qpos and qvel;
- source relative to bowl and return target;
- TCP relative to source (the observed grasp transform);
- TCP relative to bowl;
- all particle positions in source and bowl coordinates, deterministically sorted by source-frame position;
- observed settled/source fractions, source upright cosine and bowl-relative source height.

The features retain robot configuration, destination geometry, grasp geometry and particle phase because source-local action invariance alone does not make reachability or transfer dynamics invariant. No future states, hidden contact truth or unprovided action history are inference inputs.

## Source-motion label derivation

For action slot `i`, let:

- `C_i` be the observed world source pose before the action,
- `X_i` be the observed world TCP pose,
- `U_i` be the demonstrated commanded world TCP pose from verified FK,
- `G_i = inverse(C_i) @ X_i` be the observed source-to-TCP transform.

The rigid-grasp desired source pose is `C*_i = U_i @ inverse(G_i)`. The learned increment is `D_i = inverse(C_i) @ C*_i`, encoded as translation in metres, 6D rotation columns and the native gripper scalar. This is a 10D representation distinct from h01's absolute bowl-frame TCP target and h02's TCP-frame rotation-vector residual.

At execution, fresh observations provide `C` and `G = inverse(C) @ X`. The decoder projects the learned 6D orientation with safe Gram-Schmidt and computes `X* = C @ D @ G`. It then calls `panda_kinematics.solve_ik(X*, measured_qpos, robot)`. Diagnostics expose convergence, task residuals, iterations, limit contacts, joint change, world target and the observed grasp transform. The native gripper scalar is passed through unchanged.

For a valid training label, this algebra reconstructs the demonstrated commanded TCP target before numerical IK error. It also changes the required TCP target when a different but rigid grasp transform is observed, which is the intended prior.

## Model, normalization and gradient path

`SourceMotionPolicy` flattens the two 130D frames and conditions the unchanged standard `DiffusionBackbone(260, 10, training_config)`. The masked DDPM epsilon objective remains active and is the total loss; `prior_loss` is differentiable zero because the label/decoder geometry supplies the prior. All backbone parameters are registered for optimization, EMA and checkpointing.

The framework fits representation normalization only from valid source-increment labels, with padding masked. Inputs use explicit physical scales. Temporal behavior is unchanged: two frames, 16 slots spanning t-1 through t+14, execute slots 1-8, 20 Hz.

## Expected benefit, assumptions and limitations

The falsifiable expected benefit is that, for a secure rigid grasp whose container-to-TCP transform differs from the nominal transform, the decoded TCP target changes so that the source follows the demonstrated local motion and attitude rather than the nominal end-effector path. This should preserve source orientation and lip motion better under off-center grasping and modest displaced entry.

The core assumption is rigidity during each action. Slip, grasp loss, pose noise, compliance and collision violate it. The policy cannot prove contact or recover from those failures. Training grasp variation is modest, so unusual-grasp benefit is expected rather than observed. Source-local increments are not a far-destination anchor and can accumulate error; h01 covers major bowl displacement. h02 better handles discharge-phase ambiguity.

## Handoff

Entry requires a lifted source moving rigidly with the TCP. Exit preserves the task definition: at least ten particles settled, source itself approximately upright and still held, and outbound motion begun. `return_place_release` is nominally ready near 1680, with the fully supervised `[1460,1740)` interval supporting flexible takeover.
