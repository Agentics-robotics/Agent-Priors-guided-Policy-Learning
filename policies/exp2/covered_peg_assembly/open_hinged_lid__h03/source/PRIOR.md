# Prior: phase-predictive world Cartesian policy

## Decision and targeted failure

This policy targets phase ambiguity at handle closure versus pulling, maximum opening versus release, and released-lid settling versus retreat. It deliberately keeps an absolute world Cartesian action representation, which differs from both companion priors. A registered causal condition encoder is jointly trained by the main diffusion loss and a short-horizon future phase objective.

Observed evidence: at index 200 the lid is closed and the gripper command is open; by 220 the close command coincides with initial lid motion; the fingers remain closed while lid position rises through 740--750 to roughly 1.71--1.74 rad; open is commanded at 760; around 800 the released lid can have high velocity; and by 820--949 it settles near 1.35--1.40 rad while the TCP retreats. The same ordering is visible in `demo12208` despite peg-position variation. This supports the phase labels but does not prove contact or recovery.

## Inputs and high-level intent

`build_inputs` emits two causal frames of 43 dimensionless features: world lid pose, world TCP pose, TCP-to-handle vector computed from public handle geometry, normalized lid angle and velocity, angle-to-release, measured qpos/qvel including fingers, and the normalized release argument. The required `lid_pose_field` is resolved in each observation. `release_angle_rad` is bound to 1.73 for every demonstration and enters both direct conditioning and angle-to-release. It is intent/geometry conditioning, not a scripted threshold and not an actuator command.

No future observation, true contact, unprovided action history or auxiliary ground truth enters deployment inference. The framework preserves two causal observations across skill switches and duplicates only the first episode observation when necessary.

## Actions and task-space conversion

The 10-D learned action is an absolute world TCP position in metres, a world rotation represented by its first two matrix columns, and the original gripper scalar. Labels use `panda_kinematics.commanded_tcp_poses` on demonstrated native joint commands. Sampled orientations are projected to SO(3) with deterministic Gram-Schmidt and a finite degeneracy fallback. The world target is sent only to `panda_kinematics.solve_ik` from freshly measured joints. Diagnostics report convergence, residuals, iterations, joint-limit contacts and maximum joint change; failures are not silently replaced. Gripper sign semantics are unchanged.

The framework fits the representation normalizer on valid world-action labels. Matching label decode reconstructs the original task-space target; public FK after nonunique IK is checked to the declared 0.005 maximum position/rotation/gripper tolerance.

## Registered encoder and auxiliary gradient path

The model owns a trainable 86-to-256-to-256 MLP condition encoder. Its latent conditions the unchanged standard `DiffusionBackbone`. A registered 256-to-256-to-32 phase head reshapes to `[16,2]` and predicts, for each demonstration slot, the **after-action** lid position divided by 1.85 and the demonstrated gripper sign. `encode_targets` supplies `phase_targets` and an explicit `phase_mask` copied from the framework validity mask, so padded slots contribute no auxiliary loss.

`compute_loss` applies masked epsilon loss to the action diffusion output and masked MSE to the phase sequence. Total loss is `diffusion_loss + 0.05 * phase_loss`; reported `prior_loss` is the weighted auxiliary term. The phase gradient flows through the phase head and shared condition encoder, while the diffusion gradient also trains that encoder and the backbone. All modules are registered, optimized, checkpointed and EMA-tracked. At inference the auxiliary head is not used to override observed state or learned action.

## Expected benefit, handoff and limitations

The expected benefit is a phase-discriminative causal latent around the short close/pull/release transitions. It is falsifiable by premature release, failure to release, or observed command/phase mismatch. It is not demonstrated recovery and the auxiliary prediction is not a contact or safety signal.

All twelve `[0,950)` segments remain action supervised, including the full 2,520 actions in outgoing `[740,950)` overlap. Handoff relies on observed open fingers, box exposure, clear TCP, low lid velocity and unchanged peg. Absolute world targets are sensitive to large lid displacement; h01 covers that failure by construction. h02 is preferred for unusual robot/TCP entries. A wrong caller release angle, unseen collision, unreachable target or noisy pose remains outside established support.
