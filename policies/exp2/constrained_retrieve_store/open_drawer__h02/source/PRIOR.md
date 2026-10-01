# Fresh-current-TCP body-frame residual prior

## Decision and target failure

Deployment may invoke the skill from an intermediate state produced by another policy or with small TCP tracking error. This policy learns the correction from each slot's measured TCP to its demonstrated commanded TCP, expressed in the measured TCP body frame. Every executed slot is recomposed from a fresh physical TCP observation instead of a predicted future state. The expected benefit is feedback-like tolerance to small intermediate-state offsets; it is not evidence of recovery from unseen contact failure.

The inspected demonstrations provide action supervision across approach (`demo12100` indices 0:160), close/initial drawer motion (165:180 in `demo12101`), pull (200:240), full-open hold and release (250:275), clearance (280:340), and departure (340:439). The two causal frames expose drawer motion, robot motion and measured fingers at each phase.

## Inputs and argument

`open_stop_m` is required in metres and bound to 0.300. It normalizes drawer progress. Each causal frame has 47 deterministic features:

- drawer progress and velocity;
- world measured TCP position/6D rotation;
- `target_pose` relative to TCP;
- `object_pose` relative to TCP;
- all measured arm/finger positions and velocities.

The framework provides exactly two causal frames, duplicating only the initial observation, and preserves them over policy switches. No future state, contact truth, image or undeclared state is used.

## Action conversion

For each slot, `encode_targets` computes the FK pose of the demonstrated seven-joint command and then `inverse(measured_tcp_pose) * commanded_pose`. Its seven outputs are local translation in metres, local rotation vector in radians and demonstrated gripper scalar. This uses each slot's pre-action measured TCP.

At execution, `decode_action` constructs the local SE(3) correction and right-composes it with the fresh current measured TCP. It never integrates corrections across slots or denoising calls. The resulting world TCP target goes only through `panda_kinematics.solve_ik` from fresh measured joints. Diagnostics expose target pose, correction norms, convergence, residuals, iterations and limit contacts. The gripper's sign semantics are unchanged. Framework representation normalization fits valid transformed labels, while padding remains masked.

## Model and gradients

The standard published `DiffusionBackbone` is unchanged: 94 flattened causal features condition diffusion of 16 by 7 actions. Its registered parameters receive masked epsilon-loss gradients. There is no auxiliary head; `prior_loss` is a connected zero and does not replace diffusion learning. No data augmentation is used.

## Coverage and handoff

All actions 0:440 from all twelve occurrences are supervised. Actions 300:440 retain the full 140-action/7 s overlap with `retrieve_tunnel_block`. Switch after full stable opening, handle release and arm clearance; retain causal history.

## Applicability and limitations

Prefer this policy when the observed state remains near a demonstrated phase but TCP/configuration is slightly offset. Similar local geometry can still be phase ambiguous, so drawer progress/velocity, finger state and two-frame motion are essential. The representation does not force approach to follow a largely displaced drawer or departure to follow a largely displaced block; the drawer-frame and dual-anchor policies cover those cases. Large predicted corrections can leave training support, violate clearance or fail IK. Closed fingers are not proof of handle contact, and no demonstration establishes recovery from a missed handle, premature release or disturbed block.
