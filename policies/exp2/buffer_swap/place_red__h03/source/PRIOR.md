# Fresh-TCP body-twist residual prior

## Decision and target failure

This independent policy targets moderate intermediate TCP/configuration mismatch and phase variation at the `place_blue` handoff. Each action is a local SE(3) correction relative to the measured TCP immediately before that action. At execution every represented slot is recomposed from the fresh measured TCP, rather than replaying an absolute demonstration target. This is intended to feedback-recenter modest tracking/IK offsets while retaining relational scene and robot configuration inputs.

## Evidence and expected benefit

Observed entries differ: `demo10100` index 620 has blue around z=0.040 m, while `demo10104` index 610 enters earlier around z=0.055 m and reaches opening near 630. The inspected sequences then include retreat (660--700), transit (720--760), red descent/contact (780--800), closure/lift (820--860), carry, lowering/release (900--980), and retreat (1000--1056). This supports phase diversity within the cut. It does not demonstrate recovery from an unseen predecessor error. The expected benefit is that a moderate TCP displacement changes the world target but preserves the local corrective transform.

## Inputs and typed intent

The required semantic identifiers select red, red_goal and blue fields. Each of two causal frames contains 61 features: scaled qpos/qvel and fingers, world TCP pose, TCP-body vectors to both objects and both goals, object orientations relative to TCP, own-goal vectors in object frames, and four phase distances. These preserve reachability, kinematics, gripper state and object-goal relationships. There is no future state, action history, contact truth or hidden phase input. Initial observation padding and cross-policy history are framework-owned.

## Training labels and action conversion

Verified FK converts each demonstrated arm command to its exact commanded world TCP pose. For slot `i`, `encode_targets` computes `inverse(T_tcp_pre_action_i) @ T_command_i`. The seven learned values are body-frame translation in metres, SO(3) logarithm/rotation vector in radians, and native gripper scalar. The framework fits normalization only on valid represented labels and applies the provided masks.

`decode_action` resolves the fresh measured TCP, maps the rotation vector through the exponential map, computes `T_tcp_fresh @ T_relative`, and invokes `panda_kinematics.solve_ik` from fresh qpos. The returned solver joints are not replaced on nonconvergence. Diagnostics publish local translation/rotation norms, convergence, position/rotation residuals, iterations, limit contacts and joint change. Native gripper sign semantics remain unchanged. Prediction slots and execution remain fixed at t-1..t+14 and slots 1..8.

## Model and gradient path

An unchanged standard `DiffusionBackbone` uses condition dimension 122 and action dimension 7. There is no auxiliary head, recurrent state or decoder script. Masked epsilon diffusion loss reaches every trainable backbone parameter; framework numerical settings, optimizer, 20,000 updates and EMA remain fixed.

## Skill coverage and overlap

All actions in every assigned range are bound. The full approximately 230--240 action overlap is especially important here because it supervises local corrections through blue lower/open, retreat, transit, red close and initial lift. Entry requires blue low over/supported in its goal and finite scene/robot state. Exit is unchanged `red_at_goal AND blue_at_goal`; there is no successor.

## Assumptions, limitations and complementary coverage

Residual corrections assume moderate mismatch and demonstrated relational phase. They can accumulate error over long transport and cannot infer contact, recover a missed grasp, stop slip, stabilize blue, avoid an unseen obstacle or force unreachable IK. The relative rotations in training stay close to identity and away from the pi log branch. Large red source displacement is handled more directly by `place_red__h01`; translated destination geometry is handled by `place_red__h02`. None of these expected benefits is final-test or recovery evidence.
