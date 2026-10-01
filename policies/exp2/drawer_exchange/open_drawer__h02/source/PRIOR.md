# Fresh-TCP local increment servo

## Decision and target failure

This policy targets modest, reachable deviations in TCP pose, joint configuration and tracking lag, including entry states produced by another policy. Demonstrations contain long free-space approach, constrained pulling, release, large reorientation toward red, descent and lift, while measured and commanded TCP poses are not identical during motion. The expected benefit is local correction from fresh physical state rather than return to a memorized absolute waypoint. It is not evidence that unseen contact failures can be recovered.

## Mechanism

Causal inputs preserve measured qpos/qvel and express the articulated drawer frame and red pose relative to the measured TCP. Drawer progress, velocity and remaining desired opening are explicit. The caller's drawer pose calibrates these relational inputs but does not specify actuator motion.

For every slot, `encode_targets` obtains the demonstrated commanded TCP through public FK and computes `inverse(measured_TCP_before_action) @ commanded_TCP`. The learned 7-vector is local translation in metres, local rotation vector in radians, and unchanged native gripper scalar. `decode_action` forms the same SE(3) transform and composes it on the fresh measured TCP at the physical execution step. It then invokes only `panda_kinematics.solve_ik` from fresh qpos. Diagnostics publish convergence, residuals, iterations, joint-limit contacts, increment norms and target world TCP pose. This per-step composition is deliberately different from the drawer-frame absolute action of the displacement prior.

## Model and training

The standard Diffusion Policy backbone is unchanged. It receives two flattened frames of 39 deterministic relational features. There is no auxiliary head; masked epsilon diffusion remains the complete learned objective and `prior_loss` is differentiable zero. All modules are registered by the backbone. Representation normalization is fitted by the framework on valid local-transform labels, not on native joint scales. There is no augmentation. Every action in every [0,470) segment is supervised, retaining the full 230:470 overlap. H2/16/8, slot alignment, 20 Hz, DDPM100, EMA and the 20,000-update budget remain fixed.

## Evidence and expected use

In demo1000, the policy must cover approach through 140, drawer motion at 160-230, opening at 240, retreat at 260-300, red approach after 320, contact near 445 and lift by 470. Demo1005 reaches a different red Y and final arm configuration while retaining these responsibilities. Those observations motivate preserving robot configuration and local geometry. They do not prove robustness to deliberately perturbed starts.

Prefer this model when scene geometry is close to training but the arm is modestly off-path. Use the drawer-frame policy when a reliable cabinet displacement is the dominant variation. Use the phase-conditioned world policy when nominal geometry but stage ambiguity is dominant. Monitor drawer progress, TCP/front relation and red/TCP co-motion; a repeated local command without the expected physical response is a failure cue.

## Handoff and limitations

`red_transfer` can receive once the drawer is above 0.26 m and red is exposed because both skills supervise actions from 230. The clean nominal switch is closed fingers with observable red/TCP upward co-motion near 470. Local transforms can accumulate drift off distribution, apply an inappropriate nominal push after missed contact, and do not make a distant object reachable. No contact truth, collisions, alternative approach side, blocked drawer, drop or recovery appears in training. IK nonconvergence is reported rather than hidden.
