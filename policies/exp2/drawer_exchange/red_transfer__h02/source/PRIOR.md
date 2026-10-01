# Destination-centered placement prior

## Decision and target failure

This policy targets displacement of the outside pad and sensitivity during carried-red descent, support, release, and retreat. It learns commanded TCP position as an offset from the `red_goal` observed at chunk construction. A translated goal therefore translates the decoded Cartesian targets before IK. Orientation remains an absolute world 6D rotation because the marked pad supplies a position but no observed orientation.

Observed evidence is limited to a fixed goal. In `demo1000`, held-red transport reaches the goal area by 530-560, descends through 580 and 594-604, opens around 608-610, and retreats by 620-650. `demo1001` shows different final red XY/orientation and release timing around the same goal. Generalization to a relocated goal is expected from the representation and remains conditional on reachability and clearance; it is not demonstrated success.

## Inputs and calling contract

The caller supplies closed identifiers: `destination="red_goal"`, `carried_object="red"`, and `successor_object="blue"`. `build_inputs` resolves them in both causal observations and freezes the newest destination point for one sampled chunk.

Each 50-D history row retains scaled measured Panda joints, fingers, and velocities; goal-relative TCP, red, and blue positions; their world 6D rotations; the blue-goal offset from red_goal; and drawer position/velocity. This preserves robot configuration and contact-relevant fingers while removing a needless absolute translation from scene positions. A 3-D intent vector records the bound identifiers. No future states or contact truth enter inference.

## Action representation and conversion

The 10-D label is `[goal-relative world-axis XYZ metres, absolute world rotation 6D, gripper scalar]`. `encode_targets` uses verified FK of each demonstrated native arm command. It subtracts the causal chunk red_goal from commanded TCP position and copies the first two world rotation columns. Valid masks remain framework-owned and representation normalization is fit only after labels are built.

`decode_action` adds the same causal goal point, safely orthonormalizes the sampled 6D rotation, and passes the resulting world TCP pose to `panda_kinematics.solve_ik` from the freshly measured joints. The represented gripper value is appended unchanged and retains the common nonnegative-open/negative-close rule. Diagnostics report convergence, residuals, iterations, limit contacts, and joint change. The finite fallback for degenerate 6D samples is a declared nonlearned projection; valid encoded rotations round trip exactly.

## Model and optimization

An unchanged independent `DiffusionBackbone` consumes the flattened H2 relational features and intent. There are no auxiliary modules or privileged labels. Masked epsilon diffusion is the complete learned objective; reported `prior_loss` is a differentiable zero. All standard DDPM/EMA settings, 20,000 updates, H2/16/8 timing, and 20 Hz control remain fixed.

## Cut and handoff

Every authorized `[230,850)` action is bound for all twelve demonstrations. Thus the model retains `[230,470)` overlap with `open_drawer` and `[600,850)` overlap with `blue_insert`. The successor overlap intentionally includes red descent and release before retreat and blue pickup, preventing the receiving context from treating the predecessor red object as its intended blue target. Nominal readiness is red stationary on the pad plus blue/TCP co-motion during initial lift; these are handoff cues, not extra completion conditions.

## Applicability and limitations

Prefer this policy when red is acquired or near acquisition and a changed reachable red_goal or pad-relative drop accuracy is the main concern. It does not make a far-displaced red pickup invariant, and it is not invariant to blue displacement in the handoff. A goal translated into an unreachable or obstructed region remains physically invalid. The goal is frozen for eight executed slots, and no training evidence establishes moving-goal behavior, missed-grasp recovery, off-pad recovery, or collision avoidance. Use red-object equivariance for displaced pickup and phase-conditioned local correction for uncertain intermediate/post-release entry.
