# Phase-local translation-anchor prior

## Decision and target failure

`place_red__h04` targets reachable translation of the buffered red source while avoiding a moving-object reference for the long destination-directed transport. Its TCP position reference is selected from each causal state:

1. use observed `red_goal` when red is more than 0.05 m above goal z or within 0.12 m horizontally of the goal;
2. otherwise use the translation of whichever object is nearer the TCP: blue during predecessor release and red during source acquisition.

The output orientation remains in world-axis 6D. Thus translating an active red source translates approach, contact, closure and initial-lift TCP targets by construction, but after observable lift the reference becomes the fixed observed destination so transport and placement retain a goal-directed displacement. Blue-relative prefix targets similarly preserve the predecessor release if blue is translated. This is an expected generalization property, not observed out-of-distribution success.

## Evidence and relation to validation

The only policy validation is in-distribution replay-entry evaluation. `place_red__h01` passed 3/12, whereas h02 and h03 passed 12/12; all h01 failures retained `blue_at_goal` and missed `red_at_goal`, and no original run had an unconverged IK step. It is an unconfirmed inference, not an observed cause, that continuously reanchoring held-red actions to moving red can weaken destination-directed transport.

The demonstrations support the geometric phase order. In `demo10100`, blue opens around 640, the TCP retreats through 700, approaches table-supported red through 740--800, closes by 820, lifts red through 840--860, transports through 880--920, lowers through 940--960, opens at 980 and retreats through 1040. In `demo10104`, the corresponding events occur from the earlier entry at 610 through release around 970 and retreat around 1030. Red source positions in the observed training entries are tightly near `(-0.181, 0, 0.020)` m, so displaced-source pickup is not demonstrated.

## Inputs and invocation

Required arguments are `target_object="red"`, `destination="red_goal"`, and `handoff_object="blue"`. They select real observed scene fields and are not actuator or formula inputs. The fixed switch constants belong to this prior rather than the caller.

Each of two causal frames contributes 71 features. These include scaled qpos/qvel including both fingers; TCP, red and blue positions relative to red_goal; world rotation-6D for the three poses; each object's own-goal displacement; inter-goal displacement; TCP-object distances; world goal and TCP positions; TCP and destination displacement from the selected anchor; and a one-hot blue/red/destination anchor identity. The robot and world terms retain configuration, reachability and kinematic dependence while relational terms expose phase and task geometry. Initial history is framework-duplicated only at episode start, and causal history remains available across policy switches.

## Cartesian labels and conversion

For each supervised slot, `panda_kinematics.commanded_tcp_poses` converts the demonstrated seven-joint command to its exact world TCP command. `encode_targets` resolves the anchor from that slot's pre-action observation and emits ten unnormalized values: world TCP position minus the selected anchor translation in metres, world rotation-6D, and the original native gripper scalar. The framework fits midpoint/half-range normalization only on valid represented labels and applies the supplied mask.

At execution, `decode_action` resolves the same anchor rule from the fresh observation, adds the fresh anchor translation, projects 6D to SO(3) by stable Gram--Schmidt, and calls the verified `panda_kinematics.solve_ik` from freshly measured qpos. It does not substitute a different command if IK is unconverged. Diagnostics report anchor identity and switch geometry together with convergence, position/rotation residuals, iterations, lower/upper limit contacts, maximum joint change and posture status. The represented gripper scalar is appended unchanged, preserving the common nonnegative-open/negative-close rule. Slots remain t-1 through t+14 and the host executes slots 1--8.

## Model and gradients

The only trainable module is an independent unchanged standard `DiffusionBackbone` with condition dimension 142 and action dimension 10. No auxiliary head, recurrent state, script, future input or contact label is used. The masked epsilon diffusion loss is the full objective and supplies gradients to every backbone parameter; `prior_loss` is a differentiable zero report. The fixed 20,000-update DDPM/AdamW/EMA recipe is unchanged.

## Skill, overlap and handoff

All actions in all twelve assigned `[start,stop)` ranges are bound. The approximately 230--240 action overlap with `place_blue` remains supervised from final blue lower/open through retreat, transit, red approach/closure and initial lift. Entry permits blue still low over its goal, although preference is strongest once blue is stable and red is ready for pickup. Exit is unchanged `red_at_goal AND blue_at_goal`. Open fingers and high retreat are demonstrated terminal guidance only; there is no successor.

## Assumptions and limitations

The memoryless anchor rule is not contact detection. It assumes a table-supported central red outside the 0.12 m destination radius, a conventional lift exceeding 0.05 m above goal z before horizontal transport, finite poses, and collision-free reachability. An unusual low carry, slip, pose noise near a switch, red already near goal, or source displacement that changes nearest-object ordering can select the wrong anchor. Translation equivariance does not provide yaw equivariance, obstacle avoidance, stable grasp, force reasoning, recovery, or feasible IK for arbitrary displacement. Prefer h02 when destination anchoring or a source already near goal dominates, and h03 for moderate TCP/configuration mismatch. No available result establishes unseen-error recovery.
