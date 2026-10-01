# Destination-anchored placement prior

## Decision and target failure

This independent policy targets translation of the observed red destination and placement drift near that destination. Every commanded TCP position is learned after subtracting the current observed `red_goal` position. The world orientation remains 6D because the tabletop goal supplies no orientation. Thus translating a reachable goal translates the decoded carry/lower/release/retreat target by construction, while source-to-goal and robot features condition the preceding pickup.

## Evidence and expected benefit

Observed `demo10100` indices 860--920 carry lifted red toward the goal, 940--960 lower it, 980 opens, and 1000--1040 retreat. `demo10104` shows the same destination sequence at 850--1056. Final red centers are tightly near x=-0.351 and y=0.195--0.196 while the marker is fixed at (-0.35, 0.20, 0.02) m. This demonstrates the sequence but not translated-goal success. The falsifiable expected benefit is lower sensitivity of the latter trajectory to a reachable marker translation than a raw world-position action policy.

## Causal inputs and arguments

`target_object="red"`, `destination="red_goal"`, and `handoff_object="blue"` select actual causal fields. Each of two frames has 62 features: scaled qpos/qvel including finger state, TCP/red/blue positions relative to red_goal, world rotation-6D for each pose, object-to-own-goal vectors, inter-goal displacement, TCP-object distances, and world goal/TCP positions for reachability. Arguments are semantic identifiers only. Initial history duplication and cross-skill causal history are framework-owned.

## Labels and decoding

`encode_targets` obtains the exact demonstrated commanded TCP pose with verified FK. A ten-dimensional label stores `p_command - p_red_goal` in metres, world rotation-6D, and demonstrated gripper scalar for every valid slot. `decode_action` resolves the fresh observed goal, adds its translation, projects rotation-6D by stable Gram--Schmidt, and passes the world target to `panda_kinematics.solve_ik` from fresh measured qpos. It returns the solver result and explicit convergence, position/rotation residual, iteration, joint-limit and joint-change diagnostics; no alternate controller or silent correction is used. Slot alignment remains t-1 through t+14 with slots 1--8 executed.

## Model and optimization

The model uses an unchanged standard `DiffusionBackbone`, condition dimension 124 and action dimension 10. There is no auxiliary head or privileged deployment input. Framework-fitted representation normalization uses valid destination-relative labels only. The masked epsilon loss provides the gradient to every trainable module; the standard fixed DDPM/AdamW/EMA recipe is unchanged.

## Skill and overlap

All twelve complete assigned ranges are bound. The approximately 230--240 shared actions with `place_blue` remain action-supervised from blue lowering/opening through red initial lift, although exact destination equivariance is most relevant after red is held. Entry remains blue low over/at its goal and red observable. The unchanged exit is `red_at_goal AND blue_at_goal`; demonstrated opening and high retreat do not add predicates. There is no successor.

## Assumptions, failures and complementary coverage

The destination must remain observable, collision-free and reachable. This frame does not by itself solve a displaced source, failed contact, slip or unstable blue. Early blue-release actions rely on learned relationships because they are also encoded relative to red_goal. Goal rotation is not represented by the task. `place_red__h01` directly covers displaced red pickup, while `place_red__h03` feedback-recenters moderate intermediate TCP offsets. None claims recovery unsupported by demonstrations.
