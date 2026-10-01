# Manipulated-object-equivariant prior

## Decision and target failure

This policy targets displacement or yaw of the buffered red block beyond the demonstrations. The inspected demonstrations keep the source near `(-0.181, 0.000, 0.020)` m, so source displacement is not empirically covered. The learned Cartesian target is instead expressed in the pose of the object currently being manipulated. A causal nearest-3D-TCP rule selects blue during final predecessor release and red during red acquisition and transport. Therefore translating or rotating selected red transforms the decoded pickup-side TCP target with red by construction.

This is an expected generalization benefit, not observed recovery. Red must remain visible, supported, collision-free and reachable. The prefix uses blue so red displacement does not incorrectly translate blue-release actions.

## Evidence and scope

In `demo10100`, indices 620/640 show blue lowering and opening, 700 is the high retreat, 740--800 approaches red, 820 closes, 860 is lifted red, 900 transports, 960 lowers, 980 opens, and 1040--1063 retreats/holds. `demo10104` enters earlier at 610 with blue around z=0.055 m, opens around 630, acquires red around 800--820, lifts by 850, and releases/retreats around 980--1056. These are observations from authorized training slices. They support the phase relationships but not unseen-error recovery.

## Inputs and arguments

The required arguments are `target_object="red"`, `destination="red_goal"`, and `handoff_object="blue"`. They select actual state keys and are not actuator parameters. Each of two causal frames contributes 69 deterministic features: scaled qpos/qvel including fingers, world TCP pose, TCP/target/handoff poses in the active-object frame, both goals in active and corresponding object frames, TCP distances and the selected-object bit. Robot configuration and world TCP retain reachability and kinematic dependence; relational features provide phase and task geometry. Initial episode history padding is framework duplication, and causal history continues across policy switches.

## Action conversion

For each supervised slot, `panda_kinematics.commanded_tcp_poses` converts the demonstrated seven arm targets to their world TCP command. `encode_targets` applies the selector to that slot's pre-action observation and computes `inverse(T_active) @ T_command`. The ten labels are relative position in metres, the first two rotation-matrix columns (6D), and the original native gripper scalar. Valid-label representation normalization is fitted by the framework.

At execution, `decode_action` applies the same selector to the fresh observation, projects 6D to SO(3) by stable Gram--Schmidt, composes the relative target into world, and calls the supplied `panda_kinematics.solve_ik` from freshly measured qpos. It returns the solver result without substitution and reports convergence, position/rotation residuals, iterations, limit contacts and maximum joint change. The eighth native component is the represented gripper scalar; the common executor retains the sign rule. Slots represent t-1 through t+14 and only slots 1--8 execute.

## Model and gradients

The model is an independent standard `DiffusionBackbone` with condition dimension 138 and action dimension 10. No backbone change, auxiliary head, scripted motion, future input or contact label is used. The masked epsilon diffusion loss is the complete objective; its gradient reaches all backbone parameters. The framework owns DDPM normalization, masking, optimizer, EMA and the fixed 20,000-update numerical recipe.

## Coverage and handoff

All twelve assigned `[start,stop)` ranges are bound. The approximately 230--240 action overlap with `place_blue` is action-supervised, including blue lower/open, retreat, transit, red approach/close and initial lift. Entry requires blue low over or supported in its goal and red observable. Exit remains exactly `red_at_goal AND blue_at_goal`; open gripper/high retreat are demonstrated guidance only. There is no successor skill.

## Limitations and complementary policies

The nearest-object transition can be discontinuous near equal distances and is not contact estimation. Pose noise, unreachable displacement, wrong attachment, slips, unstable blue, or unseen collision can still fail. Destination displacement is covered more directly by `place_red__h02`; moderate intermediate tracking mismatch is covered by `place_red__h03`. No package claims recovery absent from demonstrations.
