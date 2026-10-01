# Blue-goal-frame absolute Cartesian targets

## Decision and target failure
This policy targets destination displacement and late/intermediate entries after blue pickup. TCP position labels are relative to the observed blue_goal point, while target orientation remains a canonical world wxyz quaternion. This directly anchors descent, release, and retreat geometry to the actual destination. The package is trained on the whole skill, not trimmed to placement.

## Evidence and expected benefit
Blue is carried around z=0.28-0.30 at `demo10100:520-580` and `demo10105:510-570`, descends at indices 600-620 and 590-610, and is settled/opened by 640 and 630. It remains fixed while the robot retreats and approaches buffered red. These observations support the phase and handoff description. Because all demonstrations share a fixed goal, shifted-goal benefit is expected from the coordinate construction rather than measured deployment evidence.

## Inputs and arguments
Required typed arguments are `target_object=blue`, `destination_goal=blue_goal`, and `context_object=red`, and all three select adapter fields. Each of two causal frames contains 45 features: scaled qpos/qvel; TCP, blue, and red pose relative to the destination translation; red-goal separation; and blue_goal world position. Robot state and goal world position retain configuration and reachability dependencies. Initial history duplication and history across switches are framework-owned.

## Action and conversion
The 8D label is goal-relative world TCP position in metres, canonical world wxyz quaternion, and native gripper scalar. `encode_targets` uses FK of demonstrated joint commands and the causal chunk goal. `decode_action` adds that goal, normalizes/canonicalizes quaternion, creates a world target, and calls `panda_kinematics.solve_ik` from fresh measured qpos. Quaternion normalization is an explicit projection. Diagnostics expose convergence, residuals, iterations, limit contacts, maximum joint change and pre-projection quaternion norm. Label normalization is fit only on valid authorized labels.

## Learning
The model keeps the standard `DiffusionBackbone`, conditioned on flattened 2x45 features, with 8 action channels. The masked epsilon objective is the active loss. No auxiliary module or architecture change is introduced; optimizer, EMA and checkpoint therefore cover the complete trainable model. Temporal and numerical settings are unchanged.

## Skill and handoff
All twelve assigned slices and both large overlaps are supervised. Prefer this package when blue visibly co-moves with the gripper, is elevated/en route, or is near blue_goal. Exit when blue is stationary in the goal and the gripper is open/clear. The tail can continue through buffered-red closure and visible initial lift to make the successor handoff robust.

## Limitations
Goal anchoring does not make a displaced source pickup invariant; use h01 there. Quaternion projection can turn an extreme off-distribution sample into an unintended orientation. There is no contact truth, so a closed command is not a grasp. Use h03 for modest tracking offsets. Shifted destinations can be unreachable, and no demonstration establishes recovery from drops, collisions, wrong-object grasps or disturbed red.
