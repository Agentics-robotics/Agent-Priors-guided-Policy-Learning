# Blue-object-frame Cartesian keyframes

## Decision and target failure
This policy targets blue-source displacement beyond the roughly 2 cm spread observed at the normal entry. Every commanded TCP target is learned relative to the blue pose resolved from the latest causal observation. This makes approach, alignment, closure, lift, and the start of transport transform with blue by construction. It does not claim that the complete manipulation problem is invariant: qpos, qvel, blue world pose, the destination, and buffered red remain inputs because reachability, phase, and contact geometry still matter.

## Evidence and expected benefit
In `demo10100`, blue begins near (-0.360, 0.195, 0.020) and TCP descends at indices 420-460 before lift at 480. In `demo10105`, blue begins near (-0.349, 0.203, 0.020), with descent at 410-450 and lift at 470. These are observed successful motions, not shifted-object tests. The expected benefit comes from the exact coordinate transformation: moving/rotating the observed blue frame moves the decoded pickup geometry while goal-relative conditioning changes appropriately.

## Inputs and arguments
Required arguments are `target_object=blue`, `destination_goal=blue_goal`, and `context_object=red`; adapters use them to select state fields rather than treating them as documentation only. Each of two causal frames has 51 deterministic features: scaled qpos/qvel; TCP and red pose relative to blue; blue and red goal points relative to blue; and blue world position/rotation. The first episode observation may be duplicated by the framework. History remains causal across policy switches.

## Action and conversion
The 10D unnormalized label is blue-frame TCP position in metres, relative rotation as the first two matrix columns, and native gripper scalar. `encode_targets` converts FK poses of demonstrated joint commands with the inverse chunk-start blue pose. `decode_action` projects 6D rotation with Gram-Schmidt, composes with the same chunk reference, and calls `panda_kinematics.solve_ik` from freshly measured qpos. It reports convergence, position/rotation residuals, iterations, limit contacts, and maximum joint change. The executor retains its native gripper sign rule. The framework fits normalization on valid labels in this representation.

## Learning
The registered model is the unchanged standard `DiffusionBackbone`, conditioned on the flattened 2x51 features. The diffusion epsilon loss remains the sole optimization objective; `prior_loss` is a differentiable zero and there is no auxiliary head. All modules are registered and covered by optimizer, EMA, and checkpoint state. The fixed H2/16/8, 20 Hz and numerical budget are unchanged.

## Skill, overlap, and handoff
All twelve complete assigned slices are bound. Thus the policy includes final buffered-red release/retreat and blue acquisition at entry, core blue transport/placement/release, and the successor's buffered-red approach/closure/initial lift. Prefer this package for a displaced source blue before or around pickup. Exit when blue is visibly settled at blue_goal and the gripper is open/clear, or at the later overlap cue when buffered red visibly follows a closed gripper upward.

## Limitations
The blue frame is fixed for one 8-step execution chunk and can be stale after slip. No contact truth, collision constraint, or recovery demonstrations are available. A transformed target may be unreachable; IK diagnostics must not be hidden. Use the goal-frame package for a held blue near destination and the fresh-TCP package for modest off-nominal intermediate tracking. None of the policies establishes recovery after a drop, wrong-object grasp, red displacement, or collision.
