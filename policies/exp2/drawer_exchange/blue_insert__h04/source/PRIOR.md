# Source-to-destination corridor Cartesian policy

## Decision and targeted failure

`blue_insert__h04` targets translation of the free blue block and changed source-to-destination separation without applying one source frame to the entire long skill. The twelve demonstrated segment entries place blue only around x=-0.414..-0.387 m and y=0.287..0.315 m. The new action coordinate makes near-source pickup targets translate with the causally observed blue by construction, keeps near-goal insertion targets anchored to the calibrated drawer goal, and continuously warps intermediate transport targets.

The revision evidence is nominal and in distribution: h01 passed 10/12, h02 passed 12/12, and h03 passed 10/12 under the fixed validation protocol. Every original failure retained drawer-open and red-on-pad but missed blue-inside. This motivates, but does not validate, a coordinate combining source and destination relationships. No out-of-distribution result is available.

## Calling contract and causal inputs

Required typed arguments are `source_object="blue"` and `destination_field="blue_goal"`. They select the current observed `blue_pose` and calibrated `blue_goal`; arbitrary positions, formulas and actuator commands are not accepted. At each replan, the current blue position `b` and goal `g` are frozen in chunk context for the eight executed actions and refreshed at the next chunk.

Each of the two causal observations contributes 53 fixed-scale features:

- seven arm positions, two finger positions, seven arm velocities and two finger velocities;
- TCP-to-blue and TCP-to-goal position residuals;
- TCP world orientation in 6-D;
- blue-to-goal displacement, blue world position and blue world orientation in 6-D;
- TCP-to-red displacement and red-to-red_goal residual;
- the world blue_goal and drawer position/velocity.

The resulting 2x53 input retains robot configuration, reachability, predecessor-object state, gripper state, destination and drawer dependencies. Two frames expose causal motion. History crosses policy switches; only the initial episode observation is duplicated by the framework.

## Cartesian action and exact label conversion

Each prediction slot has 11 values:

`[alpha, residual_position_world_m(3), world_rotation_6d(6), gripper(1)]`.

For a demonstrated commanded TCP position `p`, the adapter uses the causal chunk endpoints:

`d = g - b`

`alpha = clip(dot(p - b, d) / dot(d, d), 0, 1)`

`anchor = b + alpha * d`

`residual = p - anchor`.

If `dot(d,d) <= 1e-4 m^2`, the destination and observed blue are considered near-coincident for this coordinate and `alpha=1`. The residual still makes reconstruction exact. Commanded TCP poses come from FK of the demonstrated native joint commands, not from lagging measured TCP. The native gripper scalar is retained. Futures are used only by the framework to supply supervised commanded actions; deployment input is causal.

At decode, sampled alpha is clipped to [0,1], the world position is reconstructed as `b + alpha*(g-b) + residual`, and sampled 6-D orientation is deterministically projected to SO(3). The resulting world TCP pose is passed only to `panda_kinematics.solve_ik` from freshly measured joints. Diagnostics publish convergence, position/rotation residuals, iterations, limit contacts, posture status, alpha and target pose. Slot 0 represents t-1; the host executes slots 1..8 as t..t+7.

For valid labels, progress plus residual reconstructs the Cartesian target exactly before numerical IK. If the represented output is held fixed while source translates by `delta` and the goal is fixed, the decoded position changes by `(1-alpha)*delta`: contact/pickup targets near alpha=0 translate almost fully, destination targets at alpha=1 do not move, and intermediate targets warp proportionally. This is the intended falsifiable benefit. It does not imply that alpha remains correct under unseen states.

## Model, gradients and normalization

The mature `DiffusionBackbone` is unchanged. It receives the flattened 106-D causal condition and diffuses 11-D action sequences over the fixed 16-slot horizon. The masked epsilon diffusion loss is the full objective. There is no auxiliary head; `prior_loss` is differentiable zero. All trainable parameters are registered in the backbone and therefore included in optimization, EMA and checkpoints.

Input scaling is fixed in physical units as documented in the adapter. The framework fits representation midpoint/half-range normalization only from valid transformed labels in the twelve authorized bindings. No spatial augmentation is used because arbitrary transforms could violate Panda reachability or drawer/red-pad geometry.

## Coverage, overlap and handoff

All twelve assigned `[600,stop)` ranges are bound without trimming. This preserves every shared `[600,850)` action with `red_transfer`: 250 actions per trajectory and 3,000 total labels spanning red lowering/release, retreat, cross-scene transit, blue approach, closure and lift. Inspected demo1000 and demo1005 steps also cover transport, insertion, release and high retreat. The complete skill responsibility and unchanged success definition are retained.

Prefer h04 when a translated free blue or changed blue-to-goal separation is the main risk. Prefer the existing h02 for nominal or stable carried-blue states near the drawer where destination anchoring dominates. H01 remains complementary for source rotation, which h04 does not make invariant, and h03 for small fresh-TCP tracking deviations. There is no learned successor; final readiness is the unchanged simultaneous drawer-open, red-on-pad and blue-inside condition.

## Expected failure signature and limitations

A wrong alpha can choose the wrong anchor: predecessor red-release targets can shift toward blue, or pickup can be pulled toward the drawer before a secure grasp. Large shifts can be unreachable or collision-prone; the coordinate is not a motion planner. World orientation is not invariant to source or drawer rotation. The observed blue endpoint is frozen over each execution chunk, so slip is incorporated only at replanning. Finger closure, proximity and apparent co-motion do not establish contact. Demonstrations do not cover drops, re-grasps, drawer-wall impacts or recovery, and the original validation does not establish any OOD benefit.
