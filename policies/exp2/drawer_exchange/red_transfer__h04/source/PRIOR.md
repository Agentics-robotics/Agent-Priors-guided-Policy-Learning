# Phase-routed semantic-anchor Cartesian prior

## Decision and target failure

This independent policy targets three displacement failures while avoiding a single inappropriate frame over the whole long segment: moved red during pickup, moved outside pad during placement, and moved blue during the successor pickup. A required semantic phase routes the same 10-D Cartesian target through the anchor relevant to the current responsibility:

- `drawer_clear`: fixed world anchor for gripper opening and high retreat from the drawer;
- `red_acquire`: current observed red SE(3) pose for red approach, contact, closure and elevated lift;
- `red_place_release`: current observed `red_goal` translation with world axes for carried-red transport, descent, release and initial vertical clearance;
- `bridge_to_blue`: newest causal TCP SE(3) pose for a short local post-release bridge;
- `blue_acquire`: current observed blue SE(3) pose for transit, approach, closure and initial lift.

The red- and blue-relative target conversions are SE(3)-equivariant at the Cartesian label/decoder level: transforming the selected observed object frame transforms the decoded world TCP target by the same transform while preserving the learned relative target. Goal centering gives translation equivariance for pad motion. This does not make the full robot/contact problem invariant: measured joints, reachability, furniture, collisions and grasp physics remain relevant.

The fixed validation supports the design choice but not its expected out-of-distribution benefit. The original destination-centered h02 passed 10/12 with zero reported unconverged IK steps; its two failures retained an open drawer but ended with red off the pad. The all-segment red-frame h01 passed 4/12 and had 2,866 unconverged IK steps, while the all-segment local-TCP h03 passed 9/12 and its three failures included drawer closure. Those observations motivate retaining stable world/goal references where appropriate and restricting local TCP coordinates to a short bridge. They do not prove why any original failed, and no displaced-object validation exists.

## Demonstrated support and phase labels

All twelve authorized `[230,850)` segments are retained. Inspected `demo1000` and `demo1001` show a high, open-gripper drawer retreat by about actions 240-300; motion toward red from about 320; red contact/closure near 445 and co-moving lift through 470-500; carried-red transport, descent and release through 500-620; vertical clearing through about 650; a clear bridge toward blue through 650-700; and blue transit, descent, closure and initial lift through 700-850. The five contiguous bindings are `[230,320)`, `[320,500)`, `[500,650)`, `[650,700)`, and `[700,850)`.

These coarse labels are semantic supervision, not an inference-time clock. At deployment the high-level caller must infer phase from current TCP/object/gripper relationships. The split preserves every action exactly once and retains complete action-supervised overlap: `[230,470)` with `open_drawer` and `[600,850)` with `blue_insert`.

## Inputs and invocation

The required arguments are `phase`, `primary_object="red"`, `destination="red_goal"`, and `successor_object="blue"`. The identifiers are closed selectors for observed fields, not caller-supplied poses, code or actuator values. `build_inputs` resolves a fresh anchor from each causal observation for relational features and freezes the newest anchor only for the sampled chunk. Initial episode history may be duplicated by the framework; real H2 causal history is retained across policy switches.

Each 53-D history row contains scaled measured arm/finger positions and velocities; TCP, red and blue poses expressed in the selected phase anchor (position plus rotation 6D); red_goal and blue_goal positions in that anchor; and drawer position/velocity. A 5-D one-hot phase is concatenated directly into model conditioning. This removes irrelevant anchor translation/rotation from the task geometry while retaining robot configuration, finger state, motion and drawer articulation needed for reachability and contact-dependent behavior. No images, contact truth, future observations, action history or recurrent phase memory are inference inputs.

## Action representation and converters

Every slot is `[anchor-frame XYZ metres, anchor-frame rotation 6D, gripper scalar]`. `encode_targets` obtains each demonstrated world TCP target by verified FK of the seven native joint commands, computes `inverse(chunk_anchor) * commanded_tcp`, and stores translation plus the first two relative rotation columns. It does not use future states. The framework fits representation normalization from valid labels after encoding; old native-joint action scales are not reused.

`decode_action` uses the same frozen causal anchor, applies a deterministic finite Gram-Schmidt projection to sampled rotation 6D, composes the relative target into world coordinates, and invokes `panda_kinematics.solve_ik` from freshly measured arm joints. It never substitutes another controller or hides a failed solve. Diagnostics report convergence, position and rotation residuals, iterations, lower/upper limit contacts, maximum joint change and posture settling. The represented gripper scalar is appended unchanged, preserving the executor rule that nonnegative opens and negative closes. Valid encoded rotations round trip exactly up to numerical tolerance; arbitrary degenerate samples use the declared finite projection.

## Model, losses and optimization

The model is the unchanged standard `DiffusionBackbone` with flattened H2 features and the phase one-hot as global conditioning. No auxiliary head or privileged target is used. Masked epsilon diffusion is the total learned objective, and `prior_loss` is a differentiable zero. Thus all trainable parameters are registered in the backbone and receive action-diffusion gradients; optimizer, checkpoint and EMA handling are standard. The fixed history 2, horizon 16, execution 8, 20 Hz, DDPM 100/100, batch 128, seed 0 and 20,000-update recipe are unchanged.

## Handoff and use in the library

At the represented entry, the drawer is above 0.26 m but the fingers can still be closed at the drawer front, so `drawer_clear` is valid. Later starts require the phase matching physical state. `red_place_release` intentionally remains active into the `[600,650)` successor overlap while red is held, touching or newly released. `bridge_to_blue` then clears the placed object without forcing that motion to translate with blue. `blue_acquire` uses blue-relative targets for the final approach, closure and lift.

Nominal readiness for `blue_insert` is red contained on the pad and no longer following TCP, drawer still strictly above 0.26 m, and observed blue/TCP co-motion on initial lift. These are handoff cues only; they do not modify the fixed task success definition. Prefer existing h02 when phase confidence is low. Prefer h04 when the semantic phase is reliable and a displaced red, goal or blue observation makes the corresponding anchor useful.

## Limitations and falsifiable expectation

The expected benefit is better Cartesian target transfer under reachable displacement of the selected red, red_goal or blue anchor than a representation that uses an unrelated frame in that phase. This remains unconfirmed until formal training and evaluation. A wrong or premature phase switch changes coordinate semantics and can cause discontinuous or misdirected motion. Anchors are frozen for eight executed slots, so stale, noisy or fast-moving anchors bias the chunk. The 50-action bridge has less supervision and is local-TCP-relative rather than blue-invariant. Mixed phase semantics can reduce data efficiency and may perform worse than h02 even in distribution.

No anchor representation provides collision avoidance, force/contact feedback, grasp proof, or physical feasibility for unreachable displacement. The demonstrations contain narrow initial object ranges and a fixed red_goal. No successful demonstration establishes recovery from a missed red or blue grasp, an off-pad release, unintended drawer contact, changed friction, or a target behind an obstacle.
