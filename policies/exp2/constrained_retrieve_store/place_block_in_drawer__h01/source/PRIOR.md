# Live object-frame SE(3) transport

## Decision and targeted failure

This policy targets a block pose at the retrieval-to-placement handoff that is displaced or rotated beyond the narrow demonstrated entry support. It learns the commanded TCP pose in the **freshly observed block frame**. Therefore a rigid change to the observed block pose rigidly changes the decoded world TCP target by construction. This is most relevant while the object is plausibly attached: carry, alignment, descent, and opening.

Observed support is not a recovery claim. In demo12100 and demo12102, indices 680-840 show a closed gripper with the object and TCP moving together from the tunnel side toward the drawer. Indices 860-900 show descent, and 900-920 show opening and settling. Entry object poses differ modestly across demonstrations, but no demonstration proves behavior under a large unseen displacement or a premature drop.

## Inputs and calling contract

The caller supplies two closed-schema names: `manipulated_object_pose_field="object_pose"` and `destination_pose_field="target_pose"`. The former selects the action frame; the latter selects the placement goal. They are resolved rather than cached.

Each of two causal frames contains 38 deterministic features:

- object-frame TCP translation divided by 0.5 m and rotation 6D;
- object-frame destination translation divided by 0.75 m and rotation 6D;
- seven arm positions divided by 3 rad and two finger positions divided by 0.04 m;
- seven arm velocities divided by 2.5 rad/s and two finger velocities divided by 0.25 m/s;
- drawer position and velocity divided by 0.3 m and 0.3 m/s.

The full condition has 76 scalars. Relative geometry supplies the intended invariant, while measured qpos/qvel and drawer state retain posture, reachability, gripper, and fixture-motion dependencies. The first episode frame may be duplicated by the framework; causal history is preserved across policy switches.

## Action representation and converters

A 10D slot is `[p_object(3 m), R_object_to_tcp first two columns(6), gripper(1)]`. `encode_targets` obtains each demonstrated commanded TCP pose with `panda_kinematics.commanded_tcp_poses`, uses the object pose in that slot's pre-action observation, and computes `inverse(T_world_object) @ T_world_commanded_tcp`. Labels are fitted and normalized by the framework only after this conversion and only on valid training masks.

At execution, `decode_action` resolves the current object pose again for every slot, Gram-Schmidt projects rotation 6D to SO(3), and computes `T_world_object_current @ T_object_tcp_target`. It calls `panda_kinematics.solve_ik` from freshly measured qpos and appends the learned gripper scalar unchanged. Diagnostics expose convergence, position/rotation residuals, iterations, joint-limit contacts, and maximum joint change. There is no scripted completion or hidden controller.

The conversion check is task-space because IK is nonunique. Exact encoded rotations survive the 6D projection; FK position, rotation angle, and gripper error are compared with 0.005 tolerance.

## Model and gradient path

The mature published `DiffusionBackbone` is unchanged. It receives the flattened 76D causal condition and diffuses 16 slots of 10D actions. The masked epsilon objective is the entire training loss; `prior_loss` is zero because the prior is expressed by coordinates rather than an auxiliary objective. Gradients flow through all backbone parameters. Training remains history 2, horizon 16, execution 8, 20 Hz, DDPM 100/100, 20,000 updates, batch 128, seed 0, and last EMA.

## Bindings, overlap, and handoff

All twelve authorized occurrences use [680,1160), with context starting at 679. Thus every assigned action remains supervised. The 680:840 overlap with retrieval and 900:1160 overlap with closing are not trimmed.

Prefer this policy when the drawer is open, the object is plausibly held, and two frames show common object/TCP motion, especially for a displaced entry. After the object reaches approximately drawer-floor height and no longer follows TCP, switch to `place_block_in_drawer__h02`, whose destination frame does not shift a fixed handle target with an off-centre deposited object. Use `place_block_in_drawer__h03` for a locally perturbed or ambiguous release phase. `close_drawer` may receive in the shared tail but must wait for observed settling/detachment before drawer motion.

## Expected benefit and limitations

Expected benefit: attached transport targets equivary with block displacement instead of replaying memorized world poses. This expectation follows from the converter, not from rollout evidence.

The object pose must be reliable, and common motion/closed fingers do not prove contact. After release this reference is intentionally weak: an off-centre block would also shift retreat and handle targets. The policy does not recover unseen slips or drops, guarantee cavity clearance, or guarantee IK reachability. Unconverged or limit-contact IK is reported rather than concealed. No augmentation or invalid relabeling is used.
