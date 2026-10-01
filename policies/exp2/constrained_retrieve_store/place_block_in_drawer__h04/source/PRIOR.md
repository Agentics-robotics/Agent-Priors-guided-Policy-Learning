# Proximity-gated object-to-destination SE(3) frame

## Decision and targeted failure

This policy targets a displaced held-block entry that must continue through the release boundary. A single live object frame makes attached transport equivariant, but after release an off-centre resting block is a semantically weak anchor for retreat and the fixture-relative handle transition. The new action frame is therefore exactly the observed object pose while the object and TCP centers are close, and smoothly becomes exactly the observed destination pose as they separate.

The validation facts motivate, but do not prove, the mechanism. The immutable object-frame h01 passed 5/12 nominal demonstrated-entry validations, while destination-frame h02 and local-correction h03 each passed 12/12. The available protocol does not isolate the cause, and there is no out-of-distribution result. Demonstrations do show the intended geometric transition: in demo12100 the object remains close to the TCP through index 900, opens and separates by 910, and rests at drawer-floor height by 920. demo12102 opens at 900 and settles by 905; demo12104 opens at 895 and settles by 900.

## Hybrid reference and structural invariance

Let `D`, `O`, and `E` be the live destination, object, and measured TCP world poses. Let `d` be the Euclidean distance between the object and TCP origins. The object weight is

- `w=1` for `d <= 0.025 m`;
- `w=0` for `d >= 0.055 m`;
- cubic smoothstep of `(0.055-d)/(0.055-0.025)` between those distances.

The reference translation is `(1-w) p_D + w p_O`. Its rotation is
`R_D Exp(w Log(R_D^T R_O))`, a proper-rotation geodesic interpolation. Thus the reference is exactly `O` at `w=1` and exactly `D` at `w=0`, without averaging rotation-matrix entries.

During close, attached-like carry and descent, a rigid translation or rotation of the observed object rigidly transforms the decoded world TCP target by construction. This is the required object-displacement invariant and covers carry, alignment, descent, and the final attached approach to release. After observed separation, an off-centre deposited block no longer moves retreat, reorientation, or handle targets because the reference is the destination. The transition band is only a causal geometric prior; it is not a contact detector.

## Inputs and typed arguments

The caller supplies a closed-schema `manipulated_object_pose_field="object_pose"` and `destination_pose_field="target_pose"`. Both are resolved in every causal observation and again for every executed slot.

Each of the two causal frames has 48 deterministic features:

- hybrid-reference-frame TCP translation divided by 0.75 m and rotation 6D (9);
- destination-frame object translation divided by 0.75 m and rotation 6D (9);
- TCP-frame object translation divided by 0.5 m and rotation 6D (9);
- seven arm positions divided by 3 rad and two finger positions divided by 0.04 m (9);
- seven arm velocities divided by 2.5 rad/s and two finger velocities divided by 0.25 m/s (9);
- drawer position and velocity divided by 0.3 m and 0.3 m/s (2);
- object weight `w` (1).

The full condition has 96 scalars. Destination and object geometry expose placement progress and proximity change. Measured qpos/qvel, finger state, and drawer state retain robot configuration, reachability, release, and fixture-motion dependencies that a relative frame cannot remove. The framework supplies two causal frames, duplicates only the initial episode observation, and preserves history across policy switches.

## Cartesian action labels and execution conversion

An 8D represented slot is `[p_reference_to_tcp(3 m), q_reference_to_tcp(wxyz,4), gripper(1)]`. `encode_targets` converts each demonstrated native joint command to its commanded world TCP pose with `panda_kinematics.commanded_tcp_poses`. For every slot, it constructs the hybrid reference from that slot's pre-action `target_pose`, `object_pose`, and measured `tcp_pose`, then computes `inverse(T_world_reference) @ T_world_commanded_tcp`. `pose_vector` provides a canonical unit quaternion with nonnegative w. Only valid masked labels from the authorized bindings fit the framework representation normalizer.

At execution, `decode_action` recomputes the hybrid reference from the fresh current observation for each of slots 1 through 8. It normalizes the represented quaternion, left-multiplies the relative pose by the fresh world reference, and passes only that world TCP target to `panda_kinematics.solve_ik`, initialized from freshly measured qpos. The seven absolute joint targets receive the learned gripper scalar unchanged. Diagnostics report IK convergence, position and rotation residuals, iterations, lower/upper limit contacts, maximum joint change, posture diagnostics, live object weight, and object/TCP distance. An unconverged result is reported rather than silently altered, and the decoder contains no task-completing script.

The independent conversion check is task-space because Panda IK is nonunique. For an encoded demonstration target and the same pre-action state, the hybrid reference is deterministic and quaternion normalization is exact. FK position error in metres, rotation angle in radians, and unchanged gripper error are compared with tolerance 0.005.

## Model, gradients, and normalization

The mature published `DiffusionBackbone` is unchanged. It receives the flattened 96D two-frame condition and predicts epsilon for 16 slots of 8D represented actions. The masked epsilon diffusion objective is the full loss. `prior_loss` is zero because the prior is implemented by causal input features and action coordinates rather than an auxiliary head. Diffusion gradients update all backbone parameters, which are included in the optimizer, EMA, and saved state.

The fixed recipe remains history 2, prediction horizon 16, execution chunk 8, 20 Hz, DDPM 100 training and inference steps, batch 128, 20,000 updates, seed 0, and final EMA selection. Input scales are fixed physical-unit scales. Action normalization is fitted by the framework after hybrid-frame label construction and only from valid authorized training labels. No augmentation or relabeling is used.

## Bindings, overlap, and handoff

Every demonstration `demo12100` through `demo12111` binds the entire `[680,1160)` interval with context start 679. This preserves all assigned responsibility and both substantial action-supervised overlaps: 680:840 with retrieval and 900:1160 with closing.

Prefer h04 when the drawer is open and a plausibly held, possibly displaced block begins close to the TCP and must be carried through release. Continue while the observed close-to-separated transition is coherent. Use h02 when the block is already separated or deposited and purely fixture-anchored completion is appropriate. Use h03 for a near-nominal local release/retreat ambiguity, and h01 only for a reliably attached interval where an always-object frame is desired. `close_drawer` may receive in the shared tail, but drawer motion waits for two causal observations showing a stationary floor-height block separated from the TCP; the weight alone is insufficient.

## Expected benefit, failure signature, and limitations

Expected benefit is structural, not observed performance: object-dominant transport should move with a displaced held block, while destination-dominant post-release targets should not follow the deposited block. A falsifying signature is reference-weight collapse while the object should still be attached, or weight persistence/chatter after release followed by inconsistent retreat or handle motion.

Object/TCP distance is not contact truth. A false close-proximity cue, noisy pose in the transition band, or an unseen premature drop can choose an inappropriate frame. Smooth blending limits discontinuity but does not recover a dropped object. Large displacements can violate cavity clearance, Panda reachability, collision constraints, or demonstrated posture support. The policy assumes reliable object and destination poses, cannot prove attachment or settling, and retains a long post-release transition learned from only twelve narrow successful demonstrations. No out-of-distribution success is claimed.
