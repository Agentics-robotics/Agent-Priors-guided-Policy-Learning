# Prior: dual-reference position with absolute world orientation

## Decision and targeted failures

This independent policy targets three observed or expected failures: source displacement during acquisition, final position outside the bowl footprint, and loss of upright orientation during an extended rollout. Each demonstrated TCP command is represented redundantly by its position in the freshly observed source-container frame and its position in the freshly observed destination-bowl frame. The decoder reconstructs both world positions and blends them with a deterministic causal weight. Open or low acquisition uses the source branch; closed clear-lift transport uses the bowl branch. Orientation is a separate absolute world-frame 6D target rather than an increment from the measured TCP.

The design is motivated by complementary in-distribution validation facts, not a causal performance claim. h02 finished over the bowl in 12/12 runs but upright in 0/12. h03 finished upright in 12/12 and over the bowl in 7/12. h03 had the best exit rate at 7/12. Those outcomes motivate destination-relative absolute positioning plus non-incremental orientation, but the policy is newly trained and may be worse.

Demonstrations show phase variation. Demo12300 first commands closure at index 148 and has source z about 0.029 m at 180. Demo12303 commands closure at 136, is closed by observation 140, and has source z about 0.056 m at 180. Demo12306 is also closed by 140 and reaches source z about 0.060 m at 180. All stage near the bowl by 320-360. Initial source centers span only about x=-0.443 to -0.458 m and y=-0.202 to -0.214 m, so deployment displacement is not demonstrated.

## Causal inputs and call arguments

The required typed arguments are `source_object="container"` and `destination_object="bowl"`. They select observed pose fields and are resolved on every execution step. They are not actuator inputs or phase/contact labels.

Each causal frame has 80 fixed-scale features: source-to-TCP, bowl-to-TCP and source-to-bowl poses; world source, bowl and TCP positions; source and TCP world orientations; measured qpos/qvel; finger width; particle centroid in the source frame; source and TCP translations across the two frames; the declared local grasp site; and the deterministic phase weight. The oldest frame has zero motion by construction. The framework duplicates the initial observation at episode start and preserves history across switches.

Measured joints, velocities and world positions deliberately remain in the condition. Relative coordinates do not remove Panda reachability, joint-limit, collision, gravity or configuration dependence. Particle centroid and co-motion are context only and are not contact truth.

## Causal phase weight

The destination weight is the product of two clipped ramps. A closure factor rises as measured finger width falls below 0.05 m and saturates over 0.02 m. A lift factor rises as the source origin moves above the bowl-origin proxy plus 0.003 m and saturates over 0.04 m. Thus table-supported/open behavior uses the source position branch, while clearly closed and elevated behavior uses the bowl branch; intermediate values smoothly blend.

The weight is deterministic and recomputed from each action observation for interpretation and from each fresh execution observation for decoding. It is not learned, not stateful and not advanced during denoising. It cannot certify a secure grasp. Because both encoded position branches reconstruct the identical demonstration target, changing the weight does not introduce a discontinuity or alter a valid label.

## Action representation and conversion

The learned action has 13 dimensions:

1. source-frame TCP position xyz in metres;
2. destination-bowl-frame TCP position xyz in metres;
3. absolute world TCP orientation as the first two rotation-matrix columns (6D);
4. the native gripper scalar.

`encode_targets` obtains world commanded TCP poses through `panda_kinematics.commanded_tcp_poses`. For every slot it uses that slot's pre-action source and bowl observations to compute both exact local positions and encodes the command's world rotation. These pre-action observations and commanded poses are training labels only; deployment forward receives only two causal observations.

`decode_action` composes both sampled local positions with the fresh observed source and bowl poses, blends the resulting world points by the fresh causal weight, robustly projects sampled 6D orientation to SO(3), and constructs one world TCP pose. It then calls `panda_kinematics.solve_ik` from freshly measured joints and appends the unchanged gripper scalar. Diagnostics expose convergence, position and rotation residuals, iterations, limit contacts, joint change, posture status, both branch positions, the blend weight and the world target. An unconverged result is reported rather than replaced.

For an encoded demonstration label, both branches recover the same world position, so every blend weight recovers that position. The encoded 6D columns recover the same commanded rotation. The independent conversion check therefore compares URDF FK of IK output to the commanded TCP in task space; it does not require matching a nonunique joint solution.

## Model, normalization and supervision

The standard `DiffusionBackbone` is unchanged. It consumes the flattened two-frame 160-dimensional condition and predicts epsilon for 16 slots of 13D actions. No auxiliary head is used. Masked epsilon diffusion is the only optimization objective and `prior_loss` is a differentiable zero. All learned parameters are registered in the backbone and included in optimizer, EMA and checkpoints.

The host fits representation midpoint/half-range normalization only from valid 13D labels. Input channels use declared physical scales rather than evaluation-fitted statistics. Temporal settings remain two causal frames, prediction horizon 16 with slots t-1 through t+14, execution slots 1 through 8, and 20 Hz. Formal training remains the fixed 20,000-update DDPM/EMA recipe.

All twelve authorized [0,360) segments are action-supervised. The complete [180,360) interval remains substantial overlap with `controlled_pour_and_right`, including closed lift, carry, high staging and the first small tilt.

## Expected benefit, handoff and limitations

Before clear lift, translating or rotating the source while preserving handle geometry produces the same source-relative approach coordinates by construction. After clear lift, moving the destination within reach produces the same bowl-relative staging coordinates. Absolute world orientation avoids repeated rotational increment accumulation and directly represents the demonstrated near-upright carry posture. These are expected, falsifiable benefits; no out-of-distribution evaluation supports them yet.

Enter with reliable source/bowl poses and either an open table-supported source or a nearby closed state with plausible source/TCP co-motion. Handoff requires observed closure, co-motion, clearance, upright posture and high bowl-relative staging. Never treat the blend weight as proof of contact.

The weight can be wrong after a miss, slip, unusual support height or noisy pose. Redundant 13D labels may reduce sample efficiency. The source branch cannot make unreachable displacement feasible, and the bowl branch does not plan collision-free transport. Absolute world orientation assumes the grasp geometry and gravity direction remain relevant. No demonstration establishes failed-grasp recovery, slip recovery, obstacle avoidance, large object rotation or robustness to severe intermediate-state errors.
