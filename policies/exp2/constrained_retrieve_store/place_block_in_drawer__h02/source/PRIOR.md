# Live destination-frame cavity anchoring

## Decision and targeted failure

This policy targets displacement of the open drawer and its placement goal. It learns every commanded TCP target relative to the current `target_pose`, so the decoded alignment, descent, release, retreat, and fixture-relative handle transition rigidly follow a changed observed destination by construction.

Observed evidence establishes the nominal sequence only. In demo12100, indices 820-900 converge above and descend toward the target near x=-0.175, y=0. At 920 the object is at drawer-floor height while TCP remains above it; 940-960 retreat upward; 1080-1159 approach the handle. demo12102 and demo12104 show different opening/settling timing around 895-905. These successes do not prove unseen destination-displacement performance.

## Inputs and typed arguments

The caller supplies `destination_pose_field="target_pose"` and `manipulated_object_pose_field="object_pose"`. Both are required closed-schema fields and select actual observed vectors. The destination is resolved in both causal frames and again at every executed slot.

Each causal frame has 38 features:

- destination-frame TCP translation divided by 0.75 m and rotation 6D;
- destination-frame block translation divided by 0.75 m and rotation 6D;
- seven arm positions divided by 3 rad and finger positions divided by 0.04 m;
- seven arm velocities divided by 2.5 rad/s and finger velocities divided by 0.25 m/s;
- drawer position/velocity divided by 0.3 m and 0.3 m/s.

The destination-canonical geometry exposes placement progress. Measured qpos/qvel and drawer state retain reachability, posture, gripper and fixture-motion dependencies that a relative frame alone cannot remove. The framework supplies two causal frames, duplicates only the initial episode observation, and preserves history across switches.

## Action representation and conversion

An 8D slot is `[p_destination(3 m), q_destination_to_tcp(wxyz,4), gripper(1)]`. During training, the demonstrated command is converted from native joint targets by `panda_kinematics.commanded_tcp_poses`. For every prediction slot, `encode_targets` computes `inverse(T_world_destination_pre_action) @ T_world_commanded_tcp`; `pose_vector` supplies a canonical unit quaternion with nonnegative w. Only valid authorized labels fit the framework representation normalizer.

At execution, the quaternion is normalized, the live destination is read from the fresh current observation, and `T_world_target = T_world_destination_current @ T_destination_tcp`. `panda_kinematics.solve_ik` starts from freshly measured qpos. Its seven absolute joint targets receive the unchanged learned gripper scalar. The decoder reports convergence, task residuals, iterations, limit contacts and maximum joint change and contains no task-completing script.

Task-space conversion uses 0.005 tolerance across FK position metres, rotation radians and gripper scalar because joint solutions are nonunique. Quaternion normalization is exact on encoded labels; an identity fallback only makes a degenerate sampled quaternion finite.

## Model and learning

The published `DiffusionBackbone` is unchanged. Its 76D condition is the two-frame feature tensor flattened; it predicts epsilon for 16 by 8 represented actions. Masked epsilon diffusion loss remains active and is the full objective. There is no auxiliary head and `prior_loss` is zero: the prior is fully implemented by the input/action coordinates and live-frame decoder. All backbone parameters receive diffusion gradients and are included in optimizer, EMA and checkpoint.

The numerical recipe remains fixed: history 2, horizon 16, execution 8, 20 Hz, DDPM 100 train/inference steps, batch 128, 20,000 updates, seed 0, and last EMA selection.

## Bindings and handoff

Every demonstration `demo12100` through `demo12111` binds the full [680,1160) interval with context 679. This preserves 160 retrieval-overlap actions (680:840) and 260 close-overlap actions (900:1160).

Prefer this policy whenever the destination/drawer is displaced or the skill is at alignment, lowering, release, retreat, or handle transition. It is the cover for the object-frame prior after deposit: an off-centre resting block does not move this policy's handle target. If a plausibly attached block enters far outside demonstrated destination-relative support, prefer `place_block_in_drawer__h01` for transport. If the geometry is near nominal but release phase is ambiguous, prefer `place_block_in_drawer__h03`. close_drawer waits for two-frame settling and detachment evidence.

## Expected benefit and limitations

Expected benefit: a rigidly displaced observed destination yields a correspondingly displaced Cartesian plan without relearning world coordinates. This is a structural expectation, not final-test evidence.

The destination pose must be reliable and rigidly related to the drawer cavity/handle geometry. Large object displacement at entry is still a learned extrapolation rather than an action invariant. The policy cannot prove attachment, recover a premature drop, guarantee cavity collision clearance, or guarantee that a shifted goal is reachable. All IK failures remain explicit. No augmentation is used.
