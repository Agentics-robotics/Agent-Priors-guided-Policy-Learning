# Phase-adaptive drawer/red-frame Cartesian diffusion

## Decision

H04 is a complementary displacement prior. It uses the articulated drawer frame for the high approach, drawer-front acquisition, constrained pull, release and early retreat, then uses the observed red-block frame for the red approach, descent, closure and initial lift. Its purpose is to make the early responsible motion follow a reachable displaced/rotated drawer calibration and the late pickup motion follow an independently displaced red block. This benefit is by construction in the Cartesian conversion but is not supported by an out-of-distribution rollout result.

The original in-distribution validation used nominal training starts and the final rule `drawer_position >= 0.29`. H01 passed 3/12, H02 passed 6/12 and H03 passed 12/12. H04 is not an assertion that H03 needs replacement at nominal geometry. It adds a different expected failure profile for cases where reliable object frames and displacement matter.

## Observable active-frame mechanism

For each physical state, the drawer frame starts at the caller's `drawer_origin_world_m`, uses `drawer_orientation_wxyz`, and translates along drawer-local -X by the fresh measured `drawer_position`. The red frame is the fresh observed red pose. The red frame is active only when:

1. `drawer_position >= red_phase_min_open_m`, and
2. either world-Z `tcp_z - red_z >= red_phase_clearance_m` or world-XY `distance(tcp, red) <= red_phase_xy_radius_m`.

Training binds these three gate values to 0.280 m, 0.280 m and 0.140 m. In demo1000, the drawer is fully open at 230 while the low TCP is still at the front, the front is released by 240, retreat rises through 280, and TCP z is about 0.369 m at 300 before lateral red approach. Later the XY-neighborhood branch keeps the red frame active through descent, close and lift. The gate uses only causal physical state and typed caller calibration. It has no simulator contact truth, future state or hidden recurrent phase.

The gate is recomputed for every demonstrated label from that slot's pre-action observation and for every executed action from the fresh current observation. Thus encode and decode share exactly the same reference-frame semantics. A changed trajectory can nevertheless cause a different gate than the model predicted for a future slot; this is a documented failure mode rather than hidden state.

## Causal inputs and retained dependencies

Each of two causal frames has 49 values:

- normalized measured qpos [9] and qvel [9], including both finger joints;
- drawer-frame TCP pose as position/0.5 m and 6D rotation [9];
- drawer-frame red pose in the same form [9];
- red-frame TCP pose in the same form [9];
- drawer opening/desired opening, velocity/0.2 m/s and remaining opening/desired opening [3];
- the observable active-frame gate [1].

This retains arm configuration, velocity, finger state, reachability cues, drawer progress and relational contact cues. Relative coordinates do not remove physical reach, collision or contact dependencies. At initial episode time only, the framework duplicates the first observation; two-frame causal history otherwise persists across policy switches.

## Cartesian actions and conversion

`encode_targets` converts each demonstrated seven-joint command to its world TCP target with the verified public URDF FK. It expresses that target in the active frame as position [3] in metres, the first two columns of its relative rotation matrix [6], and the unchanged native gripper scalar [1]. The learned action dimension is 10.

`decode_action` projects sampled 6D rotation with stable Gram-Schmidt, composes the relative target with the active frame to obtain a world TCP pose, and calls only `panda_kinematics.solve_ik(target, fresh_qpos, robot)`. It appends the represented gripper scalar unchanged. Diagnostics report active frame, gate geometry, IK convergence, position and rotation residuals, iterations, joint-limit contacts, posture quantities and world target pose. The executor applies the common gripper sign rule. For a demonstration state, encode followed by decode reconstructs the commanded task-space pose; nonunique arm joints are resolved by the verified IK posture prior.

Drawer-frame targets constructively transfer approach/pull/release/early-retreat geometry under caller-calibrated drawer displacement. Red-frame targets constructively transfer the later approach/grasp/lift geometry under observed red displacement. This does not guarantee either displacement is reachable or collision-free.

## Model, gradients and normalization

The standard `DiffusionBackbone` is unchanged. It receives the flattened 98-value two-frame condition and predicts epsilon for `[B,16,10]`. Masked epsilon diffusion is the complete learned objective. There is no auxiliary head, synthetic augmentation or task-completing decoder script; `prior_loss` is differentiable zero. The backbone is the only learned module and is therefore registered, optimized, checkpointed and EMA-tracked by the framework.

The framework fits representation normalization only on valid active-frame 10D labels, not on native joint scales. H2/16/8, slots t-1 through t+14, execution slots 1 through 8 beginning at current t, 20 Hz, DDPM100/100, batch 128, 20,000 updates, seed 0 and last-EMA selection remain fixed.

## Coverage, evidence and handoff

All 12 bindings supervise every action in `[0,470)`. No action is trimmed by the gate. Actions 230:470 remain the complete overlap with `red_transfer`: 240 actions and 12.0 seconds per occurrence, 2,880 actions total. Demo1000 indices 200/230/240/280/300 show pull completion, release and retreat; 400/445/460 show red alignment, closure and lift. Demo1001 and demo1005 retain the same responsibilities with red y near -0.077 m and -0.055 m respectively.

`red_transfer` can receive once drawer_position exceeds 0.26 m and red is exposed because its supervision begins at 230. The stronger nominal handoff additionally observes a stable drawer, released front, closed fingers and red/TCP upward co-motion near 470. The active-frame gate is not a handoff proof.

## Applicability and limitations

Prefer H04 when drawer calibration and red pose are reliable and reachable scene displacement is the dominant concern. Prefer H03 at nominal geometry when object-frame estimates or switch cues are unreliable.

No shifted drawer or independent red displacement was demonstrated or validated. The hard gate can switch incorrectly, chatter, or reinterpret future chunk slots if actual execution differs from nominal. Noisy red pose can move late targets and bad drawer calibration can move early targets. Large shifts/rotations, unreachable poses, collisions, missed front contact, blocked pulls, misgrasps, drops and recovery are outside evidence. A closed gripper does not prove grasp, and IK convergence does not prove safe physical interaction.
