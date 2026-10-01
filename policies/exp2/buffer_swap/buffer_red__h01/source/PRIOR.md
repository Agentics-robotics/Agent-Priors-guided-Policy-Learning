# Red-anchored SE(3) Diffusion Policy

## Decision and target failure

This independent policy targets deployment displacement of the red block. The commanded Cartesian pose for every prediction slot is learned relative to the red pose observed at chunk start. This makes approach, alignment, closure, lift and the local held-object relation transform with that observed pose by construction. It does **not** assert that the fixed-buffer transport decision is invariant: the model receives the red-to-buffer relation and must learn that part.

Prefer this model when red is outside or near the edge of the demonstrated initial patch but remains observed and reachable. Prefer `buffer_red__h03` when destination relocation or final placement dominates, and `buffer_red__h02` for uncertain mid-skill TCP/phase entries.

## Evidence and expected benefit

Observed evidence is from successful demonstrations only. In demo10100 red begins near (-0.355,-0.209), while demo10103 begins near (-0.351,-0.204). Their index-40 TCP commands are above their respective red positions; closure is near 100, red/TCP rise together at 120-160, translate toward the center at 180-220, lower at 240-260 and release by 280. The tail then retreats and acquires blue through index 479. This supports object-relative structure and overlap labels, not unseen-error recovery.

The falsifiable expected benefit is reduced sensitivity of red approach/grasp targets to red-pose displacement compared with a world-action policy. Failure remains likely when the displacement makes the red-to-buffer route, collisions or Panda posture unlike training.

## Inputs and arguments

The closed call contract requires `target_object="red"`, `successor_object="blue"`, and `buffer_pose_world=[x,y,z,qw,qx,qy,qz]` in world metres/wxyz. The authorized labels bind the buffer to `[-0.181,0,0.02,1,0,0,0]`. Object names actually select observation fields; the buffer defines relational features.

For each of two causal observations, `build_inputs` emits 60 physical/analytic features: scaled qpos/qvel and fingers; TCP, blue and buffer poses relative to red; world TCP pose; and red/buffer world positions. Positions are divided by 0.5 m, arm qpos by 3 rad, arm qvel by 2.5 rad/s and fingers by 0.04 m. Rotations use 6D matrix columns. These fixed unit scales are not fitted on evaluation data. World and robot channels intentionally preserve reachability and IK dependencies that relative coordinates cannot remove. Episode-start history duplication is framework-owned; history remains causal across policy switches.

## Action conversion

The learned 10-vector is local translation in metres, local rotation-6D, and native gripper scalar. `encode_targets` uses FK of each demonstrated seven-joint command and computes `inverse(T_red_chunk) @ T_tcp_command`. The same chunk anchor comes from the latest causal red observation and remains fixed for the eight executed outputs. `decode_action` safely Gram-Schmidt projects sampled 6D values, computes `T_red_chunk @ T_local`, and calls `panda_kinematics.solve_ik` from freshly measured qpos. It reports convergence, position/rotation residuals, iterations, limit contacts and maximum joint change. It never substitutes another controller. The gripper sign is unchanged. Representation midpoint/half-range normalization is fitted by the framework on valid red-frame labels only.

The standard DiffusionBackbone is unchanged. There is no auxiliary head; masked epsilon diffusion is the full objective, and `prior_loss` is differentiable zero.

## Temporal scope, overlap and handoff

History is 2, horizon 16 represents t-1 through t+14, and execution uses slots 1-8 at 20 Hz. Every assigned action in all 12 segments is bound. The 210-230 action overlap covers red lowering/release, opening, retreat, transit to blue, closure and initial blue lift. Core handoff readiness requires observed red stability near the actual buffer and open/clear fingers; a later switch through blue lift is also label-supported. Preserve causal history at switching.

## Limitations

The policy has no contact truth, images, collision model or recovery demonstration. Closed fingers are not grasp proof. A stale/noisy red pose corrupts the reference frame. The red frame is moving while held, though it is frozen within each chunk and refreshed at replanning. Large displacements can be unreachable or create unsupported obstacle relations. IK diagnostics are feasibility cues, not task-success guarantees. The unchanged task success definition remains `red_at_goal AND blue_at_goal`; this skill only supplies the assigned red-buffer responsibility and overlap.
