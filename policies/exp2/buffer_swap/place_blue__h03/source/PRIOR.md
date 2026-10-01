# Fresh-TCP-frame incremental Cartesian servo prior

## Decision and target failure
This package targets modest intermediate-state and handoff offsets. Each demonstrated command is represented as a local transform from that slot's measured pre-action TCP. During execution, each sampled slot is composed with the freshly measured TCP rather than accumulated through predicted states. This supplies deterministic per-step re-anchoring without hidden recurrent phase state.

## Evidence and expected benefit
Valid entries differ: `demo10100:260` has red/TCP near z=0.023 m, whereas `demo10105:250` is near z=0.047 m and still descending. The slices then contain retreat, blue approach/closure/lift, carry, placement/release, retreat, and buffered-red pickup. Two causal frames provide object motion, gripper width, robot velocity, and relational phase cues. Exact local labels plus fresh decoding are expected to reduce compounding tracking error and tolerate modest plausible offsets. Successful demonstrations do not prove arbitrary recovery.

## Inputs and arguments
Required arguments `target_object=blue`, `destination_goal=blue_goal`, and `context_object=red` select the actual observation fields. Each causal frame has 45 features: scaled qpos/qvel; blue and red poses relative to measured TCP; blue and red goals in the TCP frame; and TCP world position/6D rotation. Thus local invariance retains world reachability and robot configuration. The framework owns initial observation duplication and preserves history across switches.

## Action and conversion
The 7D label is local TCP-frame translation in metres, local rotation vector in radians, and native gripper scalar. For each training slot, `encode_targets` uses the measured pre-action TCP from `action_observations` and the FK pose of the demonstrated joint command. These futures are labels only. `decode_action` builds the same transform from the represented slot, composes it with `current_observation.tcp_pose`, and calls `panda_kinematics.solve_ik` from fresh qpos. It reports IK convergence/residuals/limits and local correction magnitudes. There is no accumulation across slots.

## Learning
The standard `DiffusionBackbone` consumes flattened 2x45 features and predicts epsilon for 7 action channels. The masked diffusion loss is active; no auxiliary head is present and `prior_loss` is zero. All trainable state is therefore in the registered backbone and included in optimizer, EMA and saved state. H2/16/8, 20 Hz and the numerical recipe remain fixed.

## Skill and handoff
Every action in all twelve assigned slices is supervised, preserving both large overlaps and the complete core blue relocation. Prefer this policy for an early/late but plausible handoff or after a small execution offset. Exit on blue settled at blue_goal with gripper open/clear, or use the later cue of buffered red visibly moving upward with the gripper.

## Limitations
Local increments can be phase-aliased and are not anchored by construction to gross source or destination displacement; use h01 or h02 respectively. Fresh composition cannot infer contact or recover a dropped block, wrong-object grasp, collision, or disturbed red. Large offsets may require labels outside demonstrated support, and IK feasibility remains dependent on current joints. Repeated denoising calls never update physical history.
