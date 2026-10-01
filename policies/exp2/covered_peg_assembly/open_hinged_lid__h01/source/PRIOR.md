# Prior: moving-lid SE(3) equivariance

## Decision and targeted failure

This policy targets displacement and rigid reorientation of the manipulated lid assembly beyond the demonstrations. It learns the commanded TCP pose in the **currently observed lid body frame**, rather than replaying a world-frame trajectory. The intended invariant covers free-space handle approach, grasp alignment, and hinge-constrained pulling. Release and retreat remain supervised in the same representation so the assigned skill and outgoing overlap are not truncated, although the moving reference is less desirable after release.

Observed evidence is limited: in `demo12200` and `demo12208`, the lid starts at the same world hinge pose with `lid_position=0`; the commanded TCP reaches the handle region by indices 180--200; the gripper command changes from open to closed at 220; lid motion is synchronized with closed fingers through 740--750; release is commanded at 760; and clearance continues through 949 while the peg stays fixed. The twelve demonstrations therefore give almost no installation-pose variation. Equivariance under unseen displacement is an expected benefit of the conversion, not a demonstrated success claim.

## Inputs and retained dependencies

`build_inputs` produces two causal frames of 38 dimensionless features. Each frame includes lid-frame TCP pose, lid-frame peg and hole positions, TCP-to-local-handle error, normalized lid angle and velocity, and scaled measured `qpos/qvel`. Relative geometry removes the nuisance lid transform, while joints and velocities retain robot configuration, reachability, dynamic phase and gripper-width dependencies. The initial episode observation may be duplicated by the framework; later policy switches retain two real causal frames.

The required argument `lid_pose_field` is bound to `lid_pose`. It selects the observed world pose used by both input construction and action conversion. The reference is resolved at every replan, and decoding deliberately resolves it again from each fresh current observation.

## Action labels and decoder

For each valid demonstration slot, `panda_kinematics.commanded_tcp_poses` converts the seven demonstrated arm targets to their world TCP pose. With the pre-action observed lid pose,

`T_lid_tcp = inverse(T_world_lid) * T_world_tcp_command`.

The unnormalized 10-D learned action is lid-frame translation in metres, the first two columns of the relative rotation matrix, and the original gripper scalar. The framework fits representation normalization only to these authorized valid labels and retains the supplied mask. Prediction slots are fixed at t-1 through t+14, and execution uses slots 1 through 8.

At execution, the 6-D rotation is projected to SO(3), then `T_world_lid_current * T_lid_tcp` forms a world TCP target. `panda_kinematics.solve_ik` is called from freshly measured joints. Its convergence flag, position and rotation residuals, iterations, joint-limit contacts and maximum joint change are reported; an unconverged best result is not hidden. The gripper scalar is unchanged and the common executor applies nonnegative=open, negative=close. Demonstration encode/decode is invertible in task space before nonunique IK; the declared 0.005 tolerance covers verified IK/FK residuals, not manipulation success.

## Model and losses

The mature `DiffusionBackbone` is unchanged. It receives the flattened 2x38 condition and diffuses 16x10 actions. The only training objective is masked epsilon diffusion loss; `prior_loss` is an explicit differentiable zero. There are no omitted learned modules, auxiliary labels, image inputs, future inference inputs, recurrent state, augmentation or task-completing decoder script. Training remains the fixed 20,000-update, batch-128, seed-0, DDPM/EMA recipe.

## Applicability, handoff and limitations

Prefer this package when lid-pose estimates are reliable and the assembly is displaced or reoriented while the transformed pose remains reachable. Exit only after observed open fingers, TCP clearance, a stably open lid and unchanged peg. Every `[740,950)` action is retained in all twelve bindings, yielding 2,520 overlap actions with `retrieve_peg_from_box`.

Relative coordinates do not create reachability, collision avoidance, contact truth or recovery data. The fresh moving-body reference may pull post-release retreat along with a settling panel. Large transforms can encounter joint limits, unseen obstacles or IK failure. Prefer the phase-aware world policy for release ambiguity near nominal placement, or the fresh-TCP incremental policy for unusual robot entry configurations. A closed gripper is only a proxy and does not prove grasp success.
