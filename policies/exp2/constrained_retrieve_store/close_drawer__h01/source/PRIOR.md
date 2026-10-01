# Moving-drawer-frame SE(3) prior

## Decision and targeted failure

This policy targets displacement of the drawer assembly, hence displacement of the placed block and approached handle, outside the narrow demonstrated scene range. It represents each commanded TCP target in the **current observed `target_pose` frame**. The purpose is to make release completion, clearance, handle approach, closure, release and retreat covariant to translation/rotation of that observed drawer-attached frame by construction.

Observed support is relational, not proof of deployment success. In `demo12100`, `target_pose.x` moves from about -0.175 m at drawer position 0.300 m to about 0.123 m at 0.0022 m, while the low handle-contact TCP moves from about -0.448 m to -0.142 m. Demonstrations otherwise have little lateral fixture variation, so displacement generalization is an expected benefit of the representation rather than an observed result.

## Inputs and invariant

`build_inputs` produces two causal frames of 41 values each. Each frame contains TCP and object poses relative to `target_pose` (relative xyz and rotation 6D), TCP-object displacement, drawer position/velocity, and scaled Panda qpos/qvel. Relative scene channels remove a rigid drawer-frame displacement. Joint position and velocity are deliberately retained because the robot base does not move with the drawer: reachability, posture, velocity, joint limits and contact feasibility are not invariant.

There are no high-level call arguments. At episode start only, the framework may duplicate the first observation. Across skill switches, the two real causal frames are retained by the framework.

## Action labels and decoding

For every native demonstration command, `encode_targets` obtains the exact commanded world TCP pose using `panda_kinematics.commanded_tcp_poses`. For each prediction slot it computes

`T_relative = inverse(T_target at that slot's action observation) @ T_command`.

The 10 learned values are relative xyz in metres, relative rotation as the first two rotation-matrix columns (6D), and the original gripper scalar. The representation normalizer is fitted by the framework only from valid labels in these coordinates.

At every executed slot, `decode_action` resolves `target_pose` from the fresh observation, projects rotation 6D to SO(3) by Gram-Schmidt, composes a world TCP target, and calls `panda_kinematics.solve_ik` from freshly measured qpos. It appends the unchanged learned gripper scalar. Degenerate 6D fallback, IK convergence, position/rotation residuals, iterations, joint-limit contacts and maximum joint change are reported. An unconverged result is not replaced or hidden.

Prediction slots are t-1 through t+14; slots 1 through 8 execute from current t. There is no action accumulation or hidden phase state.

## Model and gradients

The model uses the supplied `DiffusionBackbone` unchanged with the flattened 82-value causal condition and 10D action sequence. The only objective is masked epsilon diffusion loss. `prior_loss` is a differentiable zero because the scientific prior is fully expressed by causal coordinates and the action converter; no artificial auxiliary task is added. All trainable parameters belong to the backbone and are included in optimizer, EMA and checkpoint state by the framework.

## Supervision and handoff

All 12 bindings supervise every action in `[900,1330)`. In particular, `[900,1160)` is 260 actions (13 s) of real overlap with `place_block_in_drawer`, covering variable release completion, retreat/reorientation, handle approach and acquisition. Evidence inspected at index 900 includes object heights from about 0.063 to 0.124 m, followed by handle approach around 1080:1150, active closure around 1160:1240, opening/release around 1260, and high retreat by 1330.

Enter only with the block over/inside the open drawer or at a later valid partly closed state. Exit guidance remains drawer position below 0.025 m, near-zero drawer velocity, block inside near z 0.063 m, open fingers and a high clear TCP. This is guidance, not a modified success condition.

## Applicability and limitations

Prefer this policy when the observed drawer frame is reliable and coherent drawer/handle/block displacement is the dominant uncertainty. The invariant assumes unchanged rigid handle geometry relative to `target_pose`. It supplies no contact truth and no recovery labels for a missed grasp, jam, block outside the cavity, changed handle, obstruction, collision or unreachable pose. Severe rotation/translation remains constrained by the fixed robot base and IK. Prefer the TCP-local prior for local intermediate-state mismatch and the fixed closed-destination prior for partial-closure/stopping ambiguity.
