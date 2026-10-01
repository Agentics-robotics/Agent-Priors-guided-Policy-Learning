# Fixed closed-destination and closure-response prior

## Decision and targeted failure

This policy targets active-closure phase ambiguity: invocation with the drawer already partly closed, failure to continue toward the supplied 0.025 m threshold, or premature transition to release/retreat. During successful demonstrations, `target_pose.x + drawer_position` is nearly fixed around 0.125 m: for example about -0.175 + 0.300 while open and 0.123 + 0.002 while closed. The policy therefore constructs

`T_closed = T_target_pose @ Translation(+drawer_position, 0, 0)`

and anchors target translation to that fixed closed destination. This provides a common spatial endpoint as the moving drawer frame advances.

Observed closure passes through substantially different intermediate positions (for example around 0.29, 0.19, 0.08, 0.026 and 0.002 m across inspected runs), supporting progress conditioning. It does not prove recovery from a stall or jam.

## Inputs and action coordinates

Two causal input frames each contain 42 values: TCP and object poses relative to `T_closed`, TCP-object displacement, drawer position/velocity, finger width, and scaled Panda qpos/qvel. q state retains fixed-base reachability and configuration dependence.

The learned 10D action uses destination-frame xyz in metres, **world-frame** orientation in continuous rotation 6D, and native gripper scalar. Thus this prior differs from both a full moving-drawer relative pose and a current-TCP rotation-vector correction. For each slot, the commanded TCP comes from exact FK of the native command. Translation is encoded with `inverse(T_closed) @ T_command`; orientation is taken from the world command rotation.

At decode, the destination is recomputed from fresh `target_pose` and `drawer_position`. Relative translation is mapped to world coordinates, world rotation 6D is Gram-Schmidt projected to SO(3), and the final world TCP target is passed to `panda_kinematics.solve_ik` from fresh qpos. The unchanged gripper scalar is appended. IK convergence, residuals, iterations, limit contacts, joint change, destination and target poses, and degenerate-rotation fallback are reported. Unconverged IK is not hidden.

## Shared learned auxiliary and gradient path

A registered MLP maps the flattened 84-value causal input to a 256D condition. That condition is used by the unchanged supplied `DiffusionBackbone` and by a registered scalar closure-response head. Both encoder and head are in the model, optimizer, checkpoint and EMA.

For training only, `encode_targets` labels the drawer-position change from the pre-action state for execution slot 1 to the post-action state for slot 8. The label is in metres. Its mask is the product of the framework masks for slots 1 and 8, so padded futures contribute no auxiliary gradient. `compute_loss` scales the metre target by 0.1 m, applies masked MSE, and adds weight 0.05 as `prior_loss` to the primary masked epsilon diffusion loss. The gradient path is closure head -> shared condition encoder -> the representation used by diffusion. The auxiliary cannot replace action diffusion.

Inference receives only causal observations. Future state is never a model input, the head is not required for action decoding, and its output is not contact or success. Repeated denoising calls do not advance physical or recurrent state.

## Temporal alignment, normalization and supervision

Prediction slots are t-1 through t+14, and execution uses slots 1 through 8 beginning at current t. Cartesian targets use the corresponding action observation in training and fresh current observation in decoding. The framework fits representation normalization only from valid destination/world action labels.

Every action `[900,1330)` in all 12 demonstrations remains diffusion-supervised. In particular, `[900,1160)` preserves 260 actions (13 s) of overlap with `place_block_in_drawer`, covering release, clearance, handle approach and acquisition. The auxiliary mask is independent and never trims these action labels.

Enter with the block over/inside the cavity at early overlap or at a later open/partly closed state. Exit guidance stays drawer position below 0.025 m, negligible velocity, block retained near z 0.063 m, open fingers and high clear TCP. Framework success is unchanged.

## Applicability and limitations

Prefer this policy when closure progress or stopping is the dominant uncertainty and `target_pose` plus `drawer_position` form a consistent reachable prismatic destination. The construction assumes motion along target local x. World orientation labels do not make arbitrary fixture rotation invariant. The auxiliary only models response in successful demonstrations; it cannot observe contact truth, detect a jam, or prove future motion. Changed handle offset/axis, missed grasp, obstruction, block interference, collision and unreachable placement remain unsupported. The TCP-local prior better covers local hand perturbation, and the moving-drawer prior better covers coherent fixture displacement.
