# Prior: stationary hinge-canonical trajectory with explicit gripper mode

## Decision and targeted failures

`open_hinged_lid__h04` targets two related weaknesses visible in the existing library without claiming their causes are proven. First, deployment may rigidly translate or reorient the manipulated lid assembly even though the demonstrations use one installation. Second, release and clearance are a sparse transition: in inspected `demo12200` and `demo12208`, the command is closed through index 750 and becomes open at 759, after which the lid settles while the TCP retreats. The fixed in-distribution validation was 7/12 for moving-lid-frame h01, 12/12 for fresh-TCP h02, and 10/12 for world/phase h03. H03's two failed runs had only one gripper change and a low open-command fraction, but the validation output does not identify the failed release/clearance clause.

The new policy reconstructs a **stationary closed-lid installation frame**, expresses the full Cartesian trajectory in that frame, and represents the gripper as separate learned open and close scores. The expected benefit is rigid-installation equivariance without intentionally attaching post-release retreat to the moving panel, plus more diffusion supervision for the sparse gripper mode. Neither benefit has out-of-distribution validation.

## Stationary root construction and displacement invariant

For every observed state, the adapter uses the caller-selected `lid_pose` and `lid_position` with the public local hinge axis:

`T_world_root = T_world_lid * Rot(lid_hinge_axis, -lid_position)`.

The demonstrations show a fixed hinge position and a positive local-y lid rotation consistent with this relation. If a common rigid transform `G` displaces or reorients the entire lid installation, then `T_world_root` becomes `G*T_world_root` and a world TCP command transformed by the same `G` has the same root-relative label. Thus the learned approach, grasp-alignment, pull, release and retreat arm targets are invariant to that common transform by construction. This does not make transformed targets reachable or collision-free.

The root is recomputed in each causal input frame and from the fresh current observation for every executed slot. It is not integrated, cached as physical state, or advanced during denoising. `chunk_context` records only finite replan diagnostics; it is not the decode reference.

## Inputs and high-level arguments

`build_inputs` emits two causal frames of 40 dimensionless values each:

- root-relative TCP position (3) and 6D rotation (6);
- root-frame TCP-to-current-handle error (3), with the handle obtained from the public lid-local handle point and current lid pose;
- root-relative peg and hole positions (3+3);
- normalized lid angle, scaled lid velocity, normalized angle-to-release, and normalized caller release angle (4);
- measured Panda qpos and qvel, including both finger joints, scaled into 18 values.

Root-relative geometry removes the nuisance installation transform. Measured joints and velocities retain robot configuration, gripper state, dynamic phase and reachability dependencies that a Cartesian invariant alone cannot remove.

All arguments are required and used identically in training and deployment. `lid_pose_field='lid_pose'` and `lid_angle_field='lid_position'` select the observed quantities used for root construction. `release_angle_rad=1.73` is grounded in the demonstrated release region and enters direct phase and release-margin features. It is not a decoder threshold, actuator command or success rule. The framework supplies two real causal frames across policy switches and duplicates only the initial episode observation when needed.

## Cartesian labels, mode labels and decoder

For each valid demonstration slot, `panda_kinematics.commanded_tcp_poses` converts the seven demonstrated joint commands to the commanded world TCP target. With the slot's pre-action observation, the adapter encodes

`T_root_tcp = inverse(T_world_root) * T_world_tcp_command`.

The 11-dimensional unnormalized learned action is root-frame translation in metres (3), root-frame rotation as the first two rotation-matrix columns (6), and `[open_score, close_score]` (2). A native nonnegative gripper label becomes `[1,0]`; a negative label becomes `[0,1]`. The framework retains the supplied validity mask and fits representation midpoint/half-range normalization only on these authorized valid labels. There is no native-joint normalization reused for this representation.

At execution, the sampled 6D rotation is deterministically projected to SO(3), the fresh root transforms the relative target to world, and `panda_kinematics.solve_ik` obtains seven native joint targets from freshly measured joints. The decoder reports convergence, position and rotation residuals, iterations, joint-limit contacts, maximum joint change and posture diagnostics; it does not hide an unconverged best result. `open_score >= close_score` decodes to native `+1`, otherwise to `-1`, after which the common executor applies its unchanged sign rule. This score comparison is learned output conversion, not a phase script. Exact one-hot demonstration labels recover the demonstrated gripper sign. Matching-observation encode/decode reconstructs the demonstrated Cartesian target before nonunique IK, and the independent conversion check uses a 0.005 maximum task-space/gripper tolerance.

## Model and objective

The mature standard `DiffusionBackbone` is unchanged. It receives the flattened 2x40 causal condition and diffuses 16x11 actions. The masked epsilon objective assigns weight 1 to each of the nine Cartesian dimensions and weight 2 to each of the two mode dimensions, normalized by the sum of dimension weights. This keeps action diffusion fully active while modestly emphasizing close/release labels. `prior_loss` is a differentiable zero because there is no auxiliary head. All trainable parameters belong to the registered backbone and are included in optimization, checkpointing and EMA.

There are no future inference inputs, images, contact labels, action-history inputs, augmentation, recurrent state, auxiliary labels or decoder-side task-completion script. Training retains seed 0, 20,000 updates, batch 128, DDPM 100/100, EMA 0.999, and the supplied optimizer/schedule. Prediction slots are t-1 through t+14; execution uses slots 1 through 8, current t through t+7, at 20 Hz.

## Supervision, overlap and handoff

All twelve bindings cover `[0,950)` with `context_start=0`. This preserves the complete skill and every `[740,950)` outgoing overlap action: 210 actions per demonstration and 2,520 total. The overlap includes final pull, the open transition, released-lid settling, lift/retreat and transit toward box access.

Prefer h04 when lid pose and angle are coherent and assembly displacement or a stationary post-release reference is the main concern. Prefer h02 for a strongly off-path TCP or unusual arm entry because h04 predicts an absolute canonical path. Near nominal placement, h03 remains an independently trained world/phase alternative; h01 remains an independently trained moving-body alternative during contact.

Entry requires a closed or partly open lid with valid coherent pose/angle, a stationary in-box peg, and a free or plausibly handle-aligned TCP. Exit and successor readiness rely only on physical observations: open fingers, stable open lid, low lid velocity, TCP clear of the handle/panel, and unchanged peg. Preserve both causal frames at handoff.

## Limitations and falsifiable failure signature

The training data has essentially one lid installation and one initial arm configuration. It contains no deliberate displacement, missed-grasp recovery, lost-contact recovery or collision recovery. Root construction assumes lid pose orientation, scalar angle and public local axis describe the same one-DoF joint; disagreement can make the reconstructed root and world target drift as the lid moves. Large rigid transforms may violate reachability, joint limits or collision clearance. The weighted two-score mode can still release early, remain closed too long, or trade Cartesian fit against mode fit under the fixed update budget. No contact truth is available.

Observable failure signatures are growing TCP-to-handle error, root/world target drift with lid angle, closure without synchronized lid motion, stalled or reversing lid angle, premature opening, fingers remaining closed after maximum opening, repeated IK residuals or limit contacts, failure to clear the panel, or peg motion. These cues authorize stopping or switching; they do not imply demonstrated recovery.
