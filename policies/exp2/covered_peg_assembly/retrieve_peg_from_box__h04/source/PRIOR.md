# Object-selective chunk-reference Cartesian prior

## Decision and targeted failure

This independent Diffusion Policy targets a frame conflict exposed by the long retrieve responsibility. The hand initially acts on the lid, then changes active object to the peg. A peg displacement should move the post-release approach, grasp and lift, but should not move an independently located lid-release action.

The evidence is visible in the authorized trajectories. `demo12200` and the more offset `demo12208` have peg x positions near -0.604 m and -0.616 m at entry. Their commanded TCP x positions during lid release are nevertheless nearly identical at indices 740 and 760. At index 990, after release, their commanded TCP x positions are about -0.6039 m and -0.6158 m and follow the corresponding peg shift. This supports an object-selective translation reference; it is not out-of-distribution evidence.

The 10D learned action is

`[p_reference_to_tcp in world axes (m, 3), R_world_tcp first two columns (6), native gripper scalar (1)]`.

The reference is resolved from the newest causal observation once per sampled chunk and then fixed for all 16 predicted slots and 8 executed slots. It is the observed lid-handle point only when the peg remains no more than 0.04 m above the public wall height, TCP-to-handle distance is at most 0.10 m, and measured finger width is below 0.07 m. Otherwise it is the observed peg center. The early reference therefore follows lid translation, while the post-release reference makes approach, grasp and lift equivariant by construction to peg translation at every replan. World orientation is retained intentionally because the evidence does not support arbitrary peg-rotation transfer. Fixing the point within a chunk avoids changing the coordinate origin while executing one sampled short trajectory.

## Causal inputs and retained dependencies

Each of two observation frames has 73 deterministic features. The first 62 contain arm qpos/qvel, finger positions and velocities, absolute TCP/peg/lid poses relative to the public robot base translation, TCP pose relative to the peg, TCP-to-lid-handle vector and distance, finger width, peg height relative to the box wall, and lid angle/velocity. Ten additional causal motion channels describe peg velocity, TCP velocity, change in TCP-minus-peg displacement, and finger-width velocity. The final channel indicates whether that frame's selected reference is the peg. The oldest frame's motion channels are zero; the newest uses only the two supplied observations at 20 Hz. Initial duplicated history consequently yields zero recent motion.

Absolute poses and joints preserve world reachability, redundant-arm configuration and box/lid dependencies. Relative geometry exposes active-object phase. No image, future observation, action history, contact truth or recurrent phase state is used at inference. Fixed physical scaling is deterministic. The framework fits represented-action normalization only from valid labels in these bindings, and framework masks exclude padded targets.

## Labels, decoding and feasibility

For each training window, `build_inputs` stores the selected world reference point in `chunk_context`. `encode_targets` uses FK of every demonstrated seven-joint command, subtracts that single point from the commanded TCP position, retains the commanded world rotation as continuous 6D columns, and appends the demonstrated native gripper scalar. The same conversion covers prediction slots t-1 through t+14.

At execution, `decode_action` adds the same chunk-fixed point to the represented position. It projects sampled 6D rotation with deterministic safe Gram-Schmidt and constructs a world TCP pose. Only `panda_kinematics.solve_ik` converts that pose to seven native joint targets, starting from freshly measured qpos; the gripper scalar is appended unchanged. Diagnostics expose convergence, position and rotation residuals, iterations, lower/upper limit contacts, joint change, posture error, world target and selected reference. Reference addition/subtraction and valid 6D rotations round-trip in task space. IK consistency does not establish collision freedom, contact or manipulation success.

## Model and gradient path

The model is the unchanged standard `DiffusionBackbone`. It flattens `[2,73]` causal features into the global condition and predicts epsilon for `[16,10]` actions. The only objective is framework-masked epsilon diffusion loss. `prior_loss` is a differentiable zero because the prior is implemented by causal input structure and Cartesian conversion; no artificial auxiliary head is used. Every trainable parameter is registered in the backbone and is therefore included in optimization, EMA and checkpoint state.

## Relation to original validation

The available in-distribution validation reports h01 at 0/12, h02 at 4/12, and h03 at 8/12. h03 elevated the peg in all twelve runs with no IK-unconverged steps, while four final failures had the cover closed. h04 retains h03-style causal phase features but does not simply repeat h01's full fresh peg-pose SE(3) representation: it uses only a selected world-axis translation point, keeps orientation in world coordinates, applies the lid reference before release, and fixes the reference inside each chunk. The expected benefit is a better tradeoff between phase discrimination and translated-peg transfer. That benefit remains unconfirmed until formal validation.

## Coverage, calling and handoff

Call arguments are `{}`. Every action in each `[740,1270)` occurrence is supervised. This preserves 210 incoming overlap actions `[740,950)` with `open_hinged_lid` and 150 outgoing overlap actions `[1120,1270)` with `reorient_and_stage_peg`.

Prefer invocation when the observed reference is unambiguous and the translated path remains reachable. Continue through release, open clearance, approach, closure and shared lift. Handoff only when fingers are closed, peg height is above the wall with margin, and the two frames show peg/TCP co-motion with a stable relative pose. Finger closure alone is not contact truth. The unchanged exit scoring also requires the cover to remain above 0.9 rad.

## Limitations and expected failure signature

The hand-written selector is supported only by successful demonstrated geometry. Unseen intermediate states can select the wrong object and cause a target jump, lid contact, or descent offset from the peg. Translation equivariance does not remove arm reachability, box-wall clearance, lid interference, grasp/contact mechanics or perception error. It does not cover arbitrary peg rotation. The policy does not actively support or reopen the cover after release, so the cover-closure failure observed for h03 can remain. Demonstrations provide no supervision for misses, drops, regrasp, blocked lifts or recovery; if peg z remains low, peg/TCP co-motion breaks, IK is infeasible, or the cover closes, the high-level agent should stop or select an appropriate separately trained skill rather than assume recovery.
