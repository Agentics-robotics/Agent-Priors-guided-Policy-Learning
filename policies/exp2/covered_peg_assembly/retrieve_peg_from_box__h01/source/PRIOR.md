# Peg-frame Cartesian target prior

## Decision and targeted failure

This independent policy targets displacement of the peg beyond the demonstrations. The twelve start boundaries place the peg in only about a 12 mm x-range and 11 mm y-range. In the inspected offset occurrence (`demo12208`), commanded approach x follows the shifted peg relative to `demo12200`; this supports a peg-relative representation but is not deployment evidence.

The learned action is

`[p_peg_tcp (m, 3), R_peg_tcp first two columns (6), gripper scalar (1)]`.

For every training slot, `encode_targets` computes `inverse(T_world_peg_before_action) @ T_world_commanded_tcp`, where the TCP target is FK of the demonstrated native arm command. For every executed slot, `decode_action` composes the represented target with the **freshly observed** world peg pose. Consequently, approach, grasp alignment, and carried lift transform with peg translation and rotation by construction. Early lid release is covered by supervision but is not the invariant part of this prior.

## Inputs and retained dependencies

`build_inputs` returns two causal frames of 56 features. They include arm qpos/qvel, finger positions/velocities, TCP-in-peg pose, lid-in-peg pose, absolute peg and TCP poses relative to the robot base translation, and lid angle/velocity. Relative channels emphasize transferable geometry; absolute channels and joints retain reachability and configuration dependence. There are no images, future states, contact labels, or hidden phase memory. Initial duplication of the first observation is framework-owned.

Hand-scaled physical features use fixed public units and constants; no evaluation statistics are fitted. Learned action normalization is fitted by the framework only from valid represented training labels. Invalid horizon padding remains masked by the framework.

## Model and gradients

The registered model is the unchanged standard `DiffusionBackbone` with the two frames flattened as global condition. The only loss is masked epsilon diffusion loss. `prior_loss` is a differentiable zero because the geometric prior is fully implemented in inputs and action conversion; no auxiliary head is needed. All trainable parameters are in the registered backbone and therefore optimizer, EMA, and checkpoint state.

## Conversion and feasibility

Rotation labels use the continuous 6D representation. Decode applies Gram-Schmidt; a deterministic fallback only regularizes degenerate sampled vectors. The resulting world target is passed only to `panda_kinematics.solve_ik` with freshly measured joints. Diagnostics expose convergence, position and rotation residuals, iterations, joint-limit contacts, maximum joint change, and target TCP pose. The native gripper scalar is preserved. This projection and IK do not establish manipulation success.

## Evidence and expected benefit

At index 990, `demo12200` has peg x about -0.6038 m and commanded TCP x about -0.6039 m, while `demo12208` has peg x about -0.6157 m and commanded TCP x about -0.6158 m. Similar relative descent/closure/lift structure appears at indices 1020-1240. The expected, falsifiable benefit is better translation/rotation transfer for a displaced reachable peg than the world-absolute prior. It is not an observed test result.

## Calling, overlap, and handoff

Call arguments are `{}`. Every `[740,1270)` segment is bound, including 210 incoming overlap actions `[740,950)` with `open_hinged_lid` and 150 outgoing overlap actions `[1120,1270)` with `reorient_and_stage_peg`. Prefer invocation after observed lid release/clearance. Handoff only when closed fingers, peg height above the wall, and two-frame peg/TCP co-motion jointly support carrying.

## Limitations

Peg framing does not cancel arm reachability, box-wall clearance, lid interference, perception error, grasp mechanics, or contact dependence. A large displacement can be infeasible. Early lid release depends on lid geometry and is better covered by the phase-observable world policy. Demonstrations contain successful pickups only and provide no evidence for miss, drop, regrasp, or blocked-lift recovery.
