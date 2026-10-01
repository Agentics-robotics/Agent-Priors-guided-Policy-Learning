# Hole-frame SE(3) trajectory prior

## Decision and target failure

This policy targets displacement of the square-hole fixture beyond the single world pose represented in the twelve demonstrations. It covers the complete assigned responsibility: final alignment above the hole, controlled descent, +hole-x insertion, and terminal stabilization. The expected benefit is equivariance of the learned motion to the observed hole pose; it is not a claim that every moved hole is reachable or collision-free.

## Evidence

At demo12200 indices 1380-1460 the grasped peg is aligned and lowered from about 0.424 m to the insertion line. Indices 1480-1520 advance along +x, and 1540-1554 hold the successful pose. Demo12205 starts with the largest observed lateral offset (peg y about 0.160 m) and reaches the same nominal hole relation. Every binding retains [1380,1500), the 120-action overlap with `reorient_and_stage_peg`, plus insertion and hold.

## Mechanism and inputs

`build_inputs` resolves `destination_object=hole` and `held_object=peg` from each of two causal observations. Each 63-channel frame contains hole-relative TCP, peg, and goal poses; TCP-relative peg pose; the absolute world hole pose; and measured qpos/qvel. Poses use translation plus continuous 6D rotation. Relative translations are scaled by 0.5 m, hole world translation by 1 m, qpos by 3 rad, and qvel by 2.5 rad/s. Relative channels provide the desired geometric invariant, while absolute hole pose and robot state retain workspace, configuration, velocity, and reachability dependence. Initial episode history is framework-duplicated; history remains causal across skill switches.

The learned action is `[hole-frame xyz metres, hole-frame rotation 6D, gripper scalar]`. `encode_targets` obtains the demonstrated commanded TCP pose with the verified FK helper and computes `hole_T_commanded_tcp` using that slot's action observation. `decode_action` projects sampled 6D values to SO(3), composes them with the freshly observed `world_T_hole`, and calls `panda_kinematics.solve_ik` from freshly measured qpos. The native gripper scalar is preserved. Solver convergence, position/rotation residuals, iterations, joint-limit contacts, and maximum joint change are reported; an unconverged result is not hidden.

## Model and training

The unchanged public Diffusion Policy backbone receives the flattened two-frame 126-vector condition and predicts epsilon for 16 slots of 10D actions. There are no auxiliary modules or losses; `prior_loss` is differentiable zero and action diffusion remains the full objective. Representation normalization is fitted by the framework only on valid transformed labels. The fixed H2/16/8, 20 Hz, DDPM, optimizer, EMA, and 20,000-update recipe is unchanged.

## Contract and handoff

Call with `{"destination_object":"hole","held_object":"peg"}` only when the peg is securely held, clear of the box, near axis alignment, and the transported path appears reachable. All assigned ranges in `training_bindings.json` are supervised. Exit only under the supplied head-x, y/z, axis-alignment, and stability definition. There is no learned successor; retain terminal closed-gripper actions.

## Assumptions and limitations

The hole observation is assumed accurate and semantically consistent. SE(3) transport preserves labels geometrically but cannot preserve joint feasibility, clearance, contacts, or dynamics after a large fixture move. The data contain one hole world pose and nominal successful insertions only. There is no contact truth and no evidence for a dropped peg, severe orientation error, collision recovery, jamming, or retry behavior. The TCP-residual policy is the complementary choice for small incoming execution perturbations; the phase-aware world policy is the complementary choice for ambiguous timing near the demonstrated workspace.

Provenance: frozen prior plan hash `0c7abe9c128d01c843b7de380175616f9a9369a076d939368b768d3eac8107df`; cut plan hash `2ae5a2895d6b7907c7d8b047a04c9ef42fbea4ce995788b0fd08dd7ccdcaea88`; dataset SHA-256 `9249213e5d1915185d1f90acd07fb9cb6482642ea69297a0fb798eadd92a13ec`.
