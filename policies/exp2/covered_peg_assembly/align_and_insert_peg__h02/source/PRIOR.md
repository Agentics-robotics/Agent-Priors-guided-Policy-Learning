# Fresh-TCP residual servo prior

## Decision and target failure

This policy targets modest intermediate-state and tracking differences at handoff. The predecessor may switch anywhere in the 120-action overlap, and measured TCP poses lag their demonstrated commanded targets throughout the descend-and-insert motion. Each learned target is therefore represented as a local transform from the TCP observed immediately before that demonstrated action and is re-anchored to the fresh measured TCP for each executed slot.

## Evidence

Demo12200 indices 1380-1470 show alignment and descent with visible measured-versus-commanded tracking differences; 1480-1520 show forward insertion; 1530-1540 show stabilization. Demo12205 supplies the largest observed incoming lateral correction. The evidence is nominal successful behavior, not injected-error recovery. Every assigned action remains supervised, including [1380,1500) in all twelve trajectories.

## Mechanism and inputs

`build_inputs` uses two causal frames. Per frame it encodes TCP-relative hole, peg, and target poses; hole-relative peg pose; absolute world TCP pose; and measured qpos/qvel. Pose rotations use 6D columns only in the observation representation. The destination and held-object arguments select `hole_pose` and `peg_pose`. Absolute TCP and joint channels retain world configuration and reachability dependencies that a purely local representation would discard.

The 7D learned action is `[delta xyz metres in current TCP coordinates, relative rotation vector radians, gripper scalar]`. `encode_targets` computes `measured_tcp_T_commanded_tcp` separately for every slot using that slot's action observation and verified FK of its native joint command. At physical execution, `decode_action` exponentiates the rotation vector, left-composes the transform with the freshly observed world TCP, and invokes `panda_kinematics.solve_ik` from fresh measured qpos. It does not integrate predictions, update history during denoising, or use stale future states. Solver convergence, residuals, iterations, limits, and maximum joint change are exposed along with residual magnitudes.

This representation creates a feedback-like re-anchoring: if the actual TCP differs modestly from the nominal intermediate state, the same predicted local corrective relation starts from the actual state. That is an expected structural benefit; successful nominal demonstrations do not prove unseen disturbance recovery.

## Model and training

The standard public Diffusion Policy backbone is unchanged. It receives a flattened 126D causal condition and predicts epsilon for 16 slots of 7D residual actions. There is no auxiliary loss; the prior is entirely in the input and action coordinate choices. Framework-fitted representation normalization uses valid transformed training labels only. History 2, horizon 16, execution chunk 8, 20 Hz, DDPM, EMA, and the declared 20,000-update recipe remain fixed.

## Contract and handoff

Call with `{"destination_object":"hole","held_object":"peg"}` when the grasp is visibly secure and a local correction is plausibly collision-free and reachable. The high-level agent should continue to monitor hole-relative peg geometry because a local residual is not a completion signal. Exit only through the unchanged task success evaluator and retain terminal closed-gripper actions.

## Assumptions and limitations

Fresh TCP pose and measured qpos are assumed accurate. Re-anchoring can accumulate drift, apply an unsafe local direction under a large offset, or lose the exact destination anchor. It provides no contact truth and no support for dropped pegs, severe angular errors, collisions, jams, or retries. Prefer the hole-frame policy for moved fixtures and the phase-aware world-target policy for ambiguous timing near the demonstrated workspace.

Provenance: frozen prior plan hash `0c7abe9c128d01c843b7de380175616f9a9369a076d939368b768d3eac8107df`; cut plan hash `2ae5a2895d6b7907c7d8b047a04c9ef42fbea4ce995788b0fd08dd7ccdcaea88`; dataset SHA-256 `9249213e5d1915185d1f90acd07fb9cb6482642ea69297a0fb798eadd92a13ec`.
