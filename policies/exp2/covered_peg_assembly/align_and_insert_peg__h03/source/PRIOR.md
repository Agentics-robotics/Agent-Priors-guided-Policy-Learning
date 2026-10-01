# Causal phase-aware world-target prior

## Decision and target failure

This policy targets phase ambiguity caused by variable switch timing. A two-frame handoff may be high during alignment/descent, on the insertion line during +x advancement, or already in the stable terminal tail. The model predicts absolute world-frame Cartesian TCP targets so that it retains the demonstrated robot/workspace pattern; it is intentionally complementary to the displaced-fixture and local-residual policies.

## Evidence and labels

In demo12200, peg z falls from about 0.424 m at 1380 to about 0.103 m at 1460. The peg center then advances from approximately x=-0.312 m to x=-0.182 m, with repeated terminal targets after 1520. Demo12205 includes additional initial y correction before the same phase sequence. Demo12208 reaches near completion by 1500 and retains a stable tail. All full assigned segments and every [1380,1500) overlap remain action-supervised.

The auxiliary target is derived from the current causal history frame, never from an inference future or contact truth. Poses are expressed in the observed hole frame. Class 0 (`approach_descend`) applies while peg-center z is more than 0.03 m above the observed target-center z. At insertion height, class 1 (`insert`) applies while peg-center x is more than 0.008 m behind the observed target-center x. Otherwise class 2 (`stabilize`) applies. The target has its own always-valid window-level mask because the two causal history frames are valid for every bound sample, including framework duplication at episode start. These thresholds describe nominal geometric progress, not contact or task success.

## Inputs, learned modules, and gradient paths

Each of two causal frames has 86 channels: world TCP, peg, hole, and target poses; hole-relative TCP and peg poses; TCP-relative peg pose; measured qpos/qvel; and explicit peg-head-in-hole coordinates, axis alignment, and finger width. Poses use translation plus 6D rotation. The fixed numerical scaling is documented in `adapters.py`; no evaluation statistics are fitted.

A registered two-layer 256D MLP encodes the flattened 172D history. A registered linear stage head predicts three logits. The diffusion condition concatenates the latent with the stage softmax, so the encoder and head receive gradients from the main diffusion loss. Masked cross-entropy with weight 0.05 adds a real auxiliary gradient path into both modules. `compute_loss` reports diffusion and weighted prior losses separately. All modules are part of the model, optimizer, EMA, and saved state. The mature public Diffusion Policy backbone itself is unchanged; only its condition is produced by the declared encoder and stage head.

## Actions and conversion

The 10D learned action is `[world xyz metres, world rotation 6D, gripper scalar]`. `encode_targets` uses verified FK of demonstrated native commands. `decode_action` projects sampled 6D vectors to SO(3), builds the absolute world TCP target, and invokes `panda_kinematics.solve_ik` from fresh measured qpos. The gripper scalar is preserved. Solver convergence, position/rotation residuals, iterations, limit contacts, and maximum joint change are reported without hiding unconverged output.

## Contract and handoff

Call with `{"destination_object":"hole","held_object":"peg","phase_strategy":"geometry_inferred"}` when the fixture is near the demonstrated workspace, the peg is visibly retained, and switch timing is uncertain. Inputs remain causal across policy switches. The model samples slots t-1 through t+14 and the framework executes slots 1 through 8 at 20 Hz. Termination remains solely the supplied head-x, y/z, axis-alignment, and stability definition; a predicted stabilize class is not success.

## Assumptions and limitations

The auxiliary geometry assumes a nominal approach ordering and accurate object poses. It does not detect contact, collision, a dropped peg, or a jam, and no such recovery is demonstrated. Absolute world actions are not invariant to hole displacement and may memorize the fixed fixture pose; use the hole-frame policy for that change. For modest tracking/handoff perturbations with secure grasp, use the fresh-TCP residual policy. The expected benefit of the stage objective is better phase preservation, not proof of unseen recovery.

Provenance: frozen prior plan hash `0c7abe9c128d01c843b7de380175616f9a9369a076d939368b768d3eac8107df`; cut plan hash `2ae5a2895d6b7907c7d8b047a04c9ef42fbea4ce995788b0fd08dd7ccdcaea88`; dataset SHA-256 `9249213e5d1915185d1f90acd07fb9cb6482642ea69297a0fb798eadd92a13ec`.
