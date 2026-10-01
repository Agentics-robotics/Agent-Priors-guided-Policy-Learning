# Phase-supervised relational conditioning with world targets

## Decision and target failure

This policy targets phase ambiguity when the high-level agent enters the long open-drawer segment at an intermediate nominal state. The continuous segment changes responsibility from handle approach to pull, front release/retreat, red approach and red close/lift. World Cartesian actions deliberately avoid moving every target with a potentially noisy object frame; this differs from the displacement prior and is intended for the nominal cabinet calibration.

## Evidence and labels

Observed successful occurrences have consistent transitions: indices 0:145 approach the front; 145:230 close and pull; 230:310 open/release/retreat; 310:445 approach and descend to red; 445:470 close and initially lift. Demo1000 observations at 140/160/230/240/300/400/445/470 show these changes through TCP geometry, drawer position/velocity, finger state and red/TCP relation. Demo1001 and demo1005 vary red Y and the post-320 arm path while retaining the same responsibilities. These occurrence-index classes are supervised descriptions of nominal demonstrations, not proof that unseen failures can be classified or recovered.

## Inputs, model and gradient path

Each of two causal frames contains normalized qpos/qvel, world TCP position/6D rotation, red pose relative to TCP, drawer progress/velocity, and drawer-center position relative to TCP. A registered per-frame MLP maps 41 values to 128 features. Both frames are flattened and a registered condition MLP produces a 256-dimensional condition for the unchanged standard diffusion U-Net. A registered linear phase head predicts five classes from the same condition.

`encode_targets` supplies one phase class for the causal current source index and a mask copied from the current execution-aligned slot's framework validity mask. `compute_loss` applies per-sample cross-entropy, multiplies by the phase mask, normalizes by valid count, and adds it with weight 0.05 to masked diffusion epsilon loss. Therefore the auxiliary objective has a real gradient through the condition encoder used by diffusion. The phase label is never an inference input, the head never replaces action diffusion, and repeated denoising calls do not advance state. Encoder/head/backbone are registered, optimized, checkpointed and EMA-tracked together.

## Cartesian conversion

Action labels are demonstrated commanded TCP poses from public URDF FK, encoded as world position in metres, world 6D rotation and native gripper scalar. `decode_action` performs robust Gram-Schmidt projection, constructs the world target, calls only `panda_kinematics.solve_ik` from fresh qpos and reports convergence/residual/limit diagnostics. Gripper sign semantics are unchanged. The framework fits normalization only to valid world-representation labels. No augmentation is used.

## Coverage, timing and handoff

All 12 [0,470) segments remain supervised. The phase classes partition labels but do not remove actions, and all 230:470 actions remain overlap with `red_transfer`. History 2, horizon 16, prediction offset -1, execution chunk 8, 20 Hz, DDPM100, EMA and 20,000 updates remain fixed. `red_transfer` may receive after opening/red exposure near 230; the stronger nominal transfer observes closed fingers plus red/TCP upward co-motion near 470.

## Applicability and limitations

Prefer nominal cabinet calibration with physically coherent phase cues but uncertain switch timing. Use drawer-frame tracking for known drawer displacement and fresh-TCP increments for modest off-path arm state. Absolute world targets can fail when the cabinet moves. Phase classes come only from successful timing and may be wrong for a missed front, blocked pull, misgrasp or drop. The auxiliary head has no contact truth, does not directly gate execution and cannot establish grasp. IK feasibility and collision freedom remain physical constraints.
