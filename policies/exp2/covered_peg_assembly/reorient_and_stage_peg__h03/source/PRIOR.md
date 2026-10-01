# Receding local-twist prior

## Decision and targeted failure

This independent policy targets phase and intermediate-state ambiguity at policy handoff. One long skill includes low lift, high clearance, simultaneous transport/reorientation, alignment, descent, and early insertion. Timing is not identical: at index 1380 `demo12205` still has peg y near 0.160 m, while `demo12208` is near y=0.180 m and already aligned; final peg x also varies approximately -0.252 to -0.185 m. A policy dominated by an absolute trajectory can issue the wrong phase when a neighbor enters at a different valid state.

The policy predicts Cartesian targets relative to the **newest measured TCP at chunk start** and reanchors after every eight executed actions. This emphasizes local correction from the observed causal state instead of a fixed world path or implicit clock. The expected benefit is robustness to successful-path off-timing states; no demonstration establishes recovery from unseen errors.

## Implemented inputs and action

For each causal frame, `build_inputs` includes scaled measured `qpos` and `qvel`; peg, hole, target and lid poses relative to that frame's TCP; TCP motion relative to the newest chunk anchor; lid state; and eight explicit geometric phase cues. Those cues are peg clearance above the box wall, peg position in the hole frame, peg/hole rotation-vector error, and TCP-to-peg coupling distance. They are deterministic causal observations, not privileged phase labels or contact truth.

`chunk_context` holds the newest causal `tcp_pose`. `encode_targets` computes the demonstrated commanded TCP via the public Panda FK and encodes `T_anchor^-1*T_command` as 3-D translation in metres, 3-D exponential-coordinate rotation in radians, and the demonstrated gripper scalar. `decode_action` applies the SO(3) exponential map, composes the target with the fixed chunk anchor, and calls the verified `panda_kinematics.solve_ik` from freshly measured joints. Solver convergence, position/rotation error, iterations, limit contacts and maximum joint change are reported; an unconverged result is not silently changed.

## Model and loss

The standard `DiffusionBackbone` is unchanged. The model flattens `[2,73]` causal features as the condition and diffuses seven-dimensional local-twist targets. The framework's masked epsilon loss is the active loss. No auxiliary head is needed; `prior_loss` is a differentiable zero. There is no recurrent phase memory, and repeated denoising calls cannot advance physical state.

## Scaling, labels, and temporal semantics

Input scaling is fixed in physical units: translations by 0.6 m, rotations by pi where represented as rotation vectors, arm position/velocity by 3, fingers by 0.04 m and 0.2 m/s, lid by 1.85 rad and 4 rad/s, clearance by 0.5 m, and coupling distance by 0.2 m. The action normalizer is fitted by the framework only on valid authorized represented labels.

Prediction slots remain t-1 through t+14. Execution remains slots 1-8 starting at t. The current TCP anchor is fixed for those outputs and refreshed at the next framework replan; this is a transform of absolute target labels, not incremental accumulation between slots. All assigned `[1120,1500)` actions and masks remain intact, including both substantial action-supervised overlaps.

## Selection and handoff

Prefer this policy when the peg appears held, TCP state is trustworthy, scene displacement is modest, and a neighboring skill may have produced a valid but off-timing intermediate state. The two frames provide direction and coupling cues. Use the outgoing overlap to transfer when the peg is near hole-axis alignment, centered in y/z and descending to hole height.

If the hole/fixture moved substantially, use the destination-frame policy. If the held peg and TCP moved together at the source with a reliable peg pose, use the held-peg anchor policy.

## Limitations

Local action coordinates do not determine a new globally collision-free route. They cannot restore lost contact, reacquire a dropped peg, undo a collision, or make an unreachable target feasible. A chunk-fixed TCP anchor can accumulate model or execution error before replanning. Gripper closure and low coupling change are only observable proxies for a grasp. The demonstrations contain successful trajectories, not general recovery behavior.
