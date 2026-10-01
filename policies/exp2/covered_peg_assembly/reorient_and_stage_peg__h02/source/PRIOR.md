# Held-peg anchor prior

## Decision and targeted failure

This independent policy targets source and held-object displacement at the retrieval handoff. Across demonstrated index 1120 states, peg x is approximately -0.610 to -0.622 m and peg z is 0.028 to 0.040 m, with matching TCP variation. All inspected commands remain closed (`-1`). Deployment may provide a peg/TCP pair displaced together by retrieval while the grasp is still valid.

The action is represented in the **newest observed peg frame**. A rigid transform of a stably held peg therefore produces the same represented lift/reorientation/transport shape and transforms the decoded Cartesian targets by construction. This expected benefit is limited to stable-grasp displacement; it is not a claim that the demonstrations contain drop or grasp recovery.

## Implemented mechanism

For each of two causal frames, `build_inputs` includes scaled measured Panda `qpos`/`qvel`, TCP/hole/target/lid poses expressed in that frame's peg coordinates, the frame peg pose relative to the newest peg anchor, lid state, and box-wall clearance. The peg-to-TCP relation and its two-frame change expose an observable attachment cue. Hole-in-peg geometry remains present so the network can account for a fixed destination rather than blindly replaying a source-relative path. Robot state retains kinematic branch and reachability dependence.

`chunk_context` stores the newest causal `peg_pose` once per sampled chunk. `encode_targets` uses exact FK of the demonstrated seven-joint command and emits position plus 6D orientation relative to that anchor, followed by the demonstrated gripper scalar. `decode_action` projects the 6D orientation to SO(3), computes `T_world_peg * T_peg_command`, and invokes only `panda_kinematics.solve_ik` from freshly measured joints. It returns the solver's convergence, residual, iteration, limit-contact and joint-change diagnostics without hiding an unconverged target.

## Model, loss, and normalization

The published `DiffusionBackbone` is unchanged. Two `[2,66]` causal feature frames are flattened as global conditioning; the output is 10-dimensional. The masked DDPM epsilon objective remains the sole active objective. No auxiliary head is used, and `prior_loss` is a differentiable zero. All learned parameters are registered and framework-managed for optimizer, EMA and checkpointing.

Inputs use fixed physical scales, not evaluation-fitted statistics: translations 0.6 m; arm position/velocity 3; finger position 0.04 m and velocity 0.2 m/s; lid angle 1.85 rad and velocity 4 rad/s; clearance 0.5 m. The framework separately fits the action representation normalizer on valid authorized labels.

Prediction slots remain t-1 through t+14, with only slots 1-8 executed. The peg anchor stays fixed inside that 8-action execution chunk and is refreshed at the next replan. Initial history duplication and masks remain framework-owned. Every action in all twelve `[1120,1500)` slices is supervised.

## Applicability and handoff

Prefer this policy only when fingers are closed and peg-to-TCP geometry is stable over two frames. It is strongest for incoming low lift, clearance and high reorientation/transport under a shifted held peg. The `[1120,1270)` retrieval overlap and `[1380,1500)` insertion overlap are both action-supervised.

Hand off to `align_and_insert_peg` after near identity orientation, hole-relative y/z centering and descent are observed. If the hole itself moved, the destination-frame prior is the stronger policy. If peg pose is unreliable but TCP is trustworthy and the state is a valid successful-path intermediate, use the local-twist prior.

## Assumptions and limitations

The policy has no contact truth. Peg-frame anchoring is inappropriate when the peg is dropped or slipping. Moving only the peg changes the hole endpoint in peg coordinates; large source-to-hole changes remain outside demonstrated support even though the relation is observed. Rigid action covariance does not preserve box/lid collision geometry or arm reachability. No labels establish reacquisition, collision recovery, or bad-grasp correction.
