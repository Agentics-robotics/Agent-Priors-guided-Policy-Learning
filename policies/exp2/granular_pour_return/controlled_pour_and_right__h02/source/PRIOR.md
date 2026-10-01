# Permutation-invariant particle-phase prior with fresh-TCP residual actions

## Decision and target failure

This policy targets phase ambiguity and modest intermediate-state deviation. The source follows a similar nominal tilt arc across demonstrations, but discharge timing differs: demo12300 at action 800 still has several particles airborne or near the source while demo12303 and demo12309 show all twelve on the bowl floor. Actions 900-1100 then cover maximum tilt and initial righting with settled particles. A policy that relies on time or particle identity can confuse “continue pouring” with “right the source.”

The mechanism has two parts: a learned permutation-invariant particle set encoder for phase evidence, and local TCP-frame SE(3) actions applied from fresh state. Prefer it for a secure grasp, mid-phase entry, unexpectedly early/late discharge, or small execution deviations. Use h01 for major bowl displacement and h03 for an atypical rigid grasp transform.

## Responsibility and evidence

All twelve `[180,1740)` segments remain action-supervised. Incoming overlap `[180,360)` and outgoing overlap `[1460,1740)` are not trimmed to isolate discharge. Evidence at actions 700-1200 in demo12300, demo12303 and demo12309 shows source tilt and substantially different particle settlement timing. These are successful nominal trajectories, not demonstrations of recovery.

## Causal inputs

`build_inputs` uses two causal frames. Initial history duplication and cross-skill history are framework-owned. For each frame it produces:

- `robot [58]`: scaled measured qpos/qvel; source relative to bowl; TCP relative to source; return target and TCP relative to bowl; settled/source fractions; upright cosine; relative source height;
- `particles [12,8]`: for each observed particle, three bowl-frame coordinates, three source-frame coordinates, a geometric bowl-settled indicator and a geometric source-contained indicator.

The same shared MLP maps every particle token to 64 dimensions. Mean and max over the 12-token axis remove particle ordering. Two pooled frames and two learned robot embeddings form a 384D condition. All trainable encoders are registered in the model and therefore enter optimization, EMA and saved state. No future observation, contact truth or action history enters deployment conditioning.

The geometric indicators use the published bowl interior, floor/wall heights, particle radius and source dimensions. They are observable approximations, not simulator contact or success labels.

## Learned action and conversion

Each 7D action contains:

1. translation in metres in the TCP frame observed immediately before the action (3),
2. local rotation vector in radians (3),
3. demonstrated native gripper scalar (1).

`encode_targets` computes `inverse(T_world_measured_tcp) @ T_world_commanded_tcp` for each slot, using commanded poses from verified FK. `decode_action` composes the local transform with the fresh current observed TCP and calls `panda_kinematics.solve_ik` from freshly measured joints. Convergence, residuals, iterations, joint-limit contacts, maximum joint change and world target are reported. The decoder does not conceal failure or add scripted task behavior.

Applying each represented slot from fresh TCP state is intended to correct modest execution deviations. It is not an absolute destination anchor, so h01 is preferred for a large bowl relocation.

## Auxiliary phase objective and gradient path

For each window, `encode_targets` supplies three current causal labels from `history[-1]`: settled fraction, source-contained fraction and source upright cosine. These labels are always valid causal observations, so `phase_mask [1]` is one. The phase head consumes the same 384D condition used by diffusion. Its masked MSE is multiplied by `0.05` and returned as `prior_loss`; the total remains diffusion epsilon loss plus this auxiliary term.

This creates a real gradient path through `phase_head`, `particle_encoder` and `robot_encoder`. It regularizes phase information but does not replace diffusion learning and is not used as privileged inference input or as the task success test.

## Backbone and temporal contract

The action denoiser is the unchanged published `DiffusionBackbone(384, 7, training_config)`. Prediction horizon is 16 at 20 Hz, representing t-1 through t+14; the executor uses slots 1-8. The framework fits action normalization only from valid represented labels and applies the supplied padding mask. The numerical budget and final-EMA selection remain unchanged.

## Expected benefit and limits

The falsifiable expected benefit is improved selection of continue-tilt versus righting behavior when particle settlement timing or entry phase differs while remaining within demonstrated relational states, plus reduced sensitivity to small TCP tracking error. Mean/max pooling makes this independent of particle index by construction.

The policy assumes reliable particle, source, bowl and TCP poses and a secure reachable grasp. It cannot infer hidden contact, recover a spill, diagnose a lost grasp, or guarantee behavior for particle configurations absent from successful data. Closed fingers alone do not prove contact. Local residuals can drift under large relocation; h01 covers the absolute displaced-bowl case. Unusual rigid grasp geometry is delegated to h03.

## Handoff

Entry may occur from early carry through tilt or righting when the grasp is secure. Exit and successor readiness retain the unchanged physical evidence: at least ten particles settled, source approximately upright and held, and outbound motion begun. Nominal readiness is near action 1680, with the complete `[1460,1740)` overlap supervised for robust handoff.
