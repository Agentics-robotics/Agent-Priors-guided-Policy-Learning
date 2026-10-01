# Causal Phase and TCP-Residual Diffusion Policy

## Decision and target failure

This independent model targets noncanonical TCP/posture entry and ambiguity between red acquisition, red transport, buffer release/retreat and the blue-overlap phase. It learns Cartesian commands as residuals from the chunk-start TCP rather than as red- or destination-frame poses. A trainable two-frame encoder is shared by diffusion and a four-class auxiliary phase head.

Prefer this model when an upstream policy hands off at an intermediate state or when TCP is not at the demonstrated home but geometry still resembles an action-supervised phase. Use `buffer_red__h01` instead for substantial red displacement and `buffer_red__h03` for an intentionally moved buffer or placement-dominant case.

## Evidence, assumptions and benefit

Observed successful trajectories contain geometrically similar low-TCP states with different intent: demo10100 and demo10105 indices 80-100 approach/close on red; 240-280 lower/open at the buffer; and 440-460 approach/close on blue. High TCP at 320-340 is post-release retreat, unlike the initial high approach. The two causal observations, fingers, red/blue motion and relational distances provide available cues. The overlap includes final red lowering through initial blue lift.

Expected benefit is fewer phase confusions and smaller action-coordinate shifts when entry TCP changes. This is falsifiable in held-out intermediate entries, but successful demonstrations do not establish recovery. The residual anchor cannot remove red displacement from the target relation.

## Inputs and calling contract

Required arguments are `target_object="red"`, `successor_object="blue"`, and the seven-value world `buffer_pose_world`. They select the actual object fields and buffer relation/phase reference. Training binds `[-0.181,0,0.02,1,0,0,0]`.

Each causal frame has 54 features: scaled qpos/qvel/fingers, world TCP position and rotation-6D, and full TCP-relative poses of red, blue and the supplied buffer. Fixed physical scaling uses 0.5 m, 3 rad, 2.5 rad/s and 0.04 m; no evaluation data are fitted. Robot configuration and world TCP are retained because reachability, joint limits and posture matter. History is causal and framework duplication is used only at episode start.

## Actions and converters

The 7D action is world-axis TCP translation residual in metres, rotation vector in radians for `R_command R_anchor^T`, and native gripper scalar. `encode_targets` computes command TCP poses using verified FK and the causal chunk anchor. `decode_action` applies the exponential map, left-multiplies anchor rotation, adds translation to anchor position and calls `panda_kinematics.solve_ik` from freshly measured joints. Solver convergence, residuals, iterations, joint-limit contacts and maximum joint change are reported. No fallback controller or task-completing script is used. Framework representation normalization is fitted only on valid residual labels.

## Learned modules, labels and losses

A registered MLP maps the flattened 108 causal values to 256 condition features. The unchanged standard DiffusionBackbone predicts epsilon for the 16x7 action sample. A registered linear phase head reads the same learned embedding. All are in the model, optimizer, checkpoint and EMA.

The auxiliary target is current causal phase:

0. red acquisition;
1. red lift/transport;
2. red buffered release/retreat;
3. blue overlap acquisition/lift.

Labels use only current red/blue/TCP geometry, finger width and the supplied buffer; they do not use contact truth or future state. The phase mask is the framework-valid current slot. Masked cross-entropy is weighted 0.1 and has a real gradient through the shared encoder; masked diffusion remains the primary objective and is always active. Repeated DDPM denoising does not update phase memory because none is exposed.

## Temporal scope and handoff

History 2, prediction 16, execution 8 and 20 Hz are unchanged. Prediction slots are t-1..t+14, with 1..8 executed. Every assigned action and each 210-230 action overlap is bound. Core handoff requires observed buffered-red stability plus open/clear fingers. A phase prediction alone is not readiness or grasp proof. Preserve causal frames across the switch.

## Limitations

The phase heuristic is coarse and can label a closed-finger miss as transport. It is auxiliary supervision, not a safety monitor. Large TCP residuals, target displacement, collisions, drops, misses and occupied buffers are not covered by successful training. No contact truth or image input exists. IK feasibility diagnostics do not establish manipulation success. The global task success definition remains unchanged; this model covers only the assigned red-buffer responsibility and overlap.
