# Phase-conditioned local Cartesian prior

## Decision and target failure

This policy targets two related deployment failures: entry at an intermediate state produced by an adjacent skill, and phase aliasing between approach/contact motions for red and blue. A required high-level phase explicitly selects one of four coarse responsibilities. Cartesian targets are represented relative to the actual TCP pose at chunk construction, so each replan predicts local corrections from the current robot state rather than replaying an absolute path.

The four phases are semantic rather than a timer: `drawer_release_to_red` opens/retreats from the drawer and approaches red; `red_grasp_transport_place` closes on red, lifts, transports, and descends; `red_release_to_blue` opens over supported red, clears it, and transits toward blue; `blue_pickup` approaches, closes, and lifts blue. The training ranges `[230,445)`, `[445,620)`, `[620,780)`, and `[780,850)` are deliberately coarse to retain within-phase timing variation.

Observed evidence supports the ambiguity but not unseen recovery. Both inspected runs start closed at the drawer front at 230, open by 240, reach red contact near 445, and lift by 470. Near placement, `demo1000` is already opening by 608 while `demo1001` remains closed at 610. Blue approach/contact and terminal lift also vary. A phase input is therefore expected to suppress object/mode confusion at switches, but no demonstration proves recovery from a missed grasp or off-pad release.

## Inputs and arguments

The caller supplies `phase` plus fixed semantic identifiers `primary_object="red"`, `successor_object="blue"`, and `destination="red_goal"`. The identifiers select current observation fields. The phase is converted to a 4-D one-hot and is concatenated directly into the backbone conditioning, so it has a real effect on action diffusion.

Each causal 44-D feature row includes scaled measured joints, fingers, and velocities; red and blue poses relative to the measured TCP; red-goal and blue-goal points in TCP axes; and drawer position/velocity. Robot configuration is retained for reachability and IK-continuity context. History is causal H2 and carries across skill switches; no action history, future state, recurrent phase memory, or contact truth is used.

## Action and conversion

The 7-D action is `[TCP-frame XYZ metres, relative rotation vector radians, gripper scalar]`, all relative to the newest causal TCP frozen for the sampled chunk. `encode_targets` obtains world TCP labels by verified FK of demonstrated joint commands, applies `inverse(chunk_tcp) * commanded_tcp`, and takes the relative matrix logarithm. The framework fits normalization only on valid labels within each phase binding.

`decode_action` applies the rotation-vector exponential, composes the relative target with the same causal TCP anchor, and calls `panda_kinematics.solve_ik` from freshly measured joints for every executed slot. The represented gripper scalar is appended unchanged and uses nonnegative-open/negative-close executor semantics. Diagnostics expose IK convergence, residuals, iterations, limit contacts, joint change, and the active phase. This is a chunk-relative target representation, not an accumulated hidden controller state.

## Model, masks, and training

An unchanged independent `DiffusionBackbone` consumes flattened two-frame features and phase. No auxiliary head is needed. Framework masks bound each split, masked epsilon diffusion remains active for all valid labels, and `prior_loss` is a differentiable zero. The model, optimizer, EMA, H2/16/8 timing, DDPM settings, seed, and 20,000-update budget otherwise follow the fixed recipe.

The four bindings per trajectory cover every authorized action exactly once. They retain all `[230,470)` overlap with `open_drawer` and `[600,850)` overlap with `blue_insert`. Splitting does not change the skill responsibility: red transfer remains primary and the complete blue pickup handoff remains supervised.

## Applicability and limitations

Prefer this policy for represented intermediate entries, ambiguous red-versus-blue approach geometry, or a need to replan from an off-nominal but supported TCP/configuration. The high-level agent must infer phase from physical observations, not elapsed time. A wrong phase can select the wrong object or gripper behavior. Local coordinates are not by construction invariant to large red or goal displacement; the red-object and destination-centered policies cover those changes. Rotation-vector labels use the principal branch and assume relative rotations remain within demonstrated support. None of the policies proves contact recovery, collision avoidance, or feasible execution of unreachable sampled targets.
