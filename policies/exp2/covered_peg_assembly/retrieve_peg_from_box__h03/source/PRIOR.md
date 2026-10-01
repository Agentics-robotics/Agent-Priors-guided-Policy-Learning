# Phase-observable world-Cartesian prior

## Decision and targeted failure

This independent policy targets active-object and phase ambiguity across the long responsibility. In `demo12200`, the hand holds the lid at indices 740-750, opens around 760-770, clears at 800-950, descends toward the peg at 990-1080, closes by 1100, and lifts through 1120-1240. `demo12208` follows the same ordering with a shifted peg and different detailed timing. These are observed successful trajectories, not evidence for recovery.

Unlike the other priors, actions remain absolute world TCP targets: position (metres), 6D rotation, and native gripper scalar. The prior acts through explicit causal phase features, retaining the demonstration's world and robot-reachability structure.

## Causal phase representation

Each of the two observation frames contributes 62 state/geometry channels plus 10 recent-motion channels, for `[2,72]` input:

- arm qpos/qvel and finger position/velocity;
- absolute TCP, peg, and lid poses relative to the public robot base translation;
- TCP pose relative to the peg;
- TCP vector and distance to the lid handle, computed from observed lid pose and public `lid_handle_local`;
- finger width, peg height relative to the public box wall, and lid angle/velocity;
- on the newest frame, two-frame peg and TCP linear motion, change in TCP-minus-peg displacement, and finger-width change.

The oldest frame's explicit recent-motion channels are zero; the newest uses only those two causal frames at 20 Hz. If the framework duplicates the first episode frame, all recent-motion terms correctly become zero. These cues distinguish lid release, open approach, near-peg closure, and carrying lift without future observations, contact truth, or hidden memory. They are not proof of contact.

Fixed physical scaling is deterministic. The framework fits represented-action normalization only from valid authorized world-action labels. Padding is masked.

## Action labels, model, and gradients

`encode_targets` obtains the commanded world TCP pose by FK of each demonstrated native arm command and encodes `[position, rotation_6d, gripper]`. `decode_action` reconstructs the world pose, uses a deterministic Gram-Schmidt fallback only for degenerate sampled 6D vectors, and calls `panda_kinematics.solve_ik` from fresh measured joints.

The standard `DiffusionBackbone` is unchanged. The flattened causal phase representation conditions epsilon prediction. Masked diffusion loss remains fully active. There is no auxiliary head; `prior_loss` is a differentiable zero because the prior is the structured input. All trainable parameters therefore remain in the registered model, optimizer, EMA, and checkpoint.

## Feasibility and diagnostics

The decoder preserves native gripper sign and reports IK convergence, position and rotation residuals, iterations, lower/upper limit contacts, maximum joint change, and world TCP target. World targets preserve configuration and reachability dependence rather than claiming invariance. IK/projection checks establish conversion consistency only, not collision freedom, contact, or success.

## Expected benefit and complementarity

The expected, falsifiable benefit is fewer wrong-phase actions at early and intermediate handoffs than a representation lacking explicit lid-handle, finger, peg-height, and co-motion cues. This policy is preferred for a still-held lid or uncertain phase. It is expected to fail on large peg displacement because the output is world absolute; the peg-frame policy covers that case. It may also be brittle from a perturbed hand configuration; the current-TCP policy targets that case.

## Coverage and handoff

Call arguments are `{}`. Every `[740,1270)` occurrence is supervised, including incoming `[740,950)` and outgoing `[1120,1270)` overlaps. Exit only on combined evidence: closed fingers, peg above the wall, and approximately stable peg-to-TCP relation while both move. No individual feature is a success condition.

## Limitations

The deterministic phase features can still alias stationary or unusual states. All training pickups succeed, so there are no failed-grasp negatives and no demonstrated miss, drop, regrasp, blocked-lid, or wall-contact recovery. World outputs do not generalize by construction to displaced pegs and cannot guarantee IK reachability or collision avoidance.
