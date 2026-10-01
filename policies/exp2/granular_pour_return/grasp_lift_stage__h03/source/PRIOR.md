# Prior: bowl-frame targets with causal phase supervision

## Decision and targeted failure

This independent policy targets phase ambiguity caused by variable closure and lift timing, particularly when invoked just after closure or after an early handoff. It predicts absolute Cartesian targets in the observed bowl frame for stable successor staging and trains a shared causal condition encoder with an auxiliary three-class phase target.

Observed evidence supports the difficulty: demo12300 first commands closure at index 148, while demo12303 commands closure by 136 and has closed fingers by observation 140. At index 180 their source heights are about 0.029 and 0.056 m respectively, yet both reach a similar bowl-relative stage around 320-360. This variation motivates phase-sensitive conditioning. It does not show recovery from a failed grasp.

## Inputs and arguments

`source_object="container"` selects the manipulated source for phase and co-motion features. `destination_object="bowl"` selects the observed destination pose and action frame. Every replan refreshes the bowl frame. Two 61-channel causal frames encode bowl-relative source/TCP poses, source-relative TCP pose, world positions, qpos/qvel, source-relative particle centroid, finger width, and source/TCP translations across the two frames. Fixed physical scales are used. Measured joints and world positions preserve reachability and robot configuration dependencies.

The phase label is computed from the same causal state represented by the history: class 0 is open/approach (finger width at least 0.05 m), class 1 is closed-low (narrower fingers and source z below 0.02 m), and class 2 is closed-lifted. It is an observation-derived proxy, not contact truth. Its mask is one for each valid window; no padded future action mask is repurposed. Inference receives no phase label and no future state.

## Learned modules and gradient paths

A registered MLP maps the flattened 122 input channels to a 128-dimensional condition embedding. The unchanged standard DiffusionBackbone consumes this embedding. A registered linear phase head also consumes it. `compute_loss` adds 0.1 times masked phase cross-entropy to masked epsilon diffusion loss. Thus the auxiliary gradient reaches both the phase head and the same condition encoder used by diffusion; action diffusion remains the main objective. All modules are part of the model, optimizer, checkpoint, and EMA. The auxiliary does not advance memory during denoising and is not used as a scripted controller.

## Actions and conversion

The 10D action is bowl-frame position xyz in metres, bowl-frame 6D orientation, and native gripper scalar. `encode_targets` computes `inverse(chunk_bowl) @ commanded_tcp_pose`. `decode_action` robustly projects 6D to SO(3), composes the chunk bowl frame, and runs verified Panda IK from fresh measured joints. It reports convergence, residuals, iterations, limit contacts, and maximum joint change. The gripper sign rule is unchanged.

The host fits normalization to valid bowl-frame labels. H2/16/8, 20 Hz, DDPM100/100, and the 20,000-update budget remain fixed. All twelve [0,360) segments are supervised, including the full [180,360) overlap with `controlled_pour_and_right`.

## Expected benefit and limitations

Expected benefit: the shared condition representation should better separate open approach, closed-low lift initiation, and closed-lifted carry despite timing variation, while bowl-frame targets stabilize the high handoff relation if the bowl pose changes within reach. This is a falsifiable expectation, not proof from design checks.

The phase head cannot certify grasp contact, identify all slips, or justify handoff by itself. The auxiliary thresholds encode only the demonstrated ordering and may fail on unseen reordering or noisy observations. Bowl-frame pregrasp targets are not invariant to source-only displacement; the source-frame policy covers that failure. Large local TCP errors are better suited to the TCP-delta policy. IK feasibility, collision clearance, geometry changes, and unseen bowl/source rotations remain limitations.
