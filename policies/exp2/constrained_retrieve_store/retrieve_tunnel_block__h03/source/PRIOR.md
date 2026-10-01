# Destination-frame geometry and clearance prior

## Decision and target failure
This independent policy targets phase ambiguity around closing, low extraction, clearance lift and the start of transfer. The authorized demonstrations consistently align near TCP z 0.089, close and pull in negative world x while the block remains near z 0.08, lift past the 0.16 m tunnel roof, and only then move toward the drawer. This policy retains that global structure by learning absolute TCP poses in the observed `target_pose` destination frame.

It also exposes rather than hides phase cues: two-frame block/TCP displacements, their coupling error, finger width, object/TCP transform, destination/object transform and signed tunnel/roof margins all enter the causal condition. These are observable proxies, not contact truth.

## Model and inputs
`build_inputs` produces two 70-channel frames. Each contains qpos/qvel, finger width, world TCP and block poses, destination-relative block pose, block-relative TCP pose, drawer state, roof clearance and four tunnel-boundary margins. Eight causal cross-history values encode block displacement, TCP displacement, displacement mismatch and current proximity. Only the two observed frames are used; future labels never enter deployment inputs.

`ClearanceGeometryDiffusion` flattens 2x70 channels and uses the unchanged public `DiffusionBackbone` with a 140-D global condition and 10-D actions. All learned parameters are in the registered backbone. No auxiliary head is needed because the phase mechanism is an analytic causal input representation. Masked epsilon diffusion remains the full learned objective and `prior_loss` is differentiable zero.

## Destination-frame targets
`encode_targets` obtains each demonstrated command's verified FK TCP pose and computes `inverse(target_pose_before_action) @ commanded_TCP`. The 10 channels are destination-frame translation in metres, rotation 6D, and the demonstrated gripper scalar. Representation normalization is fitted later only from valid bound labels.

At execution, `decode_action` projects a finite 6D rotation, composes the target with the freshly observed destination, and sends the resulting world TCP pose through verified IK from fresh qpos. It appends the represented gripper scalar and reports the world target, unprojected target, IK convergence/residuals/limits and whether the guard intervened.

## Narrow low-clearance guard
The decoder applies no scripted waypoint. It only constrains a sampled pose when all observable proxy conditions hold: measured finger width is below 0.055 m, measured TCP is within 0.06 m of the block, and block z is below the caller's 0.16 m clearance. In that case a request toward the drawer in +world x is capped at `object_x + 0.03 m`. While the block centre remains within the tunnel's x extent, target y is also clamped to the object-centre lane after subtracting block half-width and wall thickness. Orientation and gripper are unchanged.

The demonstrated low motion is negative-x extraction or lift and therefore remains unprojected; the task-space roundtrip check verifies this. The guard's purpose is conservative ordering under a bad diffusion sample, not recovery or guaranteed collision avoidance.

## Calling and handoff
HL supplies the observed block field, destination field and 0.16 m clearance. All values enter adapters. Timing remains history 2, prediction 16 and execution 8 at 20 Hz. All 12 [300,840) segments are supervised, retaining [300,440) predecessor overlap and [680,840) successor overlap.

Prefer this policy when world geometry is near training support and the high-level agent is uncertain whether a closed-near-object state should continue extracting/lifting or begin transfer. Enter with an open drawer and valid destination. Hand off after the observed block is outside and above clearance, its motion remains coupled to TCP under closed fingers, and destination-directed transfer begins.

## Assumptions, limitations and complementarity
The guard can be fooled by an empty closed gripper near the block because contact truth is absent. It cannot retry a missed grasp and may reject a valid unseen maneuver that intentionally transfers low. Neither the destination frame nor the guard proves IK reachability or collision freedom. Large reliable block displacement is better covered by h01's object-frame action. An unusual but valid predecessor TCP/configuration is better covered by h02's fresh-TCP body correction. All three remain limited by successful demonstrations with no failed-grasp recovery.
