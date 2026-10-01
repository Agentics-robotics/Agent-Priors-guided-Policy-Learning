# Drawer-frame Cartesian tracking

## Decision and target failure

This policy targets a reachable displacement of the drawer/cabinet anchor and entry after nonzero opening progress. All demonstrations use one cabinet calibration and nearly identical drawer timing, so a world-pose policy can memorize the demonstrated front. The expected benefit is transfer of the approach, front acquisition, constrained pull, release and initial retreat when the caller supplies a changed drawer pose. This is an expected benefit, not evidence of rollout success or recovery.

## Mechanism

For each causal observation, the adapter builds an articulated drawer frame. Its zero-opening pose comes from `drawer_origin_world_m` and `drawer_orientation_wxyz`; fresh `drawer_position` translates the origin along drawer-local -X. TCP and red poses are represented in that frame. Joint positions/velocities, finger state, drawer progress/velocity and remaining desired opening retain robot reachability and contact-relevant state.

Each demonstrated native arm command is converted by public URDF FK to its commanded TCP pose. `encode_targets` expresses that pose in the drawer frame from the corresponding pre-action observation as position (metres), 6D rotation, and the unchanged gripper scalar. `decode_action` recomputes the drawer frame from the fresh physical observation, maps the represented target to a world TCP pose, projects sampled 6D rotation by Gram-Schmidt, and calls only `panda_kinematics.solve_ik` from freshly measured joints. IK convergence, residuals, iterations and limit contacts are reported. Thus labels, learned outputs and execution use the same reference convention. No synthetic augmentation is used.

## Model and training

Two frames of 39 causal features are flattened as global condition for the unchanged standard Diffusion Policy backbone. There are no auxiliary heads. The diffusion epsilon loss is masked by the framework; `prior_loss` is differentiable zero. All 12 bindings supervise [0,470), including 230:470 overlap. The framework fits representation normalization on these 10-dimensional labels only. History is two frames; the framework duplicates the first episode observation only at initialization. Horizon 16, slots t-1 through t+14, execution slots 1..8, 20 Hz, DDPM/EMA and the declared 20,000-update budget are unchanged.

## Evidence and scope

In demo1000, the TCP is at the closed front around 140, drawer_position increases through 0.0106/0.0871/0.2226 at 160/180/200, and reaches 0.300 by 220-230 while the TCP and red translate with it. The gripper opens and the TCP retreats upward at 240-300. Demo1001 has the same drawer relation with a different red Y. This supports the relational mechanism and identifies narrow training support; it does not demonstrate shifted cabinets.

## Invocation and handoff

The caller must supply a finite current zero-opening drawer pose using the same local -X convention, plus a desired opening. Training binds these to [0.19,0,0.035] m, identity wxyz, and 0.300 m. Prefer this policy when those quantities are reliable. `red_transfer` may receive after drawer_position exceeds 0.26 and red is exposed; the clean nominal handoff additionally has released front contact and red/TCP upward co-motion with closed fingers. The retained 230:470 actions support earlier and later switches.

## Limitations

Frame equivariance does not establish physical contact, reachability, collision freedom or grasp success. Large shifts/rotations, bad calibration, missed front contact and failed pulls are not covered. The post-opening pickup remains drawer-referenced but is not invariant by construction to independent red displacement; only modest demonstrated red variation is available. A closed gripper is not grasp proof, and no demonstration establishes recovery from drops or misgrasps.
