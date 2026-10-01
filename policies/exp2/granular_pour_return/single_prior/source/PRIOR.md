# Single Prior Cartesian Diffusion Policy

## Decision

This submission is one full-task diffusion policy with no segmentation, runtime high-level controller, policy selection, stage label, caller-selected object or call parameter. Every original trajectory is bound once over its complete `[0, stop)` action range with `call_args={}`.

The prior is a **continuous causal relational Cartesian coordinate system**:

1. While the observed container is low, Cartesian target position is represented relative to the current container origin. This covers source approach, handle alignment, grasp and early lift under source displacement.
2. As the observed container rises, the coordinate origin changes smoothly from container-relative to bowl-relative. This covers transport, positioning over the bowl, pour and unpour under bowl displacement.
3. Only after observed particles geometrically occupy the bowl and the observed container is nearly upright does the origin change smoothly toward the observed marked-return pose. This covers return, lowering, release and retreat under marker displacement.
4. Desired TCP orientation is represented at all times relative to the current observed container orientation. This transfers handle alignment and held-object orientation changes to a rotated source pose.

These weights only define coordinates. They never emit an action, choose a policy or expose a phase. The diffusion model still learns every movement and gripper decision jointly from the full demonstrations. Anchor weights are recomputed from each fresh causal observation when an executed slot is decoded.

## Evidence and learning difficulty

The demonstrations are long (about 1,935--1,953 actions each) and contain visually/kinematically distinct but uninterrupted behavior. Detailed inspection of `demo12300` shows:

- open-gripper home at index 0;
- descent toward the container handle at indices 110--130;
- closing near index 145, followed by early lift at 175--220;
- a lifted source near index 250 and transfer toward the bowl by 280--310;
- progressive held-container rotation over the bowl, with particles starting to leave by 780 and settled in the bowl by about 820--1000;
- reverse rotation to upright through 1550--1650;
- transfer to the return marker around 1700--1760;
- lowering around 1810--1840, opening by 1870, and retreat by 1910.

Uniform full-trajectory inspection of all twelve demonstrations shows the same complete ordering with initial container variation. Initial container centers span approximately x `[-0.458,-0.443]` m and y `[-0.214,-0.202]` m. End particle scatter varies substantially, while the returned container is consistently near `[-0.470,-0.282,0]` m and upright. The bowl and return marker do not vary in training. Consequently, source translation has direct empirical support, while bowl/marker translation is a construction-based equivariance whose collision-free path coverage is not demonstrated.

The main difficulty is not merely reaching a point: contact-sensitive handle alignment, grasp timing, maintaining a grasp through a long transport and large rotation, granular transfer timing, disambiguating pre-pour and post-pour upright states, and replacing the source without releasing early all share one model. Particle geometry is therefore retained causally to distinguish material-transfer outcomes, while qpos/qvel and absolute TCP position retain robot configuration, reachability and dynamic dependencies that a purely object-relative input would discard.

## Inputs

`build_inputs` consumes exactly two causal observations, oldest first. At episode start the framework duplicates the first observation. There is no future observation, contact truth, image, action history or hidden phase input.

Each frame becomes 141 fixed-scale features:

- 9 measured joint/finger positions and 9 joint/finger velocities;
- absolute TCP position relative to the public robot base and TCP rotation 6D;
- container-to-TCP, bowl-to-TCP, return-to-TCP, container-to-bowl and container-to-return translations;
- container rotation, TCP rotation relative to the container, bowl rotation and return rotation, all as continuous 6D columns;
- all 12 particle positions in both container-local and bowl-local coordinates, sorted deterministically in bowl-local coordinates to remove particle identity dependence;
- the three continuous relational-anchor weights.

The resulting model input shape is `[2,141]`; model conditioning is the flattened 282-vector. Position channels use fixed physical scales (0.5 or 0.8 m), arm positions use 3 rad, arm velocities 2.5 rad/s, finger positions 0.04 m and finger velocities 0.2 m/s. Rotation columns and anchor weights are already bounded. This fixed physical normalization fits no evaluation data. The framework separately fits the learned-action midpoint/half-range normalizer only from valid encoded training labels.

## Learned action and converters

The learned action dimension is 10:

`[position_offset_world_m(3), container_relative_rotation_6d(6), gripper_scalar(1)]`.

For every valid training slot, `encode_targets` obtains the exact demonstrated commanded TCP pose by applying public URDF FK to the seven native joint command values. It subtracts the relational anchor computed from that slot's pre-action observation, expresses target rotation as `R_container^T R_command`, converts that rotation to its first two columns, and retains the native gripper scalar. The host mask handles edge padding; there are no auxiliary labels.

At execution, `decode_action` uses the fresh measured observation rather than stale model history. It recomputes the relational origin, adds the learned world offset, projects the two learned rotation columns to SO(3) by a guarded Gram--Schmidt operation, left-multiplies by current container rotation, and constructs one world-frame TCP target. Only `panda_kinematics.solve_ik` converts that target to seven absolute Panda joint targets, starting from freshly measured qpos. The gripper scalar is appended unchanged; the common executor applies its documented sign rule. Diagnostics expose convergence, position and rotation residuals, iterations, joint-limit contacts, maximum joint change, rotation fallback, anchor weights and observed soft bowl occupancy. An unconverged IK result is reported and never replaced by another controller.

The representation roundtrip is exact before normal floating-point/IK tolerance: encode and decode use the same observed frame and anchor, and rotation 6D reconstructs the commanded orientation. The declared independent task-space tolerance is 0.01 over metres, radians and gripper scalar; this accommodates only the shared IK numerical solution, not manipulation error.

## Relational-anchor mechanism

Let `c`, `b` and `r` be the observed container, bowl and return origins. A smooth lift value is computed from observed container height. A soft fraction of particles in the bowl is computed from particle positions transformed into the bowl frame and the supplied bowl/particle geometry. A narrow smooth upright value is computed from the container's vertical-axis cosine. The normalized origin is

`a = w_container c + w_bowl b + w_return r`,

where low source height favors `w_container`, lifted height favors `w_bowl`, and `w_return` becomes significant only when particles are in the bowl and the source is nearly upright. The position label is `p_command - a`.

Thus a translated low source translates approach/grasp targets, a translated bowl translates lifted pouring targets, and a translated return marker translates nearly upright return targets without requiring the network to reproduce the world translation. Transitions remain continuous, and the network observes the same weights it is decoded with. This covers translation of the objects approached/manipulated/used as destinations. It does not make joint reachability, obstacle avoidance, grasp mechanics or granular dynamics invariant.

## Model and gradient path

`RelationalPourPolicy` contains the published `DiffusionBackbone` unchanged, with condition dimension 282 and action dimension 10. Keeping the mature backbone avoids an unsupported architecture change; the scientific prior lives in causal input construction and action coordinates. The backbone predicts epsilon for `[B,16,10]` noisy actions. Masked epsilon diffusion loss is the total training loss. `prior_loss` is a differentiable zero because no artificial auxiliary objective is needed. All trainable parameters are registered in the backbone and therefore enter AdamW, gradient clipping, EMA and checkpoint state. There are no trainable adapter modules omitted from optimizer/EMA state.

The fixed temporal contract is history 2, prediction horizon 16, slots `t-1` through `t+14`, execution of slots 1--8, and 20 Hz. Numerical training follows the declared assignment budget: 60,000 updates, batch 128, DDPM 100 train/100 inference steps, epsilon prediction, clip-sample, AdamW at `1e-4`, weight decay `1e-6`, gradient norm 1, 500-step warmup, cosine learning-rate schedule and EMA `0.999`, selecting the last EMA for seed 0.

## Full-task entry, exit and calling contract

Entry is the ordinary task observation or a causally observed intermediate state with all declared fields valid and the required Cartesian targets reachable. Invocation is always `{}`. This policy has no successor and should be replanned continuously by the framework.

Exit remains exactly the supplied success definition: at least ten of twelve particles in the bowl and the source container upright in the marked return region within its height tolerance. An open gripper, final-looking arm posture, predicted action, or anchor weight does not establish success. `HANDOFF.json` records descriptive entry/exit evidence only and is not a runtime controller.

## Expected benefit and falsifiable limitations

Expected benefit: compared with world-frame Cartesian targets, translating the source, bowl or return marker during its relevant continuous geometric regime produces the same learned residual and therefore translates the decoded world target directly. Source-position variation across demonstrations supports the first regime; the fixed bowl/marker make the other two benefits falsifiable construction-based predictions rather than demonstrated coverage.

Expected failures and observable signatures:

- **Unreachable displacement:** IK reports `converged=false`, nontrivial residual or limit contacts; the TCP fails to approach the decoded target.
- **Missed/lost grasp:** finger closure occurs without handle alignment, container height does not follow TCP height, or TCP-to-container relative pose changes sharply. No contact truth is available.
- **Misleading anchor geometry:** a lifted but ungrasped container, premature spill, sensor noise or a dropped container can change the coordinate blend without providing a recovery action.
- **Path/collision failure:** large independent source/bowl/marker translations can require collision-free trajectories absent from training even though endpoints translate equivariantly.
- **Granular failure:** fewer than ten particles occupy bowl bounds after the tilt, particles keep moving outside the bowl, or the source empties differently under changed dynamics.
- **State ambiguity outside support:** unusual intermediate states can resemble the pre-pour or post-pour upright configuration; particle-local features reduce but do not eliminate this ambiguity.
- **Container-relative orientation failure:** after a drop, target orientation can follow the fallen container and attract the TCP rather than safely recovering.
- **Release/return failure:** source base remains outside marker bounds, z exceeds tolerance, upright cosine is too low, or fingers remain closed.

No scripted retry, phase advancement, collision planner, success detector or alternate policy is hidden in the decoder.
