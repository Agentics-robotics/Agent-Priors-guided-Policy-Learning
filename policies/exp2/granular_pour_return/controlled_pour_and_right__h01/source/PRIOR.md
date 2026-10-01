# Bowl-frame absolute TCP trajectory prior

## Decision and target failure

This policy targets displacement of the bowl relative to the fixed training scene. It learns every demonstrated commanded TCP target as an absolute pose **in the observed bowl frame**, not in world coordinates. The carry to the bowl, the pour arc and the righting arc therefore move with a translated or rotated bowl by construction. The training bowl never moves, so this is a structural expected benefit rather than observed displacement performance.

Prefer this policy when the bowl pose is reliable, the source is securely held with a near-nominal source-to-TCP transform, and the resulting world targets are reachable. Use the phase-aware h02 policy when particle discharge or entry phase is ambiguous. Use h03 when a secure grasp has an unusual rigid source-to-TCP transform.

## Evidence and retained responsibility

In demo12300, demo12303 and demo12309, actions 320-360 stage the source near `(-0.259, 0.224, 0.400)` m above the fixed bowl, actions 700-1100 execute the bowl-centered tilt/discharge region, actions 1460-1680 right the source, and action 1740 is outbound. The evidence supports the nominal relationship, not recovery or displaced-bowl success.

All twelve assigned intervals `[180,1740)` are supervised. This retains 180 actions of incoming overlap `[180,360)` with `grasp_lift_stage` and 280 actions of outgoing overlap `[1460,1740)` with `return_place_release` in every trajectory.

## Inputs and normalization

`build_inputs` consumes exactly two causal observations, oldest first. At an episode start the framework duplicates the initial observation; history remains causal across skill switches. Each frame has 130 deterministic finite features:

- measured arm/finger positions and velocities, scaled by physical ranges;
- container, TCP and return-target poses relative to the bowl;
- the observed container-to-TCP pose, retaining grasp geometry;
- all twelve particle positions in bowl and container coordinates, sorted deterministically by bowl-frame position;
- observed settled/source fractions, source upright cosine and bowl-relative source height.

Translations are divided by declared metre scales, joints by broad physical scales, and rotations use 6D matrix columns. No evaluation-fitted statistics or future states enter the condition. Measured joints are retained because changing reference frame does not remove IK reachability.

## Learned action and conversion

The 10D unnormalized action is:

1. bowl-frame target TCP position in metres (3),
2. bowl-frame target TCP orientation as continuous 6D rotation columns (6),
3. demonstrated native gripper scalar (1).

`encode_targets` obtains the commanded Cartesian pose using `panda_kinematics.commanded_tcp_poses`, then computes `inverse(T_world_bowl) @ T_world_commanded_tcp` from the action observation for every valid slot. The framework fits the representation normalizer only on these authorized labels and applies the supplied action mask, including edge padding.

`decode_action` projects 6D orientation with a safe Gram-Schmidt operation, composes the target with the **fresh current** bowl pose, and calls `panda_kinematics.solve_ik` from freshly measured qpos. It reports convergence, position/rotation residuals, iterations, joint-limit contacts, maximum joint change and target pose. It does not hide an unconverged result or run a task-completing script. The gripper scalar is preserved; the common executor applies its binary sign rule.

## Model and gradient path

`BowlFramePolicy` flattens the two 130D frames and conditions the published `DiffusionBackbone(260, 10, training_config)`. The backbone and temporal recipe are unchanged: history 2, horizon 16, slots t-1 through t+14, execution slots 1-8, 20 Hz, DDPM epsilon prediction. The masked epsilon loss is the total loss; `prior_loss` is a differentiable zero because the reference-frame architecture itself is the prior. All trainable parameters are registered in the backbone and therefore enter optimization, EMA and checkpoint state.

## Expected benefit, assumptions and limits

The falsifiable expected benefit is that, for reachable bowl displacements with otherwise comparable relationships, bowl-centered TCP waypoints translate/rotate with the bowl instead of remaining at memorized world coordinates. This covers transport-to-bowl, pouring and righting. The outbound target is retained as bowl-relative conditioning, but displacement invariance does not prove correct behavior when bowl and return target move independently.

Assumptions are reliable object poses, a secure near-nominal grasp, adequate wall clearance and an IK-reachable scene. The policy does not compensate grasp slip, infer contact truth, recover spills/collisions, or prove that closed fingers hold the source. Large grasp-transform changes are delegated to h03; phase ambiguity is delegated to h02.

## Handoff

Entry requires a securely held, lifted source. Exit requires the unchanged task evidence: at least ten particles settled in the bowl, source approximately upright and held, and outbound motion begun. Nominal successor readiness is around action 1680, while the fully supervised `[1460,1740)` overlap permits robust takeover throughout late righting and initial outbound travel.
