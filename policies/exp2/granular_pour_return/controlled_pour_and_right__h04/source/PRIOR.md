# Absolute bowl-frame source-pose prior with fresh rigid-grasp reattachment

## Decision and observed target failure

`controlled_pour_and_right__h04` targets loss of terminal source righting while also making the bowl-relative manipulation move with a displaced bowl. In the fixed in-distribution validation, h01, h02 and h03 passed 8/12, 9/12 and 6/12 runs. Every one of the 13 failed runs still had all twelve particles in the bowl but had `container_upright=false`; all 36 runs kept the gripper closed and reported no unconverged IK step. In three h03 failures the exit rule had been met earlier and was lost by the final state. These are observed facts under the supplied final-state protocol, not an identified causal mechanism and not out-of-distribution evidence.

The new structural decision is to diffuse an **absolute desired container pose in the observed bowl frame**, not an incremental source motion or a TCP residual. Each target is then attached to the robot through the freshly observed rigid container-to-TCP transform. The represented orientation is therefore the manipulated source orientation directly. This is intended to anchor late righting instead of integrating local source increments while retaining the ability to adapt the TCP target to a different rigid grasp.

Prefer h04 when the bowl, source and TCP poses are reliable, the source is securely and approximately rigidly held, bowl displacement is plausible, and direct terminal source-orientation anchoring is useful. Use h02 for a highly ambiguous partial-discharge entry where local correction is preferred. A closed gripper is not evidence of a secure grasp.

## Evidence and retained responsibility

All twelve assigned intervals `[180,1740)` are supervised. The incoming `[180,360)` overlap retains 180 actions per trajectory with `grasp_lift_stage`; the outgoing `[1460,1740)` overlap retains 280 actions per trajectory with `return_place_release`.

Inspected demonstrations support the nominal relation. In demo12300, the source is staged above the bowl near action 320, tilted through actions 700-1200, partially discharged at 800, tilted but already empty at 1460, nearly upright at 1680 and upright outbound at 1740. In demo12303, all particles are already on the bowl floor at 800, and the upright source is already outbound by 1680. Demo12309 again shows bowl-centered tilt at 700-1460, upright at 1680 and outbound at 1740. This supports particle-phase conditioning and direct tilted-to-upright source supervision. It does not demonstrate recovery or a displaced bowl.

## Causal inputs and registered encoder

`build_inputs` uses exactly two causal observations, oldest first. The framework duplicates only the initial episode observation and preserves history across policy switches. Each frame supplies:

- `robot [58]`: scaled measured arm/finger positions and velocities; container relative to bowl; TCP relative to container; return target relative to bowl; TCP relative to bowl; settled fraction, source-contained fraction, source upright cosine and bowl-relative source height;
- `particles [12,8]`: each observed particle in bowl coordinates (3), in source coordinates (3), an observable geometric bowl-settled indicator and an observable geometric source-contained indicator.

A shared registered MLP maps every particle token to 64 dimensions. Mean and maximum pooling over the twelve tokens make the learned particle summary invariant to particle ordering. A second registered MLP embeds each robot frame. The two robot embeddings and the mean/max particle embeddings from both frames form the 384-dimensional diffusion condition. Measured joints and velocities retain reachability and motion information; bowl/target and source/TCP relations retain destination and grasp dependencies. No future state, contact truth, unprovided action history or evaluation-fitted statistic enters deployment inputs.

Translations and joints are scaled by declared physical ranges in the adapter. Rotations use 6D matrix columns. These deterministic scales are identical in training and deployment. The framework separately fits the learned-action representation normalizer only from valid authorized training labels and respects padding masks.

## Learned Cartesian label

For supervised slot `i`, define:

- `B_i`: observed world bowl pose before the action;
- `C_i`: observed world source-container pose before the action;
- `X_i`: observed world TCP pose before the action;
- `U_i`: demonstrated commanded world TCP pose from `panda_kinematics.commanded_tcp_poses`;
- `G_i = inverse(C_i) @ X_i`: observed container-to-TCP transform.

Under the rigid-grasp relation, the desired source pose corresponding to the demonstrated TCP command is

`C*_i = U_i @ inverse(G_i)`.

The represented target is

`A_i = inverse(B_i) @ C*_i`,

encoded as bowl-frame source position in metres (3), source orientation as 6D rotation columns (6), and the demonstrated native gripper scalar (1). This is a 10D learned action. It is an absolute source pose in the bowl frame, unlike h01's bowl-frame TCP pose, h02's fresh-TCP local residual and h03's source-local increment.

For a rigid transformation applied to the observed bowl and its associated carry/pour/righting geometry, composing the same `A_i` with the displaced bowl moves the desired source trajectory by construction. This structural invariance covers transport to the bowl, bowl-centered pouring, righting and the bowl-relative component of initial departure. It does not guarantee behavior when the bowl moves independently of the fixed return target.

## Decode and task-space conversion

At every executed slot, the decoder uses fresh observations, not a stale chunk-start pose. With current bowl `B`, source `C` and TCP `X`, it computes `G = inverse(C) @ X`, projects the sampled 6D orientation by safe Gram-Schmidt, reconstructs `C* = B @ A`, and forms the world TCP target

`X* = C* @ G`.

For an unmodified valid training label at the same observation this algebra reconstructs `U_i` before numerical IK. The decoder calls only `panda_kinematics.solve_ik(X*, current_qpos, robot)` to obtain the seven absolute Panda joint targets and appends the represented gripper scalar. Diagnostics expose convergence, position and rotation residuals, iterations, lower/upper limit contacts, maximum joint change, posture error, world source/TCP targets and the observed grasp transform. An unconverged solution is reported rather than hidden. The common executor retains its native gripper sign rule.

The orientation projection is nonlearned and only maps arbitrary sampled 6D values to a valid rotation. It is not a task-completing script and adds no success override.

## Model, objective and temporal contract

`BowlSourcePosePolicy` contains both input MLPs and the unchanged published `DiffusionBackbone(384, 10, training_config)`. Thus all learned encoders and the denoiser are registered, optimized, included in EMA and saved in the checkpoint. The masked epsilon objective remains the total training objective. `prior_loss` is a differentiable zero because the prior is supplied by the input symmetry and label/decoder geometry; there is no artificial auxiliary target.

The fixed temporal and numerical recipe is unchanged: two causal frames, horizon 16 representing t-1 through t+14, execution slots 1-8 starting at current t, 20 Hz, DDPM epsilon prediction with 100 train/inference denoising steps, batch 128, 20,000 updates, seed 0 and final EMA selection.

## Expected benefit, assumptions and limitations

The falsifiable expected benefits are: (1) sampled terminal orientation targets refer directly to the source and remain absolute in the bowl frame, reducing local integration drift during righting; (2) bowl-relative carry, pour and righting targets translate and rotate with a reachable displaced bowl; (3) a different but rigid observed source-to-TCP transform changes the required TCP target while retaining the same desired source pose; and (4) particle-index permutations do not change the pooled particle condition. None of these benefits has been measured for h04 before submission.

The policy assumes reliable bowl, source, TCP and particle poses; an approximately rigid grasp; sufficient wall clearance; and IK-reachable targets. Slip, compliance, grasp loss and pose noise violate fresh-grasp reattachment. Mean/max pooling cannot infer hidden contact. The successful nominal demonstrations do not establish spill, collision, lost-grasp or unseen-phase recovery. Absolute anchoring does not remove diffusion multimodality over the long segment. Large bowl displacement may be unreachable, and independent bowl versus return-target changes are outside demonstrated support.

## Handoff

Entry requires a securely held, lifted or lifting source and stable source-to-TCP relation. Exit keeps the unchanged task evidence: at least ten particles settled in the bowl, the source itself approximately upright and held, and outbound motion begun. `return_place_release` is nominally ready near action 1680. The complete action-supervised `[1460,1740)` overlap permits takeover during late righting or later departure, but particle emptiness alone is not successor readiness.
