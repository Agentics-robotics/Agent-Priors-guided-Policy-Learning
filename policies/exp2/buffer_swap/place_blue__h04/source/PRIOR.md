# Adaptive landmark-frame Cartesian diffusion

## Decision and targeted failures
`place_blue__h04` targets two likely deployment failures: translation of the blue pickup beyond its demonstrated start spread, and frame/phase conflict across a long slice that begins around buffered red, manipulates blue, and ends around red again. It jointly learns a Cartesian offset and an explicit anchor choice among the freshly observed blue position, blue-goal point, and buffered-red position.

This differs from the immutable policies. h01 fixes an entire sampled chunk to blue, h02 fixes it to blue_goal, and h03 predicts a transform from the fresh TCP. h04 instead learns which task landmark makes the current target local. Its new characteristic failure is wrong or mixed landmark selection, for which nominal-geometry h03 or the simpler single-anchor packages remain alternatives.

## Observed evidence and expected benefit
The available validation is in distribution only. h01 passed 7/12 segment exits, h02 passed 1/12, and h03 passed 12/12. h02 nevertheless ended with blue at goal in all 12 runs; eleven failures were due to the final buffered-red predicate. These facts motivate preserving explicit red-neighborhood supervision, but do not prove why any original failed.

The demonstrations visit distinct neighborhoods. In `demo10100`, red release/retreat is visible at 260-340, blue approach and closure at 420-460, blue carry and placement at 480-640, and red approach/closure/lift at 720-859. `demo10105` shows the same relations from an earlier red height: entry 250, blue closure/lift 450-470, blue release 630, and red lift 830-849. Segment-start blue x/y varies by only about 2 cm across the dataset. No shifted-object result is available.

The expected benefit follows from conversion, not measured performance. In blue-anchor mode, retaining an offset while translating observed blue translates the decoded pickup target exactly, covering blue approach, closure, lift, and early transport. Goal mode similarly translates late transport, descent, release, and retreat with the observed destination. Red mode translates predecessor and successor overlap motion with buffered red. Only translation is invariant; orientations, reachability, contact, and phase remain learned or observed dependencies.

## Calling contract and inputs
The closed call requires `target_object="blue"`, `destination_goal="blue_goal"`, `context_object="red"`, and `source_region="red_goal"`. All arguments select fields used by `adapters.py`; none is documentation-only.

`build_inputs` creates two causal frames of 55 features each:

- scaled nine-channel qpos and qvel (18), retaining arm configuration, finger width, and motion;
- TCP position relative to blue plus world TCP 6D rotation (9);
- red position relative to blue plus world red 6D rotation (9);
- blue_goal-minus-blue and red_goal-minus-blue vectors (6);
- blue world position and world 6D rotation (9);
- causal source-to-destination progress, TCP-blue distance, TCP-red distance, and blue-goal distance (4).

Distances and positions use a fixed 0.5 m scale; qpos/qvel use 3 rad or rad/s. No fitted observation statistics, future state, contact truth, action history, or image enters deployment inputs. The framework preserves the two frames across switches and may duplicate the first episode observation.

## Action labels and landmark mode
The unnormalized 13D slot is:

1. world translation offset from one selected landmark, in metres (3);
2. target world rotation as the first two rotation-matrix columns (6);
3. native gripper scalar (1);
4. one-hot weights in order `[blue, blue_goal, red]` (3).

`encode_targets` obtains each demonstrated Cartesian target from FK of the native joint command. Its mode label uses that slot's causal pre-action observation. Red is selected when TCP is at least 0.03 m more proximate to red than blue. Otherwise blue_goal is selected after clipped blue progress from red_goal to blue_goal reaches 0.72; blue is selected before that. This rule defines coordinates and labels; it is not a contact or success heuristic. Per-slot pre-action observations after the current time are available only inside supervised label construction. The deployment model predicts future offsets and weights from the two current causal frames.

The framework fits midpoint/half-range normalization on valid 13D labels only. Padded slots retain the framework mask. The three one-hot channels are learned by the same masked diffusion objective as Cartesian and gripper channels, rather than by a detached auxiliary classifier.

## Decode and physical conversion
At every executed step, `decode_action` clips predicted weights to nonnegative values and normalizes them to a simplex. A degenerate all-zero vector projects to its largest raw component. The projected weights blend the freshly observed blue, blue_goal, and red translations; the learned offset is added in world coordinates. Gram-Schmidt projects the 6D values to a rotation. The resulting world TCP pose is passed only to `panda_kinematics.solve_ik` from freshly measured qpos, and the learned native gripper scalar is appended.

For supervised one-hot labels, the same landmark subtracted by `encode_targets` is selected during decoding, exactly recovering the commanded FK target before IK, up to numerical precision. Sampled mixed weights intentionally define a continuous projection but can place a target between landmarks. Diagnostics publish weights, confidence, entropy, offset norm, IK convergence, position/rotation residuals, limit contacts, joint change, and posture settling. IK failures are not hidden.

## Model and gradients
`policy.py` retains the standard `DiffusionBackbone` with a flattened 2x55 condition and 13 action channels. No backbone modification, auxiliary head, or detached learned module is used. The masked epsilon prediction loss is the total loss; `prior_loss` is a differentiable zero for reporting. All trainable parameters therefore belong to the registered backbone and are included in optimizer, EMA, and checkpoint state. The fixed history 2, horizon 16, execution chunk 8, 20 Hz, DDPM schedule, update budget, and seed are unchanged.

## Skill coverage and handoff
All twelve complete assigned `[start, stop)` slices are bound. This preserves 210-230 predecessor-overlap actions per occurrence from final red release through blue lift and 230-240 successor-overlap actions from blue descent/release through buffered-red closure/lift. Core exit remains blue settled in blue_goal with gripper open/clear. A later supported exit is blue fixed at goal while red visibly follows the gripper upward. Closure alone does not prove grasp.

Prefer h04 when the three landmarks are observed, the active neighborhood is geometrically clear, and plausible blue/red translation or full-sequence traversal is expected. Prefer h03 for modest nominal-geometry intermediate offsets when landmark mode is ambiguous; h01 avoids selector risk for an isolated displaced blue pickup, and h02 is a simpler alternative for held blue near a shifted destination.

## Limitations and falsifiable failure signature
No available test establishes the expected displacement benefit. Object or goal rotation is not made invariant. A large translation can be unreachable or can move inputs/offsets outside support. There is no contact truth, collision constraint, or demonstrated recovery from drops, wrong-object grasps, failed closure, or disturbed red.

The prior should be considered failed when weights have no dominant anchor or choose red during intended blue pickup; the decoded target falls between landmarks; blue does not follow the gripper; release is goal-offset; red moves unexpectedly; offsets leave support; or IK residual/limit diagnostics persist. These signatures are observable, but successful demonstrations do not establish recovery from them.
