# Prior: chunk-start source-frame Cartesian targets

## Decision and targeted failure

This independent policy targets displacement of the manipulated source container beyond the narrow demonstrated initial range. Every 16-slot Cartesian command is encoded relative to the observed source pose at chunk construction. The source pose is refreshed on each 8-step (0.4 s) replan. This makes handle approach and grasp geometry translation/rotation equivariant by construction; during carry it also makes each short motion source anchored, while the observed source-to-bowl transform tells the network where transport must end.

The evidence is observational rather than a recovery claim. Demo12300 indices 120-160 place the TCP at the handle, action 148 closes, and index 180 is lifted. Demo12303 closes earlier (action 136, closed observation 140) and is higher at index 180. Initial source centers across the assigned demonstrations span only about x=-0.443 to -0.458 m and y=-0.202 to -0.214 m, motivating removal of absolute source-position memorization.

## Inputs and arguments

The required arguments `source_object="container"` and `destination_object="bowl"` select real observed pose fields. Each of two causal frames contributes 51 deterministic features: source-relative TCP pose, source-relative bowl pose, world source/TCP positions, measured qpos/qvel, particle centroid and spread, and the declared local grasp site. Relative positions and robot channels use fixed physical scales; no evaluation-fitted input statistics or future state enters inference. At an episode start the framework duplicates the initial observation. History remains available across policy switches.

World source/TCP positions and measured joints deliberately remain in the condition: source-frame actions do not remove Panda reachability, joint-limit, collision, or gravity dependencies. Particle summaries retain gross payload/co-motion context, but are not contact truth.

## Actions, conversion, and training

The learned dimension is 10: source-frame position xyz in metres, source-frame rotation in continuous 6D, and the native gripper scalar. `encode_targets` obtains demonstrated commanded TCP poses with the verified URDF FK and computes `inverse(chunk_source) @ commanded_tcp`. `decode_action` robustly projects 6D to SO(3), composes the frozen source frame, and calls `panda_kinematics.solve_ik` from freshly measured joints. It reports convergence, position/rotation residuals, iterations, joint-limit contacts, and maximum joint change. It never replaces an unconverged result. Gripper sign semantics are unchanged.

The standard DiffusionBackbone is unchanged because the required invariant is fully implemented by input/action coordinates. The model flattens the two 51-channel frames into a 102-dimensional global condition. Masked epsilon diffusion is the only loss; `prior_loss` is a differentiable zero. The host fits the representation normalizer only to these valid 10D training labels. Temporal settings remain H2/16/8 at 20 Hz and formal training remains 20,000 updates with the declared recipe.

All twelve [0,360) occurrences are action-supervised. In every occurrence [180,360) remains substantial overlap with `controlled_pour_and_right`, covering lift, carry, staging, and the first small tilt.

## Expected benefit and limitations

Expected benefit: when the source pose changes but the handle layout and reachability remain valid, pregrasp target coordinates stay the same and the causal source-to-bowl condition changes predictably. This is falsifiable in the frozen evaluation but is not established by the successful demonstrations.

The prior cannot certify grasp contact, recover an unseen miss/slip, avoid an unobserved obstacle, or make unreachable placements feasible. A large mid-chunk source slip invalidates the frozen frame until replanning. Source-only displacement changes the required carry path to the fixed bowl, so transport still depends on learned source-to-bowl conditioning. Large rotations and geometry changes are outside evidence. Prefer the TCP-delta policy for local intermediate TCP errors and the phase-aware bowl policy for ambiguous already-grasped carry states.
