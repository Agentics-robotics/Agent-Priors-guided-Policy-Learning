# APPL policy implementation interface

This contract describes the policy hooks and numerical training recipe. Framework
and numerical helpers are developer-owned. Scientific policy code, adapters,
labels and semantic documents are authored by Runtime API. Read
TRAINING_DETAILS.md for the corresponding learning and handoff requirements.

## Files and metadata

Required flat files: policy.py, adapters.py, policy_contract.json,
training_bindings.json, pipeline.json, PRIOR.md, HANDOFF.json. Optional flat Python
helpers are allowed. panda_kinematics.py is pre-installed in every package as a
read-only, hash-checked developer capability; import it as panda_kinematics and do
not write, replace or shadow it. No IO/network/reflection or simulator access in generated
code. Imports: torch, numpy, math, appl.public,
experiments.exp2.sol_flexible.public and own flat modules, including panda_kinematics. Explicit absolute imports.
No __future__, dataclasses, typing or private reflection. All code runs only after
Landlock/seccomp. Read public.py for the numerical backbone and helpers.

policy_contract.json has EXACT keys:
- version: "appl.flexible.sol.v1"
- policy_id, skill_id: assignment identities
- call_args_schema: closed JSON Schema object; additionalProperties:false at each
  object; all fields required. Optional values use required nullable types, not defaults.
  Empty object is valid if the prior requires no caller arguments. Prefer meaningful
  typed arguments supported by the binding labels; no arbitrary formula/code/actuator input.
- parameter_semantics: nonempty text including units, frames and binding meaning
- observation_dependencies: list of names actually used from public state fields
- temporal: {"history_steps":2,"prediction_horizon":16,"prediction_start_offset":-1,
  "execution_steps":8,"control_hz":20}, exactly
- input_shapes: nested nonempty dictionary whose leaves are positive integer shape
  lists, excluding batch. Example {"features":[2,47]} is an example, not a required layout.
- action_dimension: any positive integer, bounded by actual memory/parameter budgets
- output_semantics: representation, units, frames, reference time, encoding/decoding
  and any nonlearned computation/projection. No generic recurrent state is exposed.
- conversion_check: {"mode":"task_space", "tolerance":number in (0,.05],
  "description":text}. All policies in this study act through Cartesian targets,
  so the mode is task_space: the framework independently compares URDF FK TCP
  position distance (metres), rotation angle (radians), and gripper scalar error,
  taking their maximum. Targets are the demonstrated native joint COMMAND's FK,
  not the lagging measured TCP; nonunique IK is allowed. The solver's default
  tolerances are 1e-4 m and 1e-3 rad. Explain any projection.
- cut_relation: how implementation preserves the assigned skill, segments and overlap

training_bindings.json is a list of EXACT objects:
{trajectory_id,start,stop,context_start,call_args,rationale}.
Supervision uses [start,stop); context_start MUST equal max(0,start-1). Every assigned
skill action must be covered. Overlap with other bindings is allowed when rationale
explains duplicate supervision and semantic consistency. Call arguments match the
contract schema. Each proposed prior records invocation_contract as JSON text
with call_args_schema and parameter_semantics, and training_bindings as JSON
text. The implementation publishes actual JSON files for its contract and bindings.

pipeline.json EXACT keys: policy_id, skill_id, heuristic_index, prior_summary,
applicable_conditions, termination_guidance, limitations, config (object).
HANDOFF.json EXACT keys: policy_id, entry_conditions, exit_conditions, overlap_role,
successor_readiness, failure_signatures, continuation_guidance, limitations, evidence.
All except evidence are nonempty strings. evidence is a nonempty list of
{trajectory_id,indices}, with inspected supporting indices inside assigned slices.
PRIOR.md explains mechanism, learned modules, gradient paths, inputs/output,
converters, normalization, support, parameter meaning, handoff and limitations.

## Adapter hooks (all unbatched CPU preprocessing)

`build_inputs(causal_history, call_args, public_context, spec)` returns
{"model_inputs":nested tensor dictionary matching input_shapes,
 "chunk_context":finite JSON object/array}.
causal_history is a list of TWO structured state dictionaries, oldest first.
Only initial episode observation is duplicated. All values are raw physical units;
qpos/qvel, tcp_pose, red_pose, blue_pose, drawer_position/velocity, red_goal/blue_goal.
Poses use world metres and wxyz quaternions. Goals include the actual observed
goal for tabletop tasks and public calibrated drawer goals. No images/contact truth.
Adapters contain deterministic numeric preprocessing, never trainable encoders;
trainable encoders must be inside the registered model so gradients/EMA include them.
The builder runs at each new sampled chunk. Chunk context is its causal reference
and remains fixed for those outputs; fresh state is supplied to each decode.

`encode_targets(demonstration_window, call_args, chunk_context, public_context, spec)`
returns {"actions":unnormalized learned actions [16,D], "auxiliary":nested tensor
dictionary, possibly {}}. Window keys: native_action [16,8], action_observations
(16 raw state dicts BEFORE actions), future_observations (16 AFTER states), mask
[16,1], history (two causal states), source_index, trajectory_id. Only this label
hook receives futures. The framework owns mask; auxiliaries must carry their own
masks and all losses must respect invalid padding. Actions outside this binding
are edge-padded and masked, even if another binding covers them.

`decode_action(represented_slot, current_observation, chunk_context, call_args,
public_context, spec)` returns {"native_action":finite list [8],
"diagnostics":finite JSON}. represented_slot is a list [D], AFTER framework inverse
representation normalization. First seven native components are absolute Panda
arm joint targets in radians; eighth is gripper scalar. Common executor clips to
native limits, then maps >=0 to +1 open, <0 to -1 close. This same binary rule is
used by frozen baselines. Do not add task-completing scripts in the decoder.
The framework's task-space check uses public FK.

## Cartesian action interface and panda_kinematics

Every policy in this study acts through Cartesian end-effector targets.
decode_action converts the learned slot into a world-frame TCP pose target and
obtains the seven native joint targets only through
panda_kinematics.solve_ik(target_pose_4x4, current_observation['qpos'],
public_context['robot']); it then appends the native gripper scalar. Report the
solver's converged flag, residuals and limit contacts in diagnostics; do not
substitute another controller or silently alter an unconverged result. The learned
representation and its frame are your choice, for example a world pose, a pose
relative to a target object or destination, or a displacement from the current TCP;
only the final conversion to native joints uses the shared solver.

panda_kinematics.py is a verified developer capability built from the exact public
URDF chain; read it with read_public. It provides forward_kinematics (optionally with
the 6x7 world Jacobian), tcp_pose, pose_vector, commanded_tcp_poses,
pose_from_observation, pose_matrix, invert_pose, relative_pose, quaternion_matrix and
matrix_quaternion (wxyz), rotation_vector_matrix and matrix_rotation_vector,
rotation_6d and rotation_from_6d, arm_limits, pose_error and solve_ik. solve_ik is
damped least-squares IK from the measured joints with joint limits, adaptive damping
and nullspace continuity toward the starting configuration; it returns joints,
converged, position_error, rotation_error, iterations and limit contacts. Its
acceptance checks covered FK against recorded simulator TCP poses, the Jacobian,
randomized round trips and closed-loop simulator replay of every training
demonstration through IK of its commanded TCP poses.

Demonstrated actions are available in Cartesian form. read_steps reports each
action's commanded_tcp_pose, the world TCP pose [x,y,z,qw,qx,qy,qz] reached by FK of
its seven joint targets. In encode_targets,
panda_kinematics.commanded_tcp_poses(demonstration_window['native_action'],
public_context['robot']) returns the same [16,7] values for a window. The measured
tcp_pose is present in every observation.

## Model, batches, normalization and numerical recipe

policy.py: build_model(spec) -> registered torch.nn.Module, 1..64M trainable parameters.
forward(noisy_action [B,16,D], timesteps scalar or [B], model_inputs) -> epsilon
[B,16,D]. model_inputs has the same nested structure with batch dimension prepended.
compute_loss(model,batch,spec) -> scalar tensors loss, diffusion_loss, prior_loss.
Total and diffusion losses must be differentiable and finite. No future labels may
be inputs to deployment forward. Pure architecture priors may return zero prior_loss.

batch: model_inputs, represented_action [B,16,D], encoded_action [B,16,D],
mask [B,16,1], auxiliary (your stacked target dictionary), noisy_action, noise,
timesteps [B], alpha_bar [B]. encoded_action is the actual DDPM clean target.
Use epsilon_loss(prediction,noise,mask) or equivalent masked mean.

spec includes training, contract, candidate_config, skill_id, public_context,
normalizer (shared original native 47-channel normalization), and AFTER label
preparation representation_normalizer. build_inputs/encode_targets must not require
representation_normalizer: it is fitted afterwards from valid training labels.
The engine scales each learned action dimension using its valid training-label
midpoint and half range (constant dimension scale 1); no evaluation data is fitted.
Do not apply the old absolute-native action scales to a different representation.
Input feature normalization can use public physical units or the shared training
normalizer; fitted/trainable transformations must be stored in the model/spec.

20,000 final updates, seed 0, batch128, DDPM100/100 cosine epsilon with clip_sample,
AdamW1e-4, weight decay1e-6, gradient clipping1, EMA.999, cosineLR with500warmup.
Default backbone widths128/256/512,timestep128,kernel5,groups8. Use the published
flexible DiffusionBackbone(condition_dimension, action_dimension, spec['training'])
as a mature building block. Prefer retaining this backbone; explain any necessary
architecture changes.
The host samples16 slots and executes slots1..8 (actions t..t+7). No temporal tuning.

Checks run full supervised-window adapter construction, causal input repeatability,
boundary/interior target roundtrips, two fresh optimizer updates, finite gradient
and weight change, DDPM sampling, decoded sample finiteness, and EMA reload equality.
No task success/performance feedback is available during implementation. Five bounded
checks per candidate; failed versions and GPU costs are retained. check_policy must
succeed before immutable submit_policy(expected_hash). Final checkpoint is last EMA.
No additional rollout-based prior revision is part of this one-round protocol.
