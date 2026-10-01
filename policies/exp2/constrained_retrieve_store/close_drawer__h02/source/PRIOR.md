# TCP-local residual and causal phase prior

## Decision and targeted failure

This policy targets variable handoff and intermediate execution states rather than large coherent scene displacement. Index 900 differs materially across successful demonstrations: `demo12100` still has the object near z 0.124 m and narrow fingers, `demo12102` has the object around z 0.079 m while the fingers open, and `demo12104` is already settled near z 0.063 m with nearly open fingers. A world-absolute sequence can alias release, clearance, handle approach, closure and terminal retreat under such timing variation.

Each action is instead a local SE(3) correction from the measured TCP to the commanded TCP. During execution the correction is composed with the **fresh realized TCP**, giving a feedback-like transformation when the intermediate TCP differs from the nominal future state. The expected benefit is tolerance to observed state/timing mismatch; the demonstrations do not establish recovery from arbitrary error.

## Causal inputs

`build_inputs` emits two frames of 44 values. They include target-relative TCP and object poses, TCP-object displacement, drawer position/velocity, finger width, object settling/height cues, and scaled qpos/qvel. The two frames expose physical changes without action history, future state, contact truth or recurrent memory. The network can distinguish descending/releasing, high clearance, handle descent/acquisition, moving closure and post-closure retreat from current geometry, articulated progress and finger state. Absolute robot joint state is retained for reachability and posture dependence.

No high-level arguments are used. The framework carries two causal frames across switches and duplicates only an initial episode observation.

## Action representation and conversion

For every prediction slot, `encode_targets` computes the demonstrated commanded world TCP using FK of the native joint command, then forms

`T_delta = inverse(T_measured_tcp before action) @ T_command`.

The 7 learned values are local xyz metres, a local rotation vector in radians and the native gripper scalar. This is not the moving drawer-frame absolute representation used by the displacement prior.

For an executed slot, `decode_action` resolves fresh `tcp_pose`, exponentiates the rotation vector, composes `T_world_target = T_tcp_fresh @ T_delta`, and passes that target to `panda_kinematics.solve_ik` from fresh measured qpos. The gripper value is appended unchanged; >=0 opens and <0 closes. IK convergence, task residuals, iterations, limit contacts, joint change and local correction magnitudes are reported. No unconverged result is hidden or replaced.

Labels and decoding use the same per-slot reference rule. Prediction slots are t-1 through t+14; only slots 1 through 8 execute from current t. Future represented slots are relative to the future measured states in training but are composed with actual fresh states at execution. There is no increment accumulation inside chunk context and no state mutation during denoising.

## Model and objective

The supplied `DiffusionBackbone` is unchanged. It receives the flattened 88-value two-frame condition and predicts epsilon for 7D action sequences. Masked epsilon loss remains the only objective; a differentiable zero `prior_loss` records that no auxiliary is required. The representation and causal features implement this prior without additional learned modules.

## Supervision and handoff

All actions `[900,1330)` in all 12 demonstrations are bound. The complete `[900,1160)` interval is action-supervised overlap with `place_block_in_drawer`: 260 actions (13 s) per run from release completion through handle acquisition. Inspected examples then show closure across intermediate drawer positions through about 0.0022, gripper opening near/after closure, and final TCP retreat near z 0.358 m.

Enter only while the block is over/inside the drawer or at a later valid phase. Exit guidance remains observed drawer position below 0.025 m, negligible velocity, enclosed block, open fingers and a clear high TCP. Phase features or gripper sign are not success predicates.

## Applicability and limitations

Prefer this model when local realized TCP/finger/progress state differs from nominal but global geometry remains near support. TCP-local corrections do not by themselves establish a globally fixed handle or closed destination, so accumulated offset can persist and large fixture displacement can lead to drift. Rotation-vector support is best for the demonstrated local corrections and can be difficult near pi. There is no contact truth and no demonstrated recovery after missed grasp, jam, obstruction, block interference, collision or unreachable state. The moving-drawer prior covers coherent fixture displacement; the fixed-destination auxiliary prior emphasizes closure progress and stopping.
