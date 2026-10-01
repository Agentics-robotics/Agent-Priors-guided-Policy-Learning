# Phase-Adaptive Semantic-Anchor Diffusion Policy

## Decision and targeted generalization failure

`buffer_red__h04` targets translation of the object relevant to the current skill phase without forcing one reference frame to describe the entire long segment. Its learned TCP **position** is expressed in one chunk-frozen semantic frame:

- observed red during approach, closure, and the start of lift;
- the high-level supplied buffer during elevated-red carry, lowering, release, and early retreat;
- observed blue during the action-supervised successor transit, acquisition, and initial lift.

This makes positional commands equivariant by construction to displacement of the selected anchor. Red approach targets translate with red; carry/release targets translate with the buffer; overlap acquisition targets translate with blue. TCP orientation is deliberately represented differently: a world-axis rotation-vector residual from the chunk-start measured TCP. This retains the numerically compact 7D action used by the in-distribution-successful h02 rather than learning a full fixed-anchor orientation representation.

Prefer h04 when spatial displacement is the main concern and causal geometry clearly identifies a demonstrated phase. Prefer `buffer_red__h02` for ambiguous phase, uncertain contact, unexpectedly elevated/dropped objects, or ordinary demonstrated geometry: h02 passed 12/12 in-distribution validation runs, while no out-of-distribution result exists for any policy.

## Evidence, expected benefit, and status

In demo10100 and demo10103, commanded TCP positions approach their differently sampled red blocks at indices 40-100, lift them at 120-160, converge on the common buffer at 200-280, retreat at 300-320, then approach/close/lift their differently sampled blue blocks at 380-479. All 12 segments retain 210-230 overlap actions through the blue lift. At measured segment ends, red is near (-0.181, 0, 0.020) m and blue is elevated.

Observed validation is limited to replayed demonstration entry states and training arguments. h01 passed 5/12, h02 passed 12/12, and h03 passed 0/12 under the final red-containment rule. These facts motivate retaining h02's compact orientation residual and causal encoder while replacing a single fixed positional anchor with a phase-relevant anchor. They do not prove the mechanism or its expected displacement benefit. The falsifiable expectation is that h04 needs smaller changes in its local positional labels when red, buffer, or overlap-blue is translated, while avoiding fixed-red labels for the blue tail and fixed-buffer labels for initial red acquisition.

## Causal phase and anchor rule

The rule uses only the latest causal structured observation and call arguments and is recomputed at every sampled chunk. Red is `buffered` when its planar distance from the supplied buffer is below 0.06 m and its height is below 0.08 m. Fingers are `closed` below 0.055 m total width and `open` above 0.065 m. The four auxiliary phases and three position modes are:

0. red acquisition / initial lift, red anchor;
1. closed fingers with unbuffered red above 0.045 m, buffer anchor;
2. buffered red before successor transit, buffer anchor;
3. buffered red plus blue above 0.035 m, TCP within 0.14 m of blue, or open fingers with TCP more than 0.08 m from red, blue anchor.

The selected mode and anchor pose are frozen in `chunk_context` for all decoded outputs of that chunk and refreshed at the next replan. This prevents repeated DDPM calls from advancing phase memory. The rule has no contact truth and is guidance, not grasp proof.

## Inputs and invocation

The closed call contract requires `target_object="red"`, `successor_object="blue"`, and `buffer_pose_world=[x,y,z,qw,qx,qy,qz]` in world metres/wxyz. Training binds the buffer to `[-0.181,0,0.02,1,0,0,0]`. The object names select actual structured observation fields; the buffer enters relational features, phase selection, the transport/release action frame, and handoff interpretation.

Each of two causal frames has 66 values:

- 18 scaled Panda qpos/qvel/finger values;
- 9 world TCP position and rotation-6D values;
- 9-value full relative poses from TCP to red, blue, and buffer;
- a 9-value full pose from active semantic anchor to TCP;
- a three-value one-hot active anchor mode.

Positions are divided by 0.5 m, arm qpos by 3 rad, arm qvel by 2.5 rad/s, and fingers by 0.04 m. Rotations use 6D matrix columns. These fixed physical scales are not fitted on evaluation data. World TCP and robot channels retain posture, joint-limit, and reachability dependencies that relative coordinates cannot remove. Framework-owned history duplication is used only at episode start, and both causal frames are preserved across policy switches.

## Action targets and decoder

Each 7D slot is `[p_anchor_x,p_anchor_y,p_anchor_z,rotvec_x,rotvec_y,rotvec_z,gripper]`:

- position is metres in the chunk-frozen active semantic anchor;
- rotation vector is radians for `R_command * R_chunk_tcp^T` in world axes;
- gripper is the demonstrated native scalar, whose sign is preserved.

`encode_targets` obtains every demonstrated command TCP pose with verified FK. It applies inverse anchor rotation/translation to command position and the SO(3) logarithm to command orientation relative to chunk-start measured TCP. `decode_action` performs the exact inverse compositions to obtain a world TCP target, then calls only `panda_kinematics.solve_ik(target, fresh_measured_qpos, robot)`. It reports convergence, position/rotation residuals, iterations, joint-limit contacts, joint change, posture status, target pose, anchor and phase. It never substitutes another controller or silently changes an unconverged result. Framework representation normalization is fitted only on valid 7D labels in these coordinates.

## Learned modules and losses

A registered MLP maps the flattened 132 causal values to a 256D condition. The unchanged standard `DiffusionBackbone` predicts epsilon for 16x7 action samples. A registered four-class phase head reads the same condition. The phase target is the causal rule above; its mask uses the valid current slot and never uses future observations. Masked cross-entropy is weighted 0.1 and provides a real gradient through the shared encoder. Masked epsilon diffusion remains the primary objective. Encoder, head, backbone, optimizer state, checkpoint, and EMA are all managed together.

## Temporal scope, overlap, and handoff

History is 2, horizon is 16, and execution is slots 1-8 at 20 Hz. Prediction slots represent t-1 through t+14. Every action in every assigned segment is bound. During overlap, the buffer mode covers final red lowering/opening and early retreat; the blue mode covers transit, blue closure, and initial lift. This preserves substantial action-supervised handoff support rather than using overlap only as context.

Core successor readiness remains observed red stability in the supplied buffer with open, clear fingers/TCP. A later switch after closed fingers around blue and visible blue lift is also supervised. Preserve causal history across the switch. Neither the deterministic mode nor auxiliary phase prediction establishes release or grasp.

## Limitations and expected failure signature

Only selected-anchor TCP position is equivariant to displacement. Orientation, Panda posture, joint limits, travel length, collision geometry, table interaction, dynamics, and contact outcomes are not invariant. Large translations or rotations may be unreachable or unsupported. The deterministic thresholds can switch early after an elevated miss, retain red after a drop, or activate blue before a secure red release. Observable failures include targets moving toward the wrong semantic site, repeated gripper toggles, red entering then leaving the buffer, red remaining elevated after opening, and persistent IK nonconvergence or limit contact.

There are no images, contact truth, collision model, occupied-buffer checks, or failed-grasp/drop recovery demonstrations. Successful demonstrations do not establish recovery. The fixed full-task success definition remains `red_at_goal AND blue_at_goal`; h04 only covers red-to-buffer relocation plus its assigned blue-acquisition overlap.
