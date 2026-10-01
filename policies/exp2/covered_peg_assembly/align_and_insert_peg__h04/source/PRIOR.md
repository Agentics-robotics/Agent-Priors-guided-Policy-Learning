# Rigid-grasp object-goal retargeting prior

## Decision and target failure

`align_and_insert_peg__h04` targets a moved square-hole fixture and a modestly changed but still rigid peg-to-TCP grasp offset at handoff. It learns the desired **peg** pose in the observed hole frame, then converts that object goal to a Cartesian TCP target through the freshly observed grasp transform. This covers the complete assigned responsibility: final correction, controlled descent, +hole-x insertion, and closed-gripper terminal stabilization.

All three original policies passed 12/12 available validation runs. Those runs began from exactly replayed demonstration entries at index 1380 and contain no out-of-distribution intervention, so they do not identify a nominal defect to retune. The h04 benefit is instead structural and unconfirmed: destination displacement is removed from the learned action coordinates, and a changed rigid grasp offset is compensated algebraically rather than treated as noise.

## Evidence and support

In demo12200, indices 1380-1460 show the aligned peg descending from z about 0.424 m to the insertion line near z=0.103 m. Indices 1480-1520 show +x insertion, and 1540-1554 show a stable terminal hold. Demo12205 index 1380 has peg y about 0.1595 m and the largest observed incoming lateral/orientation correction before converging to the same nominal sequence. Demo12208 reaches near completion earlier and retains a stable tail through index 1538.

All twelve full assigned ranges are bound. In each trajectory, [1380,1500) remains action-supervised overlap with `reorient_and_stage_peg`: 120 actions or 6.0 seconds per demonstration and 1,440 actions total. The evidence is nominal successful insertion with a fixed hole world pose; it does not show grasp-offset interventions, slips, drops, collisions, jams, retractions, or retries.

## Causal inputs and retained dependencies

The typed call is `{"destination_object":"hole","held_object":"peg"}`. The arguments resolve observed `hole_pose` and `peg_pose`; they never supply poses, formulas, code, or actuator commands. `build_inputs` uses exactly two causal observations, oldest first. Episode-initial duplication is owned by the framework, and causal history is retained across policy switches.

Each frame has 72 channels:

- hole-relative peg pose, 9 values;
- hole-relative TCP pose, 9 values;
- peg-relative TCP pose, 9 values, explicitly exposing the grasp transform;
- hole-relative task target pose, 9 values;
- absolute world hole pose, 9 values;
- absolute world TCP pose, 9 values;
- measured qpos, 9 values; and
- measured qvel, 9 values.

Every pose is translation plus continuous 6D rotation. Task-relative translations are divided by 0.5 m, peg-to-TCP translation by 0.1 m, world translation by 1 m, qpos by 3 rad, and qvel by 2.5 rad/s. Relative channels express task and grasp geometry. Absolute hole/TCP pose and measured joint state deliberately retain workspace, configuration, velocity, redundancy, and reachability dependence; relative coordinates do not make collision or IK feasibility invariant.

## Action labels and Cartesian conversion

The learned 10D action is `[hole-frame desired virtual peg xyz metres, hole-frame desired virtual peg rotation 6D, native gripper scalar]`.

For training slot k, `encode_targets` obtains `world_T_commanded_tcp[k]` by verified FK of the demonstrated native joint command. From the same slot's pre-action observation it computes

`tcp_T_peg[k] = inverse(world_T_observed_tcp[k]) @ world_T_observed_peg[k]`,

then forms the virtual rigidly transported object target

`world_T_commanded_peg[k] = world_T_commanded_tcp[k] @ tcp_T_peg[k]`

and encodes

`hole_T_commanded_peg[k] = inverse(world_T_hole[k]) @ world_T_commanded_peg[k]`.

This is a deterministic transformation of an authorized demonstrated action and its causal pre-action geometry. It is not a future observation or contact label. Framework masks remain authoritative for padded slots, and the representation normalizer is fitted only from valid transformed training labels.

At each physical execution step, `decode_action` projects the sampled 6D rotation to SO(3), resolves fresh `world_T_hole`, `world_T_peg`, and `world_T_tcp`, and computes

`world_T_desired_peg = world_T_hole @ hole_T_desired_peg`,

`peg_T_tcp = inverse(world_T_peg) @ world_T_tcp`,

`world_T_desired_tcp = world_T_desired_peg @ peg_T_tcp`.

It then calls only `panda_kinematics.solve_ik(world_T_desired_tcp, measured_qpos, robot)` and appends the learned native gripper scalar. IK convergence, position and rotation residuals, iterations, lower/upper limit contacts, maximum joint change, posture status, and grasp-transform magnitudes are reported. Unconverged output is not hidden.

Using the same observation in encoding and decoding cancels `tcp_T_peg @ peg_T_tcp`, exactly reconstructing the demonstrated commanded TCP pose before numerical IK. At deployment, replacing both the hole pose and the rigid grasp transform with fresh observations gives the intended equivariance. This conversion does not script a task phase or completion motion.

## Model and training

The standard public Diffusion Policy backbone is unchanged. It receives the flattened 144D two-frame condition and predicts epsilon for 16 slots of 10D actions. There are no auxiliary heads or learned preprocessing modules. `prior_loss` is differentiable zero; the masked action diffusion objective is the full learned objective.

The fixed recipe is unchanged: two observation frames, prediction slots t-1 through t+14, execution slots 1 through 8 beginning at current t, 20 Hz, DDPM 100 training and inference steps, batch 128, seed 0, EMA 0.999, and 20,000 updates. All actions in every binding, including overlap and stable tails, are supervised.

## Calling, handoff, and expected use

Prefer h04 when the peg is visibly retained and moves rigidly with the TCP, the hole may be displaced, or the predecessor may have produced a modestly changed rigid grasp offset. The peg should be clear of the box, near axis alignment, and on a plausibly reachable and collision-free route. h01 remains a simpler hole-frame TCP policy when nominal grasp geometry is expected; h02 is the local fresh-TCP residual option for modest tracking errors; h03 is the phase-aware absolute-world option near the demonstrated workspace.

Exit only under the unchanged success definition: peg-head x in `[-0.145,-0.09]` m, peg y/z within 0.009 m of the hole center, axis alignment at least 0.985, and a stable terminal inserted pose. There is no learned successor. Retain closed-gripper stabilization and do not use rigid-transform estimates, stage appearance, or IK convergence as completion.

## Assumptions, expected failures, and limitations

The peg and TCP observations must accurately describe a rigid grasp. Slip, object loss, or pose-estimation inconsistency invalidates the conversion and should appear as a changing peg-to-TCP transform or peg/TCP separation. Large fixture displacement or extreme grasp offset may make the converted TCP target unreachable, contact-unsafe, or colliding even though the geometric label remains valid. Other expected failures are persistent y/z or angular error, descent while offset, jamming during +hole-x motion, IK non-convergence or limit contact, and stalled progress before the head interval.

There is no contact truth and no demonstrated recovery for drops, slips, collision, jamming, retraction, regrasp, or retry. Object-goal equivariance does not guarantee Panda reachability, clearance, contact feasibility, or dynamics. The policy may persist or stall rather than recover from an unseen jam. These are expected structural limitations, not observed out-of-distribution results.

Provenance: frozen prior plan SHA-256 `a1623ff68007a22943dc6dc46160a40a23395696eeffcad38211932e9b13ecc8`; cut plan SHA-256 `2ae5a2895d6b7907c7d8b047a04c9ef42fbea4ce995788b0fd08dd7ccdcaea88`; dataset SHA-256 `9249213e5d1915185d1f90acd07fb9cb6482642ea69297a0fb798eadd92a13ec`.
