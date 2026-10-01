# Support-phase regularized world policy

## Decision and targeted failure

This independent policy targets phase ambiguity near setdown, release, and retreat. The inspected runs share similar geometry but differ in release timing: demo12300 is still closed at 1840 and opens by 1850, whereas demo12303 and demo12308 command open at 1840. Blurred causal phase features can therefore produce premature opening, delayed opening, or retreat confusion.

## Main Cartesian policy

The diffused 10D action is an absolute world TCP position in metres, world rotation in continuous 6D form, and the demonstrated gripper scalar. `encode_targets` obtains commanded TCP poses from public FK. `decode_action` projects sampled 6D rotation to SO(3), forms the world target, and invokes `panda_kinematics.solve_ik` from fresh measured joints. This world representation intentionally retains the nominal robot/support geometry and differs from the destination and fresh-TCP priors.

The two causal frames contain world source, TCP, and destination poses; destination-relative source and source-relative TCP poses; measured arm/finger configuration and velocity; and explicit source height, target x/y errors, and upright cosine. The two rows expose finite-difference motion without future state. Fixed physical scales are deterministic. The framework fits action normalization only on valid 10D labels.

## Auxiliary mechanism and gradient path

A registered trainable MLP maps the flattened 134D causal input to a 256D condition. The unchanged standard `DiffusionBackbone` consumes that condition. A registered auxiliary head consumes the same latent and predicts the open/closed command for all 16 horizon slots. `encode_targets` supplies `open_target = (native_gripper >= 0)` and a copied framework-valid mask. Binary cross entropy is masked and weighted by 0.05; total loss is masked epsilon diffusion loss plus this auxiliary term. Thus the condition encoder receives gradients from both action diffusion and phase prediction, while the auxiliary head is included in optimization, EMA, and saved state. Future labels are training-only. At inference the action diffusion remains causal, and auxiliary logits neither gate the gripper nor define success.

## Evidence, responsibility, and overlap

All twelve complete frozen `[1460, stop)` bindings are used. This includes every action in the 280-step `[1460,1740)` overlap with `controlled_pour_and_right`, all descent/open/retreat actions, and stable terminal labels. In demo12300 the source descends through indices 1780-1840, opens at 1850, remains stationary, and the TCP retreats through 1880-1920. Demo12303 and demo12308 open at 1840 and retreat earlier. Contact is inferred from low source z and stationarity; no contact truth exists.

## Calling and selection

Required arguments are symbolic `source_object="container"` and `destination="target_pose"`. Prefer this model near the nominal demonstrated return support when release phase is the dominant uncertainty. Prefer the destination-frame model for a displaced goal and the fresh-TCP model for a tilted or unusual intermediate handoff. Terminate only from physical observations of upright support in bounds, open fingers, and TCP clearance.

## Assumptions and limitations

The nominal world geometry remains relevant, object poses are accurate, grasp persists to setdown, and IK is feasible. World actions may memorize the demonstrated destination and are not displacement invariant. The auxiliary predicts demonstrated future command sign, not contact, support, or success. There are no failed placement, premature-release recovery, regrasp, collision, or corrective repouring examples. Reported IK diagnostics expose infeasibility but do not recover it.
