# Drawer-destination anchored Cartesian targets

## Decision and targeted failure

This policy targets changed drawer destination and confined final placement. Commanded TCP positions are learned relative to the causally observed `blue_goal`, in a translated but world-aligned frame. Changing that goal translates decoded final approach, lowering, release and retreat targets exactly, instead of requiring the network to extrapolate memorized world positions.

Observed evidence supports the relationship: in demo1000 the held block moves from the source-side high carry at 870 to above the drawer by 930, lowers through 950-970, is inside/released by 990, and the TCP retreats through 1010-1050. Demo1001 shows the same relations with different timing. Expected robustness to a changed goal is a construction benefit, not demonstrated collision recovery.

## Inputs and argument

`destination_field` is required and restricted to `blue_goal`. At each replan its causal world-metre point is frozen in chunk context.

Each causal frame has 44 fixed-scale values: qpos/qvel, TCP position relative to goal and world orientation, blue position relative to goal and world orientation, red-to-pad residual, absolute goal position, and drawer position/velocity. Absolute goal and robot state preserve reachability/configuration dependence; red residual preserves the simultaneous predecessor objective. Episode-start history duplication is framework-owned, while history otherwise crosses switches.

## Cartesian targets and IK

The 10-D action is `[TCP_position-blue_goal (3 m), world_rotation_6d (6), gripper (1)]`. Demonstrated target poses come from FK of the native joint commands, not lagging measured TCP. Prediction slots remain t-1..t+14 and only slots 1..8 execute.

Decoding adds the frozen destination, projects 6-D orientation to SO(3), and calls `panda_kinematics.solve_ik` from freshly measured joints. Diagnostics expose convergence, position/rotation residual, iterations, limit contacts and joint change. Position subtraction/addition and valid 6-D label conversion roundtrip the task-space target; joint nonuniqueness is permitted.

## Model, normalization and supervision

The standard unchanged `DiffusionBackbone` receives flattened 2x44 features and diffuses 10-D actions. No auxiliary head or artificial prior loss is used. The action diffusion objective remains the masked epsilon loss. Inputs use declared fixed physical scaling; the framework fits representation normalization only to valid transformed labels. Arbitrary spatial augmentation is intentionally omitted because it could make targets unreachable or inconsistent with drawer geometry.

All twelve assigned [600,stop) ranges are bound. This preserves the full [600,850) red-transfer overlap (3,000 supervised actions) before the transport/insertion specialization.

## Applicability and complementary failure

Prefer this policy after blue/TCP co-motion with closed fingers suggests a stable carry and drawer alignment dominates. It may still receive control anywhere in the overlap. For a far-displaced free source, the complementary blue-frame model provides source equivariance by construction. For uncertain intermediate timing or small tracking offsets, the fresh-TCP incremental model is complementary.

Success remains exactly drawer open AND red on pad AND blue inside in one observation.

## Limitations

A point goal omits drawer walls, lip and collision-free routing. A translated destination can be unreachable. Rotation remains world aligned because no drawer orientation argument or demonstrated rotated-drawer support exists. Source displacement is conditioned but not invariant. Finger closure and co-motion are cues, not contact truth; failed insertions and recoveries were not demonstrated.
