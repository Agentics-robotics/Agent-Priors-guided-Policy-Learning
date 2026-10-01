# Developer ordinary Cartesian DP (no API prior)

Inputs: the complete raw 47-channel public state for the two causal frames, normalized with
the shared full-demonstration normalizer and flattened into the standard Diffusion Policy
condition. No relative features, encoders, auxiliary objectives or augmentation.

Outputs: per slot [TCP position xyz (world metres), first two rotation-matrix columns (world),
native gripper scalar]. Labels are the world TCP poses reached by FK of each demonstrated
seven-joint command. decode_action orthonormalizes the 6D rotation and converts the world
TCP pose to native joint targets with the shared panda_kinematics.solve_ik from the measured
joints; solver residuals are reported as diagnostics. This is the common Cartesian action
interface of the round, not an API-designed prior.

Support: the twelve original demonstrations only. Absolute world targets carry no designed
invariance to object displacement.
