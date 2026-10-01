Generalization context: object and tool positions, and intermediate states reached during execution, may differ from the demonstrations. Design priors around relationships that should transfer while retaining relevant robot configuration, reachability and contact dependencies. Keep input representations, learned actions and action conversions consistent. Ground expected benefits in training evidence, state the assumptions and limitations, and preserve valid labels and physical feasibility in any augmentation.

Learn manipulation skills from the authorized demonstrations and make them useful under changes in object positions and intermediate execution states. Complete only the stage assigned below. Use the provided task goals, observations and robot capabilities, and distinguish observed evidence from expected benefits.

The temporal setup is fixed: two causal observation frames, prediction horizon 16, execution chunk 8, control rate 20 Hz. Prediction slots represent t-1 through t+14; execution uses slots 1 through 8, beginning at current t. The framework owns observations, numerical training, physical execution and success evaluation. Keep the supplied success definition unchanged.

Different manipulation responsibilities should favor different skills and independently trained models. Require substantial action-supervised overlap between adjacent skills to improve handoff robustness. The API chooses the responsibilities and useful overlap from evidence; a successful demonstration does not establish recovery from unseen errors.

Keep explanations direct: state the decision, its supporting evidence and its purpose. Use English for submissions. Preserve submitted artifacts and their provenance. Use only the declared tools and data, with no final-test feedback for candidate design.

Design priors that help the assigned skill generalize, and implement an independent Diffusion Policy for each prior.

Read the skill dataset, task goals, entry/exit and overlap evidence, public capabilities, INTERFACE.md, TRAINING_DETAILS.md and panda_kinematics.py. Use the demonstrations to identify the main learning difficulties and likely changes at deployment.

Design a complementary set of about three priors for this skill; justify a different number within the declared budget. Complementary means that their expected failures differ, not only their implementations. First identify the skill's most likely generalization failures from the demonstrations and deployment conditions, for example displacement of the manipulated or approached object beyond the demonstrated range, entry states produced by the preceding skill, grasp or contact errors, or phase ambiguity. Then target different failures with different priors. For each prior, state the variation or failure it targets, its mechanism, the observable conditions under which the high-level agent should prefer it, and where it is expected to fail and which other prior in the set covers that case. Do not give every prior the same action representation and reference frame, and do not assign mechanisms to prior indices by a fixed template.

At least one prior must target displacement, relative to the demonstrations, of the object this skill manipulates or approaches. Make the relevant motion invariant to that displacement by construction, for example by expressing the learned Cartesian actions relative to that object's observed pose, and state which parts of the skill it covers.

All policies act through Cartesian end-effector targets. Choose the learned action representation and its reference frame, for example world, target object, destination or current TCP. decode_action converts it to a world TCP pose that the supplied panda_kinematics.solve_ik turns into native joint targets from the freshly measured joints. Demonstrated actions are available both as native joint commands and as the TCP poses those commands reach.

A prior may act through the input representation, the Cartesian action representation and its reference frame, trainable encoders or conditioning, label-preserving data transformations or auxiliary objectives; choose what the targeted generalization requires. Keep the standard DP backbone unchanged where possible, and explain the reason and scope of any necessary backbone change. Input features and typed HL arguments remain your design choices.

An auxiliary head is optional. Give it a useful prediction target, valid masks and a real gradient path to the intended learned modules. Include all learned modules in the model, optimizer and saved state. Demonstration futures may supply training labels; inference uses causal observations and declared HL arguments. Auxiliary objectives support the main action diffusion objective.

Preserve the assigned skill and substantial action-supervised overlap. Define the policy's inputs and HL arguments, bind them to the demonstrations, and implement consistent training-target and action conversions. Keep history 2, prediction 16, execution 8 and the declared numerical training budget.

For every model, publish its prior, calling contract, entry/exit conditions and handoff guidance. Check its implementation through the supplied tools. Submit the prior set and all checked policy packages before formal training; the framework then trains the independent models and evaluates the frozen library.
