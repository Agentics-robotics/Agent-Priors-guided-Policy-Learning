Learn manipulation skills from the authorized demonstrations and make them useful under changes in object positions and intermediate execution states. Complete only the stage assigned below. Use the provided task goals, observations and robot capabilities, and distinguish observed evidence from expected benefits.

The temporal setup is fixed: two causal observation frames, prediction horizon 16, execution chunk 8, control rate 20 Hz. Prediction slots represent t-1 through t+14; execution uses slots 1 through 8, beginning at current t. The framework owns observations, numerical training, physical execution and success evaluation. Keep the supplied success definition unchanged.

Different manipulation responsibilities should favor different skills and independently trained models. Require substantial action-supervised overlap between adjacent skills to improve handoff robustness. The API chooses the responsibilities and useful overlap from evidence; a successful demonstration does not establish recovery from unseen errors.

Keep explanations direct: state the decision, its supporting evidence and its purpose. Use English for submissions. Preserve submitted artifacts and their provenance. Use only the declared tools and data, with no final-test feedback for candidate design.
