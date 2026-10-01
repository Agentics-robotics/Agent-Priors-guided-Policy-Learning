Learn manipulation skills from the authorized demonstrations and make them useful under changes in object positions and intermediate execution states. Complete only the stage assigned below. Use the provided task goals, observations and robot capabilities, and distinguish observed evidence from expected benefits.

The temporal setup is fixed: two causal observation frames, prediction horizon 16, execution chunk 8, control rate 20 Hz. Prediction slots represent t-1 through t+14; execution uses slots 1 through 8, beginning at current t. The framework owns observations, numerical training, physical execution and success evaluation. Keep the supplied success definition unchanged.

Different manipulation responsibilities should favor different skills and independently trained models. Require substantial action-supervised overlap between adjacent skills to improve handoff robustness. The API chooses the responsibilities and useful overlap from evidence; a successful demonstration does not establish recovery from unseen errors.

Keep explanations direct: state the decision, its supporting evidence and its purpose. Use English for submissions. Preserve submitted artifacts and their provenance. Use only the declared tools and data, with no final-test feedback for candidate design.

Organize the demonstrations into skill datasets with clear responsibilities and robust handoffs.

Read the task goals and inspect synchronized observations, actions and available images. Identify distinct physical responsibilities and favor a separate skill for each. For every skill, explain its subgoal, contact or motion requirements, and approximate entry and exit states. Choose the number of skills from the demonstrated responsibilities.

Choose boundaries separately for each trajectory. Inspect both boundary observations for every [start, stop) segment: actions[start:stop] are the supervised actions and observations[start:stop+1] are their states. Record the object or operation being demonstrated in each occurrence.

Give adjacent skills substantial action-supervised overlap to improve robustness to handoff timing and entry-state variation. Inspect a broad transition region and retain relevant approach, contact, release or retreat. Report shared indices, supervised action counts and duration, and explain how the overlap helps the receiving skill. If the available demonstrations limit useful overlap, explain that limitation.

Describe observable entry and exit cues, successor readiness and likely handoff difficulties. Distinguish the intended object from an object still being released, and observation context from action supervision. Cite inspected examples and record any excluded actions with reasons.

Use the supplied tools to check and submit the skill datasets, segment assignments and handoff descriptions. Keep explanations concise and grounded in the demonstrations.
