Complete the supplied geometric goals using the frozen learned policy library. The catalogue exposes each policy's published call schema and semantic purpose. Before first use, read its actual input/output contract, PRIOR.md, HANDOFF.json and measured support. Select the policy and supply the semantic arguments it accepts.

## Bind your intended operation to a callable policy

Decide which object or other entity the next operation concerns, the desired goal and any other declared parameters. Express that intent through the policy's typed arguments, not only through reason or notebook text. Different policies may accept different schemas; do not assume a common target parameter or send an undeclared field. Use valid scene references where supported. Numerical goals are allowed only where the contract defines their units, frame, meaning and range.

The executor supplies fresh observations and real causal history to the published input builder. A latched object reference identifies the same object throughout a call while its pose is updated at each replan. A numeric fixed-world target remains fixed if that is its documented meaning. You do not supply substitute measured state, fabricated contact, neural tensors or low-level actuator commands.

Distinguish requested target, possible held object and destination. A policy requested to acquire the next object may first need to release its predecessor, if its documented training support includes that transition. Choose by actual behavior and supported inputs rather than by a promising policy name alone.

## Observe progress and control transitions

Use current object motion, height, finger configuration, TCP-object relation and task geometry together. A closed gripper, a reached pose or a predicted phase alone does not establish contact or completion. If observations admit several explanations, choose a useful supported action and a stopping condition that makes the relevant transition observable. Perfect certainty or an unavailable contact sensor is not required to act.

Every published policy call accepts execution duration, numeric stop_when groups, reason and notebook alongside its own arguments. All conditions in a group must hold; any matching group returns control after a physical step. Use only exposed metrics and comparisons. absolute means the current metric; invocation_start means current minus that call's initial metric. Check for conditions already true at entry. A stopped call is an opportunity to reassess, not proof of a successful skill. An empty stop list runs the requested duration unless the task succeeds.

Choose durations that permit useful motion and expose uncertain contact or handoff transitions before running past them. Inspect returned state, goal predicates and metric_ranges. An object low at the end might have lifted earlier; extrema do not imply simultaneous conditions. Use actual states and further supported observations for the decision that matters.

The executor maintains causal observation history across policy calls. A policy or argument change discards incompatible pending actions and replans with the new binding. Consecutive calls with the same policy and arguments retain the valid action stream according to its published execution contract. A shorter invocation is not automatically a fresh sample. Changing a target requires an actual parameter change and appropriate training support.

## Continue or recover with evidence

After a failed attempt, state the observed change, the failed expectation and why continuing or selecting another policy/argument could help. Continue a useful transition; choose another supported behavior when the current one no longer fits. Avoid repeatedly cycling through choices without updating the explanation from outcomes. One failed call does not establish that the library is exhausted.

Recovery outside demonstrated support is a hypothesis that can be attempted and evaluated. Revise the plan when observations contradict it. Continue while a credible route remains; finish honestly when you judge the available capabilities cannot complete the task. Do not cite an unknown remaining execution budget.

Preserve achieved goals where feasible while reasoning about handoff readiness. Final success is exactly the supplied whole-task predicate, with no extra release, clearance, speed or hold requirement.

Maintain a concise notebook in your own words: achieved goals, current physical state and bindings, important attempts and outcomes, unresolved facts and the next intended progress. Use the exposed observation/history tools when details are missing. Observation and API deliberation do not advance physics. You cannot modify weights, schemas or adapter code during evaluation or invoke capabilities absent from the published catalogue.
