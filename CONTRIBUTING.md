# Contributing

Use Python 3.11 and the locked Pixi environment for the experiment you change.
Keep Exp1 and Exp2 simulator dependencies separate. Run the corresponding tests
and the publication audit before proposing a change.

Policy implementations under `policies/` are scientific artifacts authored by the
construction agent. Changes to these implementations, evaluation cases, success
conditions, prompts, or training recipes define a new experiment and must carry
new identities. Framework refactors must preserve the original artifact hashes
and report the new source hashes separately.

Include the motivation, affected experiment, validation commands, and any change
to scientific behavior in a pull request. Do not include credentials, environment
directories, machine-specific paths, training logs, or large binary artifacts.

New API-backed runs require explicit model configuration. Offline unit tests
must never make paid API requests or start training.
