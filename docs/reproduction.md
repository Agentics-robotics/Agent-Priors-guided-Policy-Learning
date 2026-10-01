# Reproduction

## Recompute published measurements

```bash
pixi run --locked tables
```

Table 1 is recomputed from hidden-test episodes and the already frozen candidate
selections. Table 2 is recomputed from 560 episodes, validating method/suite
denominators and paired initial-state hashes. No test outcome selects a policy.
Generated CSV/JSON/Markdown files can be compared with `results/tables/`.

## Run released policies

Follow the [asset installation guide](assets.md) to download and verify the
data/checkpoint bundle, then use the experiment guide: [Exp1](../experiments/experiment1/README.md) or
[Exp2](../experiments/exp2/README.md). Choose the device and a fresh output
directory explicitly. Exp1 policy inference and the non-agent Exp2 baselines
need no model API. APPL's fresh runtime composition requires a configured
Responses provider. Recorded actions are evidence, not substitutes for a fresh
agent evaluation.

## Retrain frozen designs or design new candidates

The experiment CLI exposes retraining for the released policy implementations.
Such a run has its own code, data, configuration and output identity. New
construction-agent sessions additionally require the configured model and API
budget. Model service changes, stochastic inference and hardware differences
can affect new outcomes; the published result files are never overwritten.

## Validation layers

- Dependency-free release tests check table accounting, path relocation, bundle
  hashes, safe extraction and credential-free transport configuration.
- The root environment runs Exp1 numerical, interface and isolation checks.
- The Exp2 environment runs its policy/runtime/registry checks.
- Full training and simulator experiments are explicit, separate operations.

Tests and examples must not silently start paid API calls or formal experiments.
