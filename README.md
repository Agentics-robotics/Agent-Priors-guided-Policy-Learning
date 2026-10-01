# Agent Priors-guided Policy Learning (APPL)

APPL uses an agent-designed structural prior both to train a skill policy and to
describe its applicability to a runtime agent. This distribution contains the
paper's **Exp1** and **final five-task Exp2**, including their baselines and
ablations. Agent+VLA and real-robot software are outside this release.

## Start with the released results

Use Python 3.11 and [Pixi](https://pixi.sh) **0.80.0 or newer**. CI uses 0.80.0;
older Pixi versions may not support the checked-in v7 lockfiles. From this checkout:

```bash
pixi install --locked
pixi run --locked verify
pixi run --locked tables
pixi run --locked audit
```

`tables` recomputes Table 1 from 14,400 individual hidden-test outcomes and frozen
candidate selections, and Table 2 from 560 paired episodes. It writes CSV, JSON,
and Markdown to `results/tables/`. No model API, GPU, or checkpoint is needed.
The lightweight command can also be run with a Python 3.11 interpreter by setting
`PYTHONPATH=src` and executing `python -m appl_release tables`.

## Experiments

| Experiment | Scope | Entry |
| --- | --- | --- |
| Exp1 | Six MetaWorld tasks; 2/5/10/20 demonstrations; B0, B1, q1, q3, q4; 144 trained systems | [Exp1 guide](experiments/experiment1/README.md) |
| Exp2 | Five ManiSkill tasks; DP, SinglePrior, APPL and four ablations; motion/task/composition suites | [Exp2 guide](experiments/exp2/README.md) |

Exp2 uses the shared posture-constrained Cartesian IK, twelve demonstrations per
task, and the policy libraries used in the paper. Its full APPL result is 20/40
motion OOD, 37/40 task-level OOD, and 8/16 composition. DP and SinglePrior have no
task-goal input, so they are included only in motion OOD. See
[scope and protocol](docs/experiments.md) for the exact method definitions.

## Data and trained policies

Policy source, contracts, provenance, evaluation cases, and compact results are
versioned here. Demonstrations and 222 trained checkpoints are separate asset
bundles, totaling approximately 36 GiB of checkpoint data. Obtain the matching
`exp1`/`exp2` release assets and install them with their manifest checksums:

```bash
pixi run --locked python -m appl_release fetch --bundle exp1 --source PATH_TO_EXP1_ARCHIVE
pixi run --locked python -m appl_release fetch --bundle exp2 --source PATH_TO_EXP2_ARCHIVE
pixi run --locked python -m appl_release verify --assets
```

An HTTPS asset URL can be supplied instead of a local archive. The manifest does
not invent download endpoints: where `url` is unset, supply the actual asset file
or URL. Asset publication is separate from code checkout. See
[asset installation](docs/assets.md).

## Environments and API configuration

Exp1 uses the root Pixi environment. Exp2 uses its independent simulator lock:

```bash
pixi install --manifest-path environments/exp2/pixi.toml --locked
pixi run --manifest-path environments/exp2/pixi.toml --locked exp2 --help
```

Training and simulator evaluation target Linux x86_64. Generated-policy workers
require Linux isolation support; GPU selection is supplied by the user. The
offline result tools use only the Python standard library.

Only new API design sessions and APPL runtime composition need a model API key.
Set your own `OPENAI_API_KEY`; the default endpoint is the public Responses API.
See [API configuration](docs/api.md). No key or private provider is distributed.

## Layout and verification

- `src/`: reusable learning, simulation, isolation, and release tools.
- `experiments/`: the two public experiment configurations and protocols.
- `policies/`: frozen construction-agent implementations and contracts.
- `assets/`: versioned download and checksum manifests.
- `results/`: published outcomes and reproducible tables.
- `tests/`: offline contract tests and experiment-specific checks.

See [architecture](docs/architecture.md), [reproduction](docs/reproduction.md),
[validation evidence](docs/validation.md), [contributing](CONTRIBUTING.md),
and [security](SECURITY.md).

Code is released under [MIT](LICENSE). Third-party components retain their
respective notices; see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md).
Use [CITATION.cff](CITATION.cff) when citing this work.
