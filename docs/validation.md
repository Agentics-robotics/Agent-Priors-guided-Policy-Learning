# Release validation

Validation date: 2026-10-01. These checks concern the published code and assets;
they do not constitute a new training run or a rerun of the paper's experiments.

## Environments and interfaces

Both independent Pixi environments were installed from their unchanged scientific
dependency locks in a separate checkout. Both lock consistency checks passed.
CI and local validation use Pixi 0.80.0, and both manifests declare that minimum
version so an incompatible installer fails with an explicit requirement.
The root environment's test task passed 289 tests with two optional platform
checks skipped (CUDA compilation and an opt-in simulator interface check).
Upstream Matplotlib emitted 14 deprecation warnings.
The Exp2 environment's complete test task passed 49 tests, including construction,
generated-library deployment, and shared release-tool checks. The package wheel
also built successfully.

Both test tasks passed with `--no-simulator-imports`, which rejects imports of
MuJoCo, MetaWorld, ManiSkill, SAPIEN, PyOpenGL and GLFW before test collection.
CI uses this mode so local graphics drivers cannot hide an offline dependency.
Exp1 data/schema/hash checks load the simulator only when constructing or using
a physical environment. Exp2 goal metrics and rule subgoals share pure numerical
functions with the simulator, preserving the original predicates and thresholds.
Fresh-process regressions cover all 900 Exp1 reset snapshots, 96 Exp1 candidate
packages and 96 Exp2 cases. The optional MetaWorld interface check was also run
separately with the real simulator and passed (one test).

All 96 Exp1 submitted candidate packages and all 78 Exp2 policy packages passed
their source and contract checks. All 900 Exp1 evaluation reset snapshots passed
their recorded hash checks; CPU resets of all six MetaWorld tasks reproduced
the recorded initial-state hashes. All 96 Exp2 cases passed offline goal-metric
and rule-subgoal checks.

An original Exp1 candidate checkpoint and an original Exp2 policy checkpoint
were loaded in isolated CPU workers and produced finite actions. The Exp2 check
also exercised Cartesian decoding and verified IK convergence. Actual training
data preparation was checked on all five Exp2 tasks and on fourth-prior, DP and
SinglePrior examples, with zero optimizer updates. These checks cannot establish
that a full GPU episode or a newly trained model will match a reported outcome.

The construction tools were additionally exercised with all five tasks' real
training demonstrations. All 83,645 media references resolved, the initial-prior
and h04 tool contracts were checked, and the evidence and Cartesian conversion
tools ran without API or GPU jobs. See the public
[construction receipt](../experiments/exp2/construction_validation.json) and
[runtime receipt](../experiments/exp2/validation.json).

## Assets and published measurements

Both complete asset archives were installed using the public installer: 264
Exp1 members and 191,376 Exp2 members. Installation checked the archive checksum,
member allowlist, file sizes and every extracted file's checksum. The 222 trained
checkpoints and submitted policy source retain their original scientific bytes.
The final complete manifest verification checked all 195,996 files with no
missing or changed entry.

Table 1 was recomputed from all 14,400 hidden-test outcomes and the frozen
development-set selections. Table 2 was recomputed from all 560 included
episodes with denominator and paired initial-state checks. Both tables match
the supplied paper at its displayed precision, excluding Agent+VLA.

The publication audit checks text and links for personal absolute paths,
private endpoints and common credential formats. Checkpoint metadata was also
inspected during export. This is a scoped audit, not a guarantee that arbitrary
future changes cannot introduce sensitive content.

## Repeat the checks

```bash
pixi run --locked test --no-simulator-imports
pixi run --manifest-path environments/exp2/pixi.toml --locked test --no-simulator-imports
pixi run --locked verify
pixi run --locked tables
pixi run --locked audit
# After installing both asset bundles:
pixi run --locked python -m appl_release verify --assets
```

No paid API request, full training run, or formal evaluation episode was launched
as part of release validation. Fresh simulator experiments and construction-agent
sessions require explicit commands and appropriate local hardware/API access.
