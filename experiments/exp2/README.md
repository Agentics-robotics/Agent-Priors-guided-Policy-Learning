# APPL long-horizon benchmark

The public implementation is `src/appl`: one policy engine, one task registry,
and one evaluator for all five tasks. Frozen API-authored implementations live in
`policies/exp2`; case definitions and training inputs live in `assets/exp2`.
Large demonstration and checkpoint files are distributed through the asset manifest.

Use the locked environment in `environments/exp2` from the repository root:

```bash
pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl list
pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl check
pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl cases --task drawer_exchange --suite motion
pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl train --task drawer_exchange --policy open_drawer__h01 --gpu 0 --output outputs/exp2/open_drawer
pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl evaluate --task drawer_exchange --method appl --suite motion --case 24092400 --gpu 0 --output outputs/exp2/drawer-full
```

`train` retrains a released frozen design using its recorded examples, representation,
DDPM recipe, and final EMA selection. It does not call the API to search for a new
prior. `evaluate` runs a fresh simulator episode, checks the paired reset hash, and
preserves results in a new output directory. Existing output directories are refused.
Neither command executes as part of installation or `check`.
Published configuration, case, geometry, prompt, normalization, and demonstration
inputs are checked against their released hashes before use. Editing a frozen
input makes the corresponding command fail rather than silently change the benchmark.

For new API-authored libraries, follow the [construction guide](construction.md):
Cut, three-prior design and training, ID verification, fourth-prior refinement,
and final interface publication. Once every skill is finalized, the same evaluator
can load that new library with `--construction-run` (APPL only):

```bash
pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl evaluate \
  --task drawer_exchange --method appl --suite motion --case 24092400 --gpu 0 \
  --construction-run outputs/exp2/new_drawer --output outputs/exp2/new_drawer_motion
```

The case and executor remain fixed. New output records identify the construction
run and catalogue hashes and explicitly mark the library as newly generated.

For API-controlled methods, set `OPENAI_API_KEY`; `--api-config` accepts explicit
standard Responses settings. The default model and reasoning match the recorded
experiment. A different provider or model produces a new reproduction, not the
original stochastic API decisions. Inference/training workers do not receive credentials.

| Method identifier | Policy variants per skill | Information exposed to controller |
|---|---:|---|
| `appl` | 4 | Prior descriptions, measured validation, final guidance |
| `without_verification` | 3 | Original prior descriptions |
| `without_prior_information` | 4 | Anonymized interfaces and measured validation |
| `without_interface_information` | 4 | Anonymized minimal callable interfaces |
| `rule4` | 4 | Fixed ordered rule controller; no API |
| `dp`, `single_prior` | One whole-task policy | Direct execution; no API |

Suites are `motion`, `task`, and `composition`; `task` means continuation from
intermediate states or early prefix goals. `full`, `rule`, and `repositioned` remain
aliases for `appl`, `rule4`, and `task`, with canonical IDs in new result records.
The released manifests
determine which method/case outcomes exist; running an additional combination is
a new evaluation. The paper reports **first attainment with executor-assisted
stopping**: every successful reported high-level task/composition episode returned
after one satisfied step and the agent then finished (Appendix B.6). The nominal
300-step outer hold therefore did not govern those reported successes. The public
runtime preserves the executed code: each invocation interrupts on first attainment,
while its outer loop ends on explicit finish, the nominal hold, or 5,000 steps.
The rule baseline ends on first case-goal attainment. New records retain both
`first_success_step` and the final predicate so this distinction stays auditable.

`python -m appl infer` loads one frozen policy in an isolated worker. Its input JSON
contains `history` (two causal observation dictionaries) and `call_args` satisfying
the policy's `policy_contract.json`. Omitting `--gpu` permits CPU inference checks;
the published evaluation used CUDA. Model weights must be installed first.

`python -m appl.policy.probe --task drawer_exchange --policy open_drawer__h01
--output outputs/exp2/batch-probe` validates the real training input path, builds
one causal training batch, and checks the action conversion in a restricted CPU
process. It performs no optimizer updates, simulation steps, or API calls and
requires the demonstration assets. See `validation.json` for the release checks.

The migration mapping in `source_map.json` records retained source hashes and
published framework hashes. Scientific policy files preserve their original bytes;
two legacy numeric capability import names resolve to the public numeric module at
load time. No historical supervisor, scheduler, private proxy, or run directory is
needed by the public evaluator.
