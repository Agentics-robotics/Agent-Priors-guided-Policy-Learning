# Experiment 1: skill generalization

Six MetaWorld tasks × N = 2, 5, 10, 20 demonstrations × B0/B1/A1–A4,
one seed and 144 trained models. Every model uses the same diffusion backbone,
20,000 updates, controller and evaluation. A1–A3 were submitted before development
feedback; A4 was designed from that feedback and trained from scratch. q1 is A1;
q3/q4 select by equal-weight C/E development success, then latency, parameter
count and candidate ID. Hidden-test and IID scores do not select candidates.

The scientific settings are in `protocol.json`; task observation/action contracts
and ordered support subsets are in `manifests/`. Original submitted candidate
bytes live under `policies/exp1/`; reported results and Table 1 inputs live under
`results/exp1/`. These paths are relative to the repository root. The published
protocol records the original protocol digest, but deliberately omits host
allocation and private API settings. It is not the original execution lock.
New reproduction runs freeze the released framework under their own identity.

## Install and verify

Run from the repository root using its locked Pixi environment:

```sh
pixi install --locked
pixi run python -m experiment1.release verify
```

Install the separate demonstration/reset/weight assets following `assets/README.md`,
then run `pixi run python -m experiment1.release verify --assets`. Data and weights
are hash-identified assets, not automatically downloaded when importing a module.
No API key is required for verification or use of the published models.

## Reproduce a published policy

The output must be a new directory. Commands use an explicitly selected physical
GPU and fail if inputs, dependencies or isolation capabilities are missing.

```sh
pixi run python -m experiment1.release train-policy --task drawer --n 2 --system A4 --gpu 0 --output runs/exp1-drawer-a4
pixi run python -m experiment1.release evaluate-policy --task drawer --n 2 --system A4 --gpu 0 --output runs/exp1-drawer-a4-test
```

Evaluation defaults to the published checkpoint. To evaluate a new fit, add
`--checkpoint runs/exp1-drawer-a4/training/latest.pt`. Each evaluation writes new
results; it does not overwrite the paper measurements. Linux with Landlock ABI ≥3,
libseccomp, MuJoCo and the selected NVIDIA GPU is required for isolated workers.
The outer process uses Pixi; child workers use the same Python interpreter.

## Run a new construction-agent study

Set `OPENAI_API_KEY`; optional `OPENAI_BASE_URL` and `OPENAI_MODEL` override the
standard Responses endpoint and recorded model. Exp1 always requests reasoning
`max`. Set `APPL_GPUS=0` or a comma-separated device list for your machine.
No local proxy, account token file or authors' GPU allocation is assumed.

```sh
pixi run python -m experiment1.release prepare-data --root runs/exp1-new
pixi run python -m experiment1.fitting --root runs/exp1-new
pixi run python -m experiment1.cli preflight --root runs/exp1-new
pixi run python -m experiment1.cli freeze --root runs/exp1-new
pixi run python -m experiment1.cli run --resume --root runs/exp1-new
```

`prepare-data` performs real simulator audits and renders the current task–N
evidence. Common fitting performs the fixed baseline infrastructure checks.
`preflight` includes a paid API interface call; `run` designs and trains a full
new study. Submitted candidates remain immutable, all three initial proposals
precede feedback, and the global selection freeze precedes hidden testing.
New API outputs are new scientific runs, not a byte-exact regeneration promise.

The shared `relative_dp` package contains only the backbone, normalization,
optimizer/checkpoint helpers and attributed Diffusion Policy vendor code used
by this experiment. Developer framework and baselines remain separate from the
96 API-authored submitted candidate packages.
