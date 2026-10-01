# Construct a new Exp2 policy library

`appl.construction` implements the paper's complete APPL construction sequence:
Cut, three independently checked priors per skill, fixed-budget training,
in-distribution verification, one additional prior, and final usage guidance.
The published policies and results remain under `policies/exp2` and `results/exp2`.
Every construction output is explicitly a **new run**, including when the same
prompts, training demonstrations and numerical recipe are used.

Install the Exp2 asset bundle, including demonstration images, and use the locked
Exp2 Pixi environment. API stages read `OPENAI_API_KEY`, with optional
`OPENAI_BASE_URL` and `OPENAI_MODEL`; reasoning effort is fixed to `xhigh`.
The GPU is selected explicitly at initialization. Generated policy checks,
training and inference run in credential-free isolated subprocesses.

```bash
pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.construction init \
  --task drawer_exchange --output runs/exp2/new_drawer --gpu 0
pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.construction inspect \
  --run runs/exp2/new_drawer
pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.construction cut \
  --run runs/exp2/new_drawer --dry-run
```

Initialization and dry runs make no API requests and start no GPU jobs. The Cut
dry run checks the actual synchronized observations, actions and image files.
Remove `--dry-run` to execute the Cut agent. The same entry supports
`buffer_swap`, `constrained_retrieve_store`, `covered_peg_assembly` and
`granular_pour_return`.

The released scientific prompts are copied unchanged. The new-run protocol
context preregisters the paper's skill IDs and exact exit rules before Cut;
the API chooses segment boundaries, overlaps, evidence and handoffs. Cut rejects
renamed or missing skills and requires every skill to have support in all twelve
training demonstrations. A different skill vocabulary requires a separately
specified verifier and is outside this reproduction protocol.

After Cut, run the following sequence **for every skill in
`cut/manifest.json`**. This example uses the drawer's `open_drawer` skill.

```bash
pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.construction design \
  --run runs/exp2/new_drawer --skill open_drawer

for prior in 01 02 03; do
  pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.construction train \
    --run runs/exp2/new_drawer --policy "open_drawer__h${prior}"
  pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.construction verify \
    --run runs/exp2/new_drawer --policy "open_drawer__h${prior}"
done

pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.construction refine \
  --run runs/exp2/new_drawer --skill open_drawer
pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.construction train \
  --run runs/exp2/new_drawer --policy open_drawer__h04
pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.construction verify \
  --run runs/exp2/new_drawer --policy open_drawer__h04
pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.construction finalize \
  --run runs/exp2/new_drawer --skill open_drawer
```

All execution stages accept `--dry-run`. Design and refinement are real API
sessions; `check_policy` performs the original bounded two-update GPU interface
check, and final training uses the task's original recipe and last EMA checkpoint.
All three originals must be submitted before their formal training starts.
Refinement can edit only `h04`; the three originals stay immutable. Verified
kinematics and demonstrated-posture tables are read-only capabilities in every
candidate package. New code imports the numerical helpers from
`appl.policy.public`; historical helper names in preserved documents refer to
that same capability.

Verification replays each training demonstration to the policy's segment entry,
runs for 1.5 times the segment length with the demonstrated argument schedule,
and measures the fixed exit condition at the final state. Refinement and final
report tools receive only all twelve current-run ID outcomes and authorized
policy documents. They cannot read benchmark outcomes, case folders, other
sessions or a shell. Failed or partial physical episodes are retained; the
pipeline does not silently restart them. Transport retry handling is confined
to the standard API client.

`library/<skill>/catalogue.json` contains four callable contracts, priors,
handoff documents, measured support, ID success rates and API-written final
guidance. Its receipt freezes the catalogue and final report. The Python entry
`appl.construction.pipeline.deployment_library(run)` validates every skill and
returns the configuration and policy dictionary consumed by
`appl.benchmark.deployment.DeploymentTools`. The evaluation CLI's
`--construction-run` option uses this library on the separately frozen cases;
those evaluation outputs never become construction feedback.

```bash
pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl evaluate \
  --task drawer_exchange --method appl --suite motion --case 24092400 --gpu 0 \
  --construction-run runs/exp2/new_drawer --output runs/exp2/new_drawer_motion_24092400
```

Only `--method appl` accepts a constructed library. Use a separately named
episode directory; the selected case, evaluator and execution budget stay fixed,
and the result identifies the new library and its hashes.

Run directories freeze their configuration, input documents, source framework,
environment lock, submissions and verification receipts. Initialize a new run
after changing the framework or relocating a run. Paths to installed assets are
resolved from `--asset-root` at initialization; no original workstation layout is
required. Historical API outputs are stochastic artifacts, so a new API run is
not a claim to regenerate the published source hashes or scores.
