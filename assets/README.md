# Frozen experiment assets

`manifest.json` records each published file's SHA-256, size, semantic path, and
optional download bundle. A null bundle means the file is included in Git.
The two asset bundles contain the original trained checkpoints, demonstrations,
Cut segments and their referenced image frames. The Exp2 bundle also contains
complete per-episode result and invocation evidence. Both bundles are published on
[Hugging Face](https://huggingface.co/datasets/Oscattt/APPL-assets/tree/b840e01bc5b355f4f40842db6fe01002e2e6ac60);
their manifest URLs pin the verified `v0.1.0` asset commit. See
[download and installation](../docs/assets.md) for the commands and direct links.

The layout is independent of the original research workstation:

- `exp1/demonstrations/<task>/train20/`: the original twenty-trajectory support
  pool; the frozen support manifests specify the N=2/5/10/20 subsets.
- `exp1/evaluation/{dev,test}/<task>/`: all frozen evaluation manifests and
  exact reset snapshots. These small snapshots are included in Git.
- `exp2/demonstrations/<task>/`: the twelve original full demonstrations.
- `exp2/datasets/<task>/<skill>/`: unchanged Cut action/observation segments,
  associated images, and a small dataset manifest.
- `exp2/cases/{motion,task,composition}/<task>/<case>/`: the paper's frozen
  initial states, case definitions, and goal contracts.
- `exp2/tasks/<task>/`: deployment context, normalization, prompts, schemas,
  task specifications, and the published copies of executed configurations.

`source_sha256` identifies the original artifact; `sha256` identifies the
published artifact. JSON copies with workstation paths are explicitly relocated,
and credential locations and transport headers are omitted. Their transformation
is recorded; tensor values, observations, actions, case selection, success flags,
and evaluation thresholds are not changed. Hashes embedded in original scientific
receipts remain historical hashes unless a manifest explicitly records its new
structural hash. Use the release manifest for published-file integrity checks.

Original research logs and private path-to-path export receipts are not part of
the public repository. Historical runs excluded from the paper have not been
merged into the released result tables.
