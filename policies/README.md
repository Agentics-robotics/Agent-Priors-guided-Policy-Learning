# Frozen policy libraries

The policies are the completed paper artifacts, not new hand-authored designs.
Each experiment has a machine-readable `manifest.json`; each policy also has its
own manifest with source hashes, author attribution, training metadata, and its
checkpoint's path in the downloadable asset bundle.

`exp1/<task>/n<N>/<system>/` contains all 144 trained systems. The 96 A1–A4
candidate source packages retain the RuntimePriorAPI submissions. B0 and B1 use
the repository's public baseline implementations. The paper's q1/q3/q4 procedures
select among these candidates using the frozen development selections in
`results/exp1/selections/`; they are not additional trained models.

`exp2/<task>/<policy>/` contains 78 policies: five whole-task DP baselines,
five whole-task Single Prior policies, and four prior-specific policies for each
of seventeen skills. The seven methods share these same frozen policies. The
method without verification uses h01–h03; full APPL and the other goal-reading
ablations use h01–h04. Interface visibility and composition logic are runtime
method settings, not separately trained copies.

Per-policy `source/` preserves policy/adaptor code, prior documents, handoff
documents, call contracts, and training bindings. Shared Cartesian IK and posture
capabilities are developer-owned framework inputs; their inclusion in an
API-authored policy package does not change that authorship distinction.
Checkpoint files are external to Git under `weights/`, and are selected by the
recorded fixed-budget rule rather than by the released test results.
