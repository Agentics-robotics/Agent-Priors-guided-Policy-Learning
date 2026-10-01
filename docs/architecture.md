# Architecture

Exp1 and Exp2 share the research principle but retain independent simulator
environments and execution contracts.

`relative_dp` provides the diffusion backbone and numerical utilities.
`experiment_interfaces` defines environment and policy contracts.
`experiment1` implements candidate design tools, isolated training, MetaWorld
evaluation and frozen development-set selection. `appl` provides the final
long-horizon benchmark's task scenes, policy engine, isolation and composition.
`appl_release` verifies assets and reproduces paper tables without importing
simulators or machine-learning libraries.

Framework code belongs under `src/`. Construction-agent submissions belong under
`policies/`, with their contracts and source identity. Experiment definitions are
under `experiments/`; results and data identities are kept separately. Every
runtime uses explicit output directories, so a new run cannot overwrite a
published scientific artifact.

The release presents one final experiment implementation. Original source hashes
remain in provenance manifests; machine scheduling history, private provider
configuration, and operational restart scripts are not runtime dependencies.

The asset manifest is the bridge from a clean checkout to the exact trained
policies and demonstrations. Each environment installs the same public Python
package while supplying its own locked numerical and simulator dependencies.
