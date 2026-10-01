# Third-party components

This project's original framework and released policy code are distributed under
MIT. Existing third-party notices remain applicable to their components.

| Component | Origin / retained evidence | Distribution |
| --- | --- | --- |
| Diffusion Policy U-Net modules | `src/relative_dp/vendor/diffusion_policy/PROVENANCE.md` and `LICENSE`; upstream commit recorded there | Vendored source; upstream MIT notice retained |
| Diffusion Policy modules used by Exp2 | `src/appl/vendor/diffusion_policy/PROVENANCE.md` and `LICENSE` | Vendored source; upstream MIT notice retained |
| MetaWorld | Git revision pinned in the root Pixi manifest and lock | Installed separately from upstream |
| ManiSkill, SAPIEN, MuJoCo | Versions and package sources pinned in the matching Pixi lock | Installed separately from upstream |
| PyTorch, Diffusers and numerical dependencies | Exact distributions and hashes in Pixi locks | Installed separately from upstream |

Simulator assets fetched by dependencies remain subject to their upstream
licenses. The project license does not replace those terms. No physical-robot
recordings, SAM model assets, or Agent+VLA weights are included in this release.
Source and dataset identity manifests document the released study's provenance.
