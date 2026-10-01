# Diffusion Policy modules

These three modules are from the official [Diffusion Policy repository](https://github.com/real-stanford/diffusion_policy), commit `5ba07ac6661db573af695b419a7947ecb704690f`:

- `diffusion_policy/model/diffusion/conditional_unet1d.py`
- `diffusion_policy/model/diffusion/conv1d_components.py`
- `diffusion_policy/model/diffusion/positional_embedding.py`

Only the two intra-package imports in `conditional_unet1d.py` were changed to
`relative_dp.vendor.diffusion_policy`. Architecture and operations are unchanged.
The upstream MIT license is retained in `LICENSE`. No visual, robotics hardware,
Hydra, or original Conda environment dependencies are required.
