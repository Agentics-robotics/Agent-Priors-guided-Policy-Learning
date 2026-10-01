"""Developer-owned generic numerical capabilities, not a scientific prior."""
import torch
from appl.vendor.diffusion_policy.conditional_unet1d import ConditionalUnet1D
from appl.public import epsilon_loss, normalize_observation


class DiffusionBackbone(torch.nn.Module):
    def __init__(self, condition_dimension, action_dimension, config):
        super().__init__()
        self.net = ConditionalUnet1D(input_dim=action_dimension, global_cond_dim=condition_dimension,
            diffusion_step_embed_dim=config['timestep_embed_dim'], down_dims=config['down_dims'],
            kernel_size=config['kernel_size'], n_groups=config['groups'], cond_predict_scale=True)

    def forward(self, sample, timestep, condition):
        return self.net(sample, timestep, global_cond=condition)


def normalize_representation(value, spec):
    n = spec['representation_normalizer']
    center = torch.as_tensor(n['center'], dtype=value.dtype, device=value.device)
    scale = torch.as_tensor(n['scale'], dtype=value.dtype, device=value.device)
    return (value - center) / scale


def denormalize_representation(value, spec):
    n = spec['representation_normalizer']
    return value * torch.as_tensor(n['scale'], dtype=value.dtype, device=value.device) + torch.as_tensor(n['center'], dtype=value.dtype, device=value.device)
