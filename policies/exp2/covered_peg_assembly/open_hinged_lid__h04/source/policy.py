import torch
from experiments.exp2.six_tasks.flexible.public import DiffusionBackbone


class StationaryHingeDiffusionPolicy(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.backbone = DiffusionBackbone(80, 11, spec['training'])

    def forward(self, noisy_action, timesteps, model_inputs):
        features = model_inputs['features']
        condition = features.reshape(features.shape[0], 80)
        return self.backbone(noisy_action, timesteps, condition)


def build_model(spec):
    return StationaryHingeDiffusionPolicy(spec)


def compute_loss(model, batch, spec):
    prediction = model(batch['noisy_action'], batch['timesteps'], batch['model_inputs'])
    squared = (prediction - batch['noise']) ** 2
    dimension_weights = torch.ones((1, 1, 11), dtype=squared.dtype, device=squared.device)
    dimension_weights[:, :, 9:11] = 2.0
    mask = batch['mask']
    denominator = mask.sum().clamp_min(1.0) * dimension_weights.sum()
    diffusion_loss = (squared * mask * dimension_weights).sum() / denominator
    prior_loss = diffusion_loss * 0.0
    return {
        'loss': diffusion_loss + prior_loss,
        'diffusion_loss': diffusion_loss,
        'prior_loss': prior_loss,
    }
