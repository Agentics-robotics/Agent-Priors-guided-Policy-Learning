import torch
from appl.public import epsilon_loss
from experiments.exp2.six_tasks.flexible.public import DiffusionBackbone


class RelationalCartesianDiffusion(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.backbone = DiffusionBackbone(
            condition_dimension=120,
            action_dimension=10,
            config=spec['training'],
        )

    def forward(self, noisy_action, timesteps, model_inputs):
        features = model_inputs['features']
        condition = features.reshape(features.shape[0], 120)
        return self.backbone(noisy_action, timesteps, condition)


def build_model(spec):
    return RelationalCartesianDiffusion(spec)


def compute_loss(model, batch, spec):
    prediction = model(batch['noisy_action'], batch['timesteps'], batch['model_inputs'])
    diffusion_loss = epsilon_loss(prediction, batch['noise'], batch['mask'])
    prior_loss = diffusion_loss * 0.0
    loss = diffusion_loss + prior_loss
    return {
        'loss': loss,
        'diffusion_loss': diffusion_loss,
        'prior_loss': prior_loss,
    }
