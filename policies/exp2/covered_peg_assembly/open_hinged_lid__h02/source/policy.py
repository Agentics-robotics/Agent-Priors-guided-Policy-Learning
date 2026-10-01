import torch
from appl.public import epsilon_loss
from experiments.exp2.six_tasks.flexible.public import DiffusionBackbone


class FreshTCPDiffusionPolicy(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.backbone = DiffusionBackbone(72, 7, spec['training'])

    def forward(self, noisy_action, timesteps, model_inputs):
        features = model_inputs['features']
        condition = features.reshape(features.shape[0], 72)
        return self.backbone(noisy_action, timesteps, condition)


def build_model(spec):
    return FreshTCPDiffusionPolicy(spec)


def compute_loss(model, batch, spec):
    prediction = model(batch['noisy_action'], batch['timesteps'], batch['model_inputs'])
    diffusion_loss = epsilon_loss(prediction, batch['noise'], batch['mask'])
    prior_loss = diffusion_loss * 0.0
    return {
        'loss': diffusion_loss + prior_loss,
        'diffusion_loss': diffusion_loss,
        'prior_loss': prior_loss,
    }
