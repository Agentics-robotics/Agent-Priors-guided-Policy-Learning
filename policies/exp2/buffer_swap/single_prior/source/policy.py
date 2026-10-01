import torch
from experiments.exp2.sol_flexible.public import DiffusionBackbone
from appl.public import epsilon_loss


class FullTaskRelativePolicy(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.condition_encoder = torch.nn.Sequential(
            torch.nn.Linear(144, 256),
            torch.nn.Mish(),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, 256),
            torch.nn.Mish(),
        )
        self.backbone = DiffusionBackbone(256, 14, spec['training'])

    def forward(self, noisy_action, timesteps, model_inputs):
        features = model_inputs['features']
        condition = self.condition_encoder(features.reshape(features.shape[0], 144))
        return self.backbone(noisy_action, timesteps, condition)


def build_model(spec):
    return FullTaskRelativePolicy(spec)


def compute_loss(model, batch, spec):
    prediction = model(batch['noisy_action'], batch['timesteps'], batch['model_inputs'])
    diffusion = epsilon_loss(prediction, batch['noise'], batch['mask'])
    prior = prediction.sum() * 0.0
    return {'loss': diffusion, 'diffusion_loss': diffusion, 'prior_loss': prior}
