import torch
from experiments.exp2.six_tasks.flexible.public import DiffusionBackbone
from appl.public import epsilon_loss


class HybridDrawerTcpDiffusionPolicy(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.backbone = DiffusionBackbone(96, 10, spec['training'])

    def forward(self, noisy_action, timesteps, model_inputs):
        features = model_inputs['features']
        condition = features.reshape(features.shape[0], -1)
        return self.backbone(noisy_action, timesteps, condition)


def build_model(spec):
    return HybridDrawerTcpDiffusionPolicy(spec)


def compute_loss(model, batch, spec):
    prediction = model(batch['noisy_action'], batch['timesteps'], batch['model_inputs'])
    diffusion = epsilon_loss(prediction, batch['noise'], batch['mask'])
    prior = diffusion * 0.0
    return {'loss': diffusion, 'diffusion_loss': diffusion, 'prior_loss': prior}
