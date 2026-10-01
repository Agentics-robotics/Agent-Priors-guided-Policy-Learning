import torch
from appl.public import epsilon_loss
from experiments.exp2.sol_flexible.public import DiffusionBackbone


class RedFrameDiffusionPolicy(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        condition_dimension = 2 * 44 + 3
        self.backbone = DiffusionBackbone(condition_dimension, 10, spec['training'])

    def forward(self, noisy_action, timesteps, model_inputs):
        features = model_inputs['features'].reshape(model_inputs['features'].shape[0], -1)
        intent = model_inputs['intent'].reshape(model_inputs['intent'].shape[0], -1)
        condition = torch.cat([features, intent], dim=-1)
        return self.backbone(noisy_action, timesteps, condition)


def build_model(spec):
    return RedFrameDiffusionPolicy(spec)


def compute_loss(model, batch, spec):
    prediction = model(batch['noisy_action'], batch['timesteps'], batch['model_inputs'])
    diffusion_loss = epsilon_loss(prediction, batch['noise'], batch['mask'])
    prior_loss = diffusion_loss * 0.0
    loss = diffusion_loss + prior_loss
    return {'loss': loss, 'diffusion_loss': diffusion_loss, 'prior_loss': prior_loss}
