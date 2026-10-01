import torch
from appl.public import epsilon_loss
from experiments.exp2.six_tasks.flexible.public import DiffusionBackbone


class RelationalDiffusionPolicy(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        shape = spec['contract']['input_shapes']['features']
        condition_dimension = int(shape[0]) * int(shape[1])
        action_dimension = int(spec['contract']['action_dimension'])
        self.backbone = DiffusionBackbone(condition_dimension, action_dimension, spec['training'])

    def forward(self, noisy_action, timesteps, model_inputs):
        features = model_inputs['features']
        condition = features.reshape(features.shape[0], -1)
        return self.backbone(noisy_action, timesteps, condition)


def build_model(spec):
    return RelationalDiffusionPolicy(spec)


def compute_loss(model, batch, spec):
    prediction = model(batch['noisy_action'], batch['timesteps'], batch['model_inputs'])
    diffusion_loss = epsilon_loss(prediction, batch['noise'], batch['mask'])
    prior_loss = prediction.sum() * 0.0
    loss = diffusion_loss + prior_loss
    return {'loss': loss, 'diffusion_loss': diffusion_loss, 'prior_loss': prior_loss}
