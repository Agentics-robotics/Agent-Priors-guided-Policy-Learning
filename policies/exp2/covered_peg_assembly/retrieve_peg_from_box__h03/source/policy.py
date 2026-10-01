import torch
from appl.public import epsilon_loss
from experiments.exp2.six_tasks.flexible.public import DiffusionBackbone


class PhaseObservableDiffusionPolicy(torch.nn.Module):
    def __init__(self, condition_dimension, action_dimension, config):
        super().__init__()
        self.backbone = DiffusionBackbone(condition_dimension, action_dimension, config)

    def forward(self, noisy_action, timesteps, model_inputs):
        features = model_inputs['features']
        condition = features.reshape(features.shape[0], -1)
        return self.backbone(noisy_action, timesteps, condition)


def build_model(spec):
    shape = spec['contract']['input_shapes']['features']
    condition_dimension = int(shape[0] * shape[1])
    return PhaseObservableDiffusionPolicy(condition_dimension, int(spec['contract']['action_dimension']), spec['training'])


def compute_loss(model, batch, spec):
    prediction = model(batch['noisy_action'], batch['timesteps'], batch['model_inputs'])
    diffusion = epsilon_loss(prediction, batch['noise'], batch['mask'])
    prior = diffusion * 0.0
    return {'loss': diffusion + prior, 'diffusion_loss': diffusion, 'prior_loss': prior}
