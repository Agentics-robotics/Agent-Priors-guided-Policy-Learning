import torch
from appl.public import epsilon_loss
from experiments.exp2.sol_flexible.public import DiffusionBackbone


class DrawerFramePolicy(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        feature_shape = spec['contract']['input_shapes']['features']
        condition_dimension = int(feature_shape[0] * feature_shape[1])
        self.backbone = DiffusionBackbone(condition_dimension, spec['contract']['action_dimension'], spec['training'])

    def forward(self, noisy_action, timesteps, model_inputs):
        condition = model_inputs['features'].reshape(model_inputs['features'].shape[0], -1)
        return self.backbone(noisy_action, timesteps, condition)


def build_model(spec):
    return DrawerFramePolicy(spec)


def compute_loss(model, batch, spec):
    prediction = model(batch['noisy_action'], batch['timesteps'], batch['model_inputs'])
    diffusion_loss = epsilon_loss(prediction, batch['noise'], batch['mask'])
    prior_loss = diffusion_loss * 0.0
    return {'loss': diffusion_loss + prior_loss, 'diffusion_loss': diffusion_loss, 'prior_loss': prior_loss}
