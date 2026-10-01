import torch
from experiments.exp2.six_tasks.flexible.public import DiffusionBackbone, epsilon_loss


class Policy(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.net = DiffusionBackbone(2 * len(spec['normalizer']['mean']), 10, spec['training'])

    def forward(self, noisy_action, timestep, model_inputs):
        return self.net(noisy_action, timestep, model_inputs['features'].flatten(1))


def build_model(spec):
    return Policy(spec)


def compute_loss(model, batch, spec):
    prediction = model(batch['noisy_action'], batch['timesteps'], batch['model_inputs'])
    loss = epsilon_loss(prediction, batch['noise'], batch['mask'])
    return dict(loss=loss, diffusion_loss=loss, prior_loss=loss * 0.0)
