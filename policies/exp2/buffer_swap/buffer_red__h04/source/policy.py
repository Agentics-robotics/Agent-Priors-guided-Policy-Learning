import torch
from appl.public import epsilon_loss
from experiments.exp2.sol_flexible.public import DiffusionBackbone


class SemanticAnchorPhasePolicy(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(132, 256),
            torch.nn.Mish(),
            torch.nn.Linear(256, 256),
            torch.nn.Mish())
        self.phase_head = torch.nn.Linear(256, 4)
        self.backbone = DiffusionBackbone(256, 7, spec['training'])

    def encode_condition(self, model_inputs):
        features = model_inputs['features']
        return self.encoder(features.reshape(features.shape[0], 132))

    def forward(self, noisy_action, timesteps, model_inputs):
        condition = self.encode_condition(model_inputs)
        return self.backbone(noisy_action, timesteps, condition)


def build_model(spec):
    return SemanticAnchorPhasePolicy(spec)


def compute_loss(model, batch, spec):
    condition = model.encode_condition(batch['model_inputs'])
    prediction = model.backbone(batch['noisy_action'], batch['timesteps'], condition)
    diffusion = epsilon_loss(prediction, batch['noise'], batch['mask'])
    logits = model.phase_head(condition)
    target = batch['auxiliary']['phase_index'].long().reshape(-1)
    phase_mask = batch['auxiliary']['phase_mask'].reshape(-1).to(logits.dtype)
    phase_values = torch.nn.functional.cross_entropy(logits, target, reduction='none')
    phase_loss = (phase_values * phase_mask).sum() / phase_mask.sum().clamp_min(1.0)
    prior = 0.1 * phase_loss
    return {'loss': diffusion + prior,
            'diffusion_loss': diffusion,
            'prior_loss': prior}
