import torch
from appl.public import epsilon_loss
from experiments.exp2.sol_flexible.public import DiffusionBackbone


class CausalPhaseResidualPolicy(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(108, 256),
            torch.nn.Mish(),
            torch.nn.Linear(256, 256),
            torch.nn.Mish())
        self.phase_head = torch.nn.Linear(256, 4)
        self.backbone = DiffusionBackbone(256, 7, spec['training'])

    def encode_condition(self, model_inputs):
        features = model_inputs['features']
        return self.encoder(features.reshape(features.shape[0], 108))

    def phase_logits(self, model_inputs):
        return self.phase_head(self.encode_condition(model_inputs))

    def forward(self, noisy_action, timesteps, model_inputs):
        condition = self.encode_condition(model_inputs)
        return self.backbone(noisy_action, timesteps, condition)


def build_model(spec):
    return CausalPhaseResidualPolicy(spec)


def compute_loss(model, batch, spec):
    prediction = model(batch['noisy_action'], batch['timesteps'], batch['model_inputs'])
    diffusion = epsilon_loss(prediction, batch['noise'], batch['mask'])
    logits = model.phase_logits(batch['model_inputs'])
    target = batch['auxiliary']['phase_index'].long().reshape(-1)
    phase_mask = batch['auxiliary']['phase_mask'].reshape(-1).to(logits.dtype)
    values = torch.nn.functional.cross_entropy(logits, target, reduction='none')
    phase_loss = (values * phase_mask).sum() / phase_mask.sum().clamp_min(1.0)
    prior = 0.1 * phase_loss
    return {'loss': diffusion + prior, 'diffusion_loss': diffusion, 'prior_loss': prior}
