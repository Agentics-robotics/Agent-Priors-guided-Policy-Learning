import torch
from appl.public import epsilon_loss
from experiments.exp2.sol_flexible.public import DiffusionBackbone


class IncrementPhasePolicy(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(76, 128),
            torch.nn.Mish(),
            torch.nn.Linear(128, 128),
            torch.nn.Mish()
        )
        self.backbone = DiffusionBackbone(128, 7, spec['training'])
        self.phase_head = torch.nn.Linear(128, 5)

    def condition(self, model_inputs):
        flat = model_inputs['features'].reshape(model_inputs['features'].shape[0], -1)
        return self.encoder(flat)

    def forward(self, noisy_action, timesteps, model_inputs):
        return self.backbone(noisy_action, timesteps, self.condition(model_inputs))

    def phase_logits(self, model_inputs):
        return self.phase_head(self.condition(model_inputs))


def build_model(spec):
    return IncrementPhasePolicy(spec)


def compute_loss(model, batch, spec):
    prediction = model(batch['noisy_action'], batch['timesteps'], batch['model_inputs'])
    diffusion = epsilon_loss(prediction, batch['noise'], batch['mask'])
    logits = model.phase_logits(batch['model_inputs'])
    target = batch['auxiliary']['phase'].long().reshape(-1)
    valid = batch['auxiliary']['phase_mask'].to(logits.dtype).reshape(-1)
    per_item = torch.nn.functional.cross_entropy(logits, target, reduction='none')
    phase_loss = (per_item * valid).sum() / valid.sum().clamp(min=1.0)
    prior = 0.05 * phase_loss
    return {'loss': diffusion + prior, 'diffusion_loss': diffusion, 'prior_loss': prior}
