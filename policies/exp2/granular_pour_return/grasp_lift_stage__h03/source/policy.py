import torch
import torch.nn.functional as functional
from appl.public import epsilon_loss
from experiments.exp2.six_tasks.flexible.public import DiffusionBackbone


class PhaseAwareBowlDiffusionPolicy(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.condition_encoder = torch.nn.Sequential(
            torch.nn.Linear(122, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 128),
            torch.nn.SiLU(),
        )
        self.phase_head = torch.nn.Linear(128, 3)
        self.backbone = DiffusionBackbone(128, 10, spec['training'])

    def encode_condition(self, model_inputs):
        features = model_inputs['features']
        return self.condition_encoder(features.reshape(features.shape[0], -1))

    def forward(self, noisy_action, timesteps, model_inputs):
        condition = self.encode_condition(model_inputs)
        return self.backbone(noisy_action, timesteps, condition)

    def phase_logits(self, model_inputs):
        return self.phase_head(self.encode_condition(model_inputs))


def build_model(spec):
    return PhaseAwareBowlDiffusionPolicy(spec)


def compute_loss(model, batch, spec):
    prediction = model(batch['noisy_action'], batch['timesteps'], batch['model_inputs'])
    diffusion_loss = epsilon_loss(prediction, batch['noise'], batch['mask'])
    logits = model.phase_logits(batch['model_inputs'])
    labels = batch['auxiliary']['phase'].long().reshape(-1)
    phase_mask = batch['auxiliary']['phase_mask'].to(logits.dtype).reshape(-1)
    per_sample = functional.cross_entropy(logits, labels, reduction='none')
    phase_loss = (per_sample * phase_mask).sum() / phase_mask.sum().clamp_min(1.0)
    prior_loss = 0.1 * phase_loss
    return {'loss': diffusion_loss + prior_loss, 'diffusion_loss': diffusion_loss, 'prior_loss': prior_loss}
