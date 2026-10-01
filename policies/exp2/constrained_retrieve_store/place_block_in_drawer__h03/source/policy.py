import torch
from appl.public import epsilon_loss
from experiments.exp2.six_tasks.flexible.public import DiffusionBackbone


class PhaseAwareLocalDiffusionPolicy(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.condition_encoder = torch.nn.Sequential(
            torch.nn.Linear(96, 256),
            torch.nn.Mish(),
            torch.nn.Linear(256, 128),
            torch.nn.Mish()
        )
        self.phase_head = torch.nn.Linear(128, 4)
        self.backbone = DiffusionBackbone(128, 7, spec['training'])

    def encode_condition(self, model_inputs):
        flat = model_inputs['features'].reshape(model_inputs['features'].shape[0], -1)
        return self.condition_encoder(flat)

    def phase_logits(self, model_inputs):
        return self.phase_head(self.encode_condition(model_inputs))

    def forward(self, noisy_action, timesteps, model_inputs):
        condition = self.encode_condition(model_inputs)
        return self.backbone(noisy_action, timesteps, condition)


def build_model(spec):
    return PhaseAwareLocalDiffusionPolicy(spec)


def compute_loss(model, batch, spec):
    prediction = model(batch['noisy_action'], batch['timesteps'], batch['model_inputs'])
    diffusion_loss = epsilon_loss(prediction, batch['noise'], batch['mask'])
    logits = model.phase_logits(batch['model_inputs'])
    target = batch['auxiliary']['phase_target'].reshape(-1).long()
    phase_mask = batch['auxiliary']['phase_mask'].reshape(-1).to(logits.dtype)
    per_sample = torch.nn.functional.cross_entropy(logits, target, reduction='none')
    prior_loss = (per_sample * phase_mask).sum() / phase_mask.sum().clamp_min(1.0)
    loss = diffusion_loss + 0.1 * prior_loss
    return {'loss': loss, 'diffusion_loss': diffusion_loss, 'prior_loss': prior_loss}
