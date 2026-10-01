import torch
from appl.public import epsilon_loss
from experiments.exp2.six_tasks.flexible.public import DiffusionBackbone


class PhaseAwarePolicy(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(172, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 256),
            torch.nn.SiLU(),
        )
        self.stage_head = torch.nn.Linear(256, 3)
        self.backbone = DiffusionBackbone(259, 10, spec['training'])

    def encode_stage(self, model_inputs):
        features = model_inputs['features']
        latent = self.encoder(features.reshape(features.shape[0], -1))
        logits = self.stage_head(latent)
        return latent, logits

    def forward(self, noisy_action, timesteps, model_inputs):
        latent, logits = self.encode_stage(model_inputs)
        condition = torch.cat([latent, torch.softmax(logits, dim=-1)], dim=-1)
        return self.backbone(noisy_action, timesteps, condition)


def build_model(spec):
    return PhaseAwarePolicy(spec)


def compute_loss(model, batch, spec):
    prediction = model(batch['noisy_action'], batch['timesteps'], batch['model_inputs'])
    diffusion_loss = epsilon_loss(prediction, batch['noise'], batch['mask'])
    latent, logits = model.encode_stage(batch['model_inputs'])
    labels = batch['auxiliary']['phase'].long().reshape(-1)
    phase_mask = batch['auxiliary']['phase_mask'].to(logits.dtype).reshape(-1)
    per_item = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    auxiliary_loss = (per_item * phase_mask).sum() / phase_mask.sum().clamp(min=1.0)
    prior_loss = auxiliary_loss * 0.05
    return {'loss': diffusion_loss + prior_loss, 'diffusion_loss': diffusion_loss, 'prior_loss': prior_loss}
