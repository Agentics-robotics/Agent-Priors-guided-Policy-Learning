import torch
from appl.public import epsilon_loss
from experiments.exp2.sol_flexible.public import DiffusionBackbone


class PhaseConditionedPolicy(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        feature_dimension = int(spec['contract']['input_shapes']['features'][1])
        self.frame_encoder = torch.nn.Sequential(
            torch.nn.Linear(feature_dimension, 128),
            torch.nn.Mish(),
            torch.nn.Linear(128, 128),
            torch.nn.Mish()
        )
        self.condition_encoder = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.Mish(),
            torch.nn.Linear(256, 256),
            torch.nn.Mish()
        )
        self.phase_head = torch.nn.Linear(256, 5)
        self.backbone = DiffusionBackbone(256, spec['contract']['action_dimension'], spec['training'])

    def encode_condition(self, model_inputs):
        per_frame = self.frame_encoder(model_inputs['features'])
        return self.condition_encoder(per_frame.reshape(per_frame.shape[0], -1))

    def forward(self, noisy_action, timesteps, model_inputs):
        return self.backbone(noisy_action, timesteps, self.encode_condition(model_inputs))

    def phase_logits(self, model_inputs):
        return self.phase_head(self.encode_condition(model_inputs))


def build_model(spec):
    return PhaseConditionedPolicy(spec)


def compute_loss(model, batch, spec):
    prediction = model(batch['noisy_action'], batch['timesteps'], batch['model_inputs'])
    diffusion_loss = epsilon_loss(prediction, batch['noise'], batch['mask'])
    logits = model.phase_logits(batch['model_inputs'])
    labels = batch['auxiliary']['phase'].long().reshape(-1)
    phase_mask = batch['auxiliary']['phase_mask'].to(logits.dtype).reshape(-1)
    per_sample = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    phase_loss = (per_sample * phase_mask).sum() / phase_mask.sum().clamp(min=1.0)
    prior_loss = 0.05 * phase_loss
    return {'loss': diffusion_loss + prior_loss, 'diffusion_loss': diffusion_loss, 'prior_loss': prior_loss}
