import torch
from appl.public import epsilon_loss
from experiments.exp2.six_tasks.flexible.public import DiffusionBackbone


class PhasePredictiveDiffusionPolicy(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.condition_encoder = torch.nn.Sequential(
            torch.nn.Linear(86, 256),
            torch.nn.Mish(),
            torch.nn.Linear(256, 256),
            torch.nn.Mish(),
        )
        self.backbone = DiffusionBackbone(256, 10, spec['training'])
        self.phase_head = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.Mish(),
            torch.nn.Linear(256, 32),
        )

    def encode_condition(self, model_inputs):
        features = model_inputs['features']
        return self.condition_encoder(features.reshape(features.shape[0], 86))

    def forward(self, noisy_action, timesteps, model_inputs):
        condition = self.encode_condition(model_inputs)
        return self.backbone(noisy_action, timesteps, condition)

    def predict_phase(self, model_inputs):
        condition = self.encode_condition(model_inputs)
        return self.phase_head(condition).reshape(condition.shape[0], 16, 2)


def build_model(spec):
    return PhasePredictiveDiffusionPolicy(spec)


def compute_loss(model, batch, spec):
    prediction = model(batch['noisy_action'], batch['timesteps'], batch['model_inputs'])
    diffusion_loss = epsilon_loss(prediction, batch['noise'], batch['mask'])
    phase_prediction = model.predict_phase(batch['model_inputs'])
    targets = batch['auxiliary']['phase_targets']
    phase_mask = batch['auxiliary']['phase_mask']
    squared = (phase_prediction - targets) ** 2
    phase_loss = (squared * phase_mask).sum() / (phase_mask.sum().clamp_min(1.0) * targets.shape[-1])
    prior_loss = 0.05 * phase_loss
    return {
        'loss': diffusion_loss + prior_loss,
        'diffusion_loss': diffusion_loss,
        'prior_loss': prior_loss,
    }
