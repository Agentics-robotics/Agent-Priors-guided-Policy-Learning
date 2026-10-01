import torch
from appl.public import epsilon_loss
from experiments.exp2.six_tasks.flexible.public import DiffusionBackbone


class ClosedDestinationPolicy(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.condition_encoder = torch.nn.Sequential(
            torch.nn.Linear(84, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 256),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(256),
        )
        self.backbone = DiffusionBackbone(256, 10, spec['training'])
        self.closure_head = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 1),
        )

    def encode_condition(self, model_inputs):
        features = model_inputs['features']
        return self.condition_encoder(features.reshape(features.shape[0], -1))

    def forward(self, noisy_action, timesteps, model_inputs):
        condition = self.encode_condition(model_inputs)
        return self.backbone(noisy_action, timesteps, condition)


def build_model(spec):
    return ClosedDestinationPolicy(spec)


def compute_loss(model, batch, spec):
    prediction = model(batch['noisy_action'], batch['timesteps'], batch['model_inputs'])
    diffusion_loss = epsilon_loss(prediction, batch['noise'], batch['mask'])
    condition = model.encode_condition(batch['model_inputs'])
    closure_prediction_scaled = model.closure_head(condition)
    closure_target_m = batch['auxiliary']['closure_delta']
    closure_mask = batch['auxiliary']['closure_mask']
    squared = (closure_prediction_scaled - closure_target_m / 0.1) ** 2
    auxiliary_loss = (squared * closure_mask).sum() / closure_mask.sum().clamp_min(1.0)
    prior_loss = 0.05 * auxiliary_loss
    return {'loss': diffusion_loss + prior_loss, 'diffusion_loss': diffusion_loss, 'prior_loss': prior_loss}
