import torch
from appl.public import epsilon_loss
from experiments.exp2.six_tasks.flexible.public import DiffusionBackbone


class SupportPhasePolicy(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.condition_encoder = torch.nn.Sequential(
            torch.nn.Linear(134, 256),
            torch.nn.Mish(),
            torch.nn.Linear(256, 256),
            torch.nn.Mish()
        )
        self.backbone = DiffusionBackbone(256, 10, spec['training'])
        self.open_head = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.Mish(),
            torch.nn.Linear(128, 16)
        )

    def encode_condition(self, model_inputs):
        flat = model_inputs['features'].reshape(model_inputs['features'].shape[0], -1)
        return self.condition_encoder(flat)

    def forward(self, noisy_action, timesteps, model_inputs):
        condition = self.encode_condition(model_inputs)
        return self.backbone(noisy_action, timesteps, condition)

    def predict_open_logits(self, model_inputs):
        return self.open_head(self.encode_condition(model_inputs))


def build_model(spec):
    return SupportPhasePolicy(spec)


def compute_loss(model, batch, spec):
    prediction = model(batch['noisy_action'], batch['timesteps'], batch['model_inputs'])
    diffusion = epsilon_loss(prediction, batch['noise'], batch['mask'])
    logits = model.predict_open_logits(batch['model_inputs']).unsqueeze(-1)
    target = batch['auxiliary']['open_target']
    valid = batch['auxiliary']['open_mask']
    elementwise = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction='none')
    auxiliary = (elementwise * valid).sum() / valid.sum().clamp_min(1.0)
    prior = 0.05 * auxiliary
    return {'loss': diffusion + prior, 'diffusion_loss': diffusion, 'prior_loss': prior}
