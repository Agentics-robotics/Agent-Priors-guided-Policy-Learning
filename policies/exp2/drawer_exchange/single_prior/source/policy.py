import torch
from appl.public import epsilon_loss
from experiments.exp2.sol_flexible.public import DiffusionBackbone


class RelationalFullTaskPolicy(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.condition_encoder = torch.nn.Sequential(
            torch.nn.Linear(142, 256),
            torch.nn.Mish(),
            torch.nn.Linear(256, 256),
            torch.nn.Mish(),
        )
        self.backbone = DiffusionBackbone(256, 14, spec['training'])
        self.anchor_head = torch.nn.Linear(256, 16 * 4)

    def encode_condition(self, model_inputs):
        history = model_inputs['history']
        return self.condition_encoder(history.reshape(history.shape[0], 142))

    def forward(self, noisy_action, timesteps, model_inputs):
        condition = self.encode_condition(model_inputs)
        return self.backbone(noisy_action, timesteps, condition)

    def predict_anchor_logits(self, model_inputs):
        condition = self.encode_condition(model_inputs)
        return self.anchor_head(condition).reshape(condition.shape[0], 16, 4)


def build_model(spec):
    return RelationalFullTaskPolicy(spec)


def compute_loss(model, batch, spec):
    prediction = model(batch['noisy_action'], batch['timesteps'], batch['model_inputs'])
    diffusion_loss = epsilon_loss(prediction, batch['noise'], batch['mask'])
    logits = model.predict_anchor_logits(batch['model_inputs'])
    target = batch['auxiliary']['anchor_index'].long()
    if target.ndim == 3:
        target = target[..., 0]
    anchor_mask = batch['auxiliary']['anchor_mask']
    if anchor_mask.ndim == 3:
        anchor_mask = anchor_mask[..., 0]
    anchor_mask = anchor_mask.to(dtype=logits.dtype)
    per_slot = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 4), target.reshape(-1), reduction='none'
    ).reshape(logits.shape[0], 16)
    prior_loss = (per_slot * anchor_mask).sum() / anchor_mask.sum().clamp(min=1.0)
    loss = diffusion_loss + 0.05 * prior_loss
    return {
        'loss': loss,
        'diffusion_loss': diffusion_loss,
        'prior_loss': prior_loss,
    }
