import torch
from appl.public import epsilon_loss
from experiments.exp2.six_tasks.flexible.public import DiffusionBackbone


class ParticlePhasePolicy(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.particle_encoder = torch.nn.Sequential(
            torch.nn.Linear(8, 64),
            torch.nn.Mish(),
            torch.nn.Linear(64, 64),
            torch.nn.Mish(),
        )
        self.robot_encoder = torch.nn.Sequential(
            torch.nn.Linear(58, 64),
            torch.nn.Mish(),
            torch.nn.Linear(64, 64),
            torch.nn.Mish(),
        )
        self.backbone = DiffusionBackbone(384, 7, spec['training'])
        self.phase_head = torch.nn.Sequential(
            torch.nn.Linear(384, 128),
            torch.nn.Mish(),
            torch.nn.Linear(128, 3),
        )

    def condition(self, model_inputs):
        robot = model_inputs['robot']
        particles = model_inputs['particles']
        robot_embedding = self.robot_encoder(robot).reshape(robot.shape[0], -1)
        token_embedding = self.particle_encoder(particles)
        token_mean = token_embedding.mean(dim=2)
        token_max = token_embedding.max(dim=2).values
        pooled = torch.cat([token_mean, token_max], dim=-1).reshape(robot.shape[0], -1)
        return torch.cat([robot_embedding, pooled], dim=-1)

    def forward(self, noisy_action, timesteps, model_inputs):
        return self.backbone(noisy_action, timesteps, self.condition(model_inputs))

    def predict_phase(self, model_inputs):
        return self.phase_head(self.condition(model_inputs))


def build_model(spec):
    return ParticlePhasePolicy(spec)


def compute_loss(model, batch, spec):
    prediction = model(batch['noisy_action'], batch['timesteps'], batch['model_inputs'])
    diffusion_loss = epsilon_loss(prediction, batch['noise'], batch['mask'])
    phase_prediction = model.predict_phase(batch['model_inputs'])
    phase_target = batch['auxiliary']['phase']
    phase_mask = batch['auxiliary']['phase_mask']
    squared = (phase_prediction - phase_target) ** 2
    phase_loss = (squared * phase_mask).sum() / (phase_mask.sum() * 3.0).clamp_min(1.0)
    prior_loss = 0.05 * phase_loss
    return {
        'loss': diffusion_loss + prior_loss,
        'diffusion_loss': diffusion_loss,
        'prior_loss': prior_loss,
    }
