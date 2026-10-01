import torch
from appl.public import epsilon_loss
from experiments.exp2.six_tasks.flexible.public import DiffusionBackbone


class BowlSourcePosePolicy(torch.nn.Module):
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
        self.backbone = DiffusionBackbone(384, 10, spec['training'])

    def condition(self, model_inputs):
        robot = model_inputs['robot']
        particles = model_inputs['particles']
        robot_embedding = self.robot_encoder(robot).reshape(robot.shape[0], -1)
        token_embedding = self.particle_encoder(particles)
        token_mean = token_embedding.mean(dim=2)
        token_max = token_embedding.max(dim=2).values
        particle_embedding = torch.cat([token_mean, token_max], dim=-1).reshape(robot.shape[0], -1)
        return torch.cat([robot_embedding, particle_embedding], dim=-1)

    def forward(self, noisy_action, timesteps, model_inputs):
        return self.backbone(noisy_action, timesteps, self.condition(model_inputs))


def build_model(spec):
    return BowlSourcePosePolicy(spec)


def compute_loss(model, batch, spec):
    prediction = model(batch['noisy_action'], batch['timesteps'], batch['model_inputs'])
    diffusion_loss = epsilon_loss(prediction, batch['noise'], batch['mask'])
    prior_loss = prediction.sum() * 0.0
    return {
        'loss': diffusion_loss + prior_loss,
        'diffusion_loss': diffusion_loss,
        'prior_loss': prior_loss,
    }
