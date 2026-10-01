import numpy as np
import torch
from experiment1.contracts import CandidateDesign


def rotation6(q):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1.0e-6)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z),
                        2 * (x * z - w * y), 2 * (x * y - w * z),
                        1 - 2 * (x * x + z * z), 2 * (y * z + w * x)), dim=-1)


def planar_norm(v):
    return torch.linalg.vector_norm(v[..., :2], dim=-1, keepdim=True)


class StageRoutedRelations(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.width = int(config['encoder_width'])
        self.latent = int(config['relation_latent'])
        self.stage_width = int(config['stage_width'])
        self.gate_floor = float(config['gate_floor'])
        self.stage_weight = float(config['stage_loss_weight'])
        self.rest_height = 0.02

    def fit_support(self, support_view, common_spec):
        self.rest_height = float(np.median([float(e['obs'][0, 41]) for e in support_view]))

    def deployment_state_dict(self):
        return {'rest_height_m': self.rest_height}

    def load_deployment_state_dict(self, state):
        self.rest_height = float(state['rest_height_m'])

    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(2 * self.latent + 22)
        self.grasp_encoder = torch.nn.Sequential(torch.nn.Linear(26, self.width), torch.nn.SiLU(),
                                                  torch.nn.Linear(self.width, self.latent), torch.nn.SiLU())
        self.carry_encoder = torch.nn.Sequential(torch.nn.Linear(28, self.width), torch.nn.SiLU(),
                                                  torch.nn.Linear(self.width, self.latent), torch.nn.SiLU())
        self.stage_encoder = torch.nn.Sequential(torch.nn.Linear(14, self.stage_width), torch.nn.SiLU(),
                                                  torch.nn.Linear(self.stage_width, 5))

    def condition(self, causal_history, causal_context):
        raw = causal_context['raw_history']
        hand, obj = raw[..., 0:3], raw[..., 4:7]
        phand, pobj = raw[..., 18:21], raw[..., 22:25]
        goal, ring = raw[..., 36:39], raw[..., 39:42]
        grip, pgrip = raw[..., 3:4], raw[..., 21:22]
        grasp, place = obj - hand, goal - ring
        vh, vo = (hand - phand) / 0.01, (obj - pobj) / 0.01
        orient = rotation6(raw[..., 7:11])
        grasp_features = torch.cat((grasp / 0.05, (pobj - phand) / 0.05,
                                    (obj - ring) / 0.13, orient, grip, pgrip,
                                    vh, vo, hand[..., 2:3] / 0.2, obj[..., 2:3] / 0.2,
                                    ring[..., 2:3] / 0.2), dim=-1)
        carry_features = torch.cat((place / 0.1, (goal - hand) / 0.2,
                                    (obj - ring) / 0.13, grasp / 0.05, orient,
                                    ring[..., 2:3] / 0.2, goal[..., 2:3] / 0.2,
                                    hand[..., 2:3] / 0.2, grip, vh, vo), dim=-1)
        stage_features = torch.cat((planar_norm(grasp) / 0.1, grasp[..., 2:3] / 0.1,
                                    planar_norm(place) / 0.2, place[..., 2:3] / 0.2,
                                    hand[..., 2:3] / 0.2,
                                    (ring[..., 2:3] - self.rest_height) / 0.2,
                                    goal[..., 2:3] / 0.2, grip, pgrip,
                                    planar_norm(vh), vh[..., 2:3],
                                    planar_norm(vo), vo[..., 2:3],
                                    torch.linalg.vector_norm(grasp, dim=-1, keepdim=True) / 0.1), dim=-1)
        probabilities = torch.softmax(self.stage_encoder(stage_features), dim=-1)
        grasp_gate = self.gate_floor + (1 - self.gate_floor) * probabilities[..., :2].sum(dim=-1, keepdim=True)
        carry_gate = self.gate_floor + (1 - self.gate_floor) * probabilities[..., 2:].sum(dim=-1, keepdim=True)
        grasp_latent = self.grasp_encoder(grasp_features) * grasp_gate
        carry_latent = self.carry_encoder(carry_features) * carry_gate
        origin = raw.new_tensor([0.0, 0.6, 0.0])
        world = torch.cat((hand - origin, phand - origin), dim=-1) / 0.3
        edge_skip = torch.cat((grasp, place), dim=-1) / 0.1
        state_skip = torch.cat((hand[..., 2:3] / 0.2, ring[..., 2:3] / 0.2,
                                goal[..., 2:3] / 0.2, grip, pgrip), dim=-1)
        return torch.cat((grasp_latent, carry_latent, world, edge_skip,
                          state_skip, probabilities), dim=-1)

    def training_targets(self, support_view, window_index):
        episode_labels = []
        for episode in support_view:
            actions = np.asarray(episode['actions'], dtype=np.float32)
            labels = np.zeros((len(actions), 5), dtype=np.float32)
            stage = 0
            for t in range(len(actions)):
                action = actions[t]
                if stage == 0 and action[3] > 0.3:
                    stage = 1
                elif stage == 1 and action[2] > 0.5:
                    stage = 2
                elif stage == 2 and float(np.linalg.norm(action[:2])) > 0.25:
                    stage = 3
                elif stage == 3 and action[2] < -0.5:
                    stage = 4
                labels[t, stage] = 1.0
            episode_labels.append(labels)
        aligned = np.stack([episode_labels[e][[max(0, t - 1), t]] for e, t in window_index])
        return {'support_stage': aligned}

    def denoise(self, noisy_actions, timesteps, conditioning, causal_context):
        return {'epsilon': self.dp(noisy_actions, timesteps, conditioning),
                'stage_probability': conditioning[..., -5:]}

    def training_loss(self, output, batch, common_diffusion_state):
        probability = output['stage_probability'].clamp_min(1.0e-6)
        label = batch['targets']['support_stage']
        per_history = -(label * torch.log(probability)).sum(dim=-1)
        valid_start = batch['mask'][:, 0:1]
        stage_loss = (per_history * valid_start).sum() / (2 * valid_start.sum()).clamp_min(1.0)
        return self.stage_weight * stage_loss, {'stage_nll': stage_loss.detach()}


def build_design(common_spec, config):
    return StageRoutedRelations(common_spec, config)
