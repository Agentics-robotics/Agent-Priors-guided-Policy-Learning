import numpy as np
import torch
from experiment1.contracts import CandidateDesign


def orientation_columns(q):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-6)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack((1 - 2 * (y * y + z * z),
                        2 * (x * y + w * z),
                        2 * (x * z - w * y),
                        2 * (x * y - w * z),
                        1 - 2 * (x * x + z * z),
                        2 * (y * z + w * x)), dim=-1)


class JointRelationalDynamics(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.length_scale = float(config['length_scale_m'])
        self.motion_scale = float(config['motion_scale_m'])
        self.target_scale = float(config['future_length_scale_m'])
        self.auxiliary_weight = float(config['auxiliary_weight'])
        self.edge_width = int(config['edge_width'])
        self.edge_latent = int(config['edge_latent'])
        self.mixer_width = int(config['mixer_width'])
        self.latent_dim = int(config['latent_dim'])

    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(self.latent_dim + 36, diffusion_action_dim=11)
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(21, self.edge_width), torch.nn.SiLU(),
            torch.nn.Linear(self.edge_width, self.edge_latent), torch.nn.SiLU())
        self.relation_mixer = torch.nn.Sequential(
            torch.nn.Linear(4 * self.edge_latent, self.mixer_width), torch.nn.SiLU(),
            torch.nn.Linear(self.mixer_width, self.latent_dim), torch.nn.SiLU())
        self.register_buffer('edge_roles', torch.eye(4, dtype=torch.float32))

    def condition(self, causal_history, causal_context):
        raw = causal_context['raw_history']
        hand = raw[..., 0:3]
        site = raw[..., 4:7]
        ring = raw[..., 39:42]
        goal = raw[..., 36:39]
        hand_motion = (hand - raw[..., 18:21]) / self.motion_scale
        site_motion = (site - raw[..., 22:25]) / self.motion_scale
        ring_motion = torch.cat((torch.zeros_like(ring[:, :1]), ring[:, 1:] - ring[:, :-1]), dim=1) / self.motion_scale
        points = torch.stack((hand, site, ring, goal), dim=-2)
        motion = torch.stack((hand_motion, site_motion, ring_motion, torch.zeros_like(hand_motion)), dim=-2)
        axes = orientation_columns(raw[..., 7:11])
        aperture = (raw[..., 3:4] - 0.5) * 2.0
        pairs = ((0, 1), (2, 1), (2, 3), (0, 3))
        edge_inputs = []
        vectors = []
        for index, (source, target) in enumerate(pairs):
            delta = (points[..., target, :] - points[..., source, :]) / self.length_scale
            vectors.append(delta)
            distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
            planar = torch.linalg.vector_norm(delta[..., :2], dim=-1, keepdim=True)
            relative_motion = motion[..., target, :] - motion[..., source, :]
            role = self.edge_roles[index].view(1, 1, 4).expand(raw.shape[0], raw.shape[1], 4)
            edge_inputs.append(torch.cat((delta, distance, planar,
                                          points[..., source, 2:3] / self.length_scale,
                                          points[..., target, 2:3] / self.length_scale,
                                          relative_motion, role, axes, aperture), dim=-1))
        encoded_edges = self.edge_encoder(torch.stack(edge_inputs, dim=-2))
        learned = self.relation_mixer(encoded_edges.flatten(start_dim=2))
        physical_skip = torch.cat((torch.cat(vectors, dim=-1),
                                   (hand - raw.new_tensor([0.0, 0.6, 0.0])) / 0.5,
                                   points[..., 2] / self.length_scale,
                                   aperture, (raw[..., 3:4] - raw[..., 21:22]) / 0.1,
                                   axes, hand_motion, site_motion, ring_motion), dim=-1)
        return torch.cat((learned, physical_skip), dim=-1)

    def training_targets(self, support_view, window_index):
        future = np.zeros((len(window_index), 16, 7), dtype=np.float32)
        for row, (episode_index, start) in enumerate(window_index):
            episode = support_view[episode_index]
            obs = np.asarray(episode['obs'], dtype=np.float32)
            valid = min(16, len(episode['actions']) - start)
            next_obs = obs[start + 1:start + 1 + valid]
            future[row, :valid, 0:3] = (next_obs[:, 39:42] - obs[start, 36:39]) / self.target_scale
            future[row, :valid, 3:6] = (next_obs[:, 4:7] - next_obs[:, 0:3]) / self.target_scale
            future[row, :valid, 6:7] = (next_obs[:, 3:4] - 0.5) * 2.0
        return {'future_relations': future}

    def diffusion_targets(self, encoded_actions, targets, causal_context):
        return torch.cat((encoded_actions, targets['future_relations']), dim=-1)

    def training_loss(self, output, batch, common_diffusion_state):
        epsilon = output['epsilon'] if isinstance(output, dict) else output
        mask = batch['mask']
        error = (epsilon[..., 4:11] - common_diffusion_state['noise'][..., 4:11]).square()
        auxiliary = (error * mask.unsqueeze(-1)).sum() / (mask.sum() * 7).clamp_min(1.0)
        return self.auxiliary_weight * auxiliary, {'future_relation_epsilon': auxiliary.detach()}


def build_design(common_spec, config):
    return JointRelationalDynamics(common_spec, config)
