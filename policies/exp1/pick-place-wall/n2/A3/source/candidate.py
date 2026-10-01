import numpy as np
import torch
from experiment1.contracts import CandidateDesign


class RelationalMotionDiffusion(CandidateDesign):
    def build_modules(self, common_dp_factory):
        width = self.design_config['edge_hidden_dim']
        edge_dim = self.design_config['edge_embedding_dim']
        self.dp = common_dp_factory(4 * edge_dim + 31, diffusion_action_dim=11)
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(24, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, edge_dim),
            torch.nn.LayerNorm(edge_dim),
            torch.nn.SiLU()
        )
        self.register_buffer('edge_roles', torch.eye(4, dtype=torch.float32))

    def condition(self, causal_history, causal_context):
        raw = causal_context['raw_history']
        h = raw[..., 0:3]
        o = raw[..., 4:7]
        g = raw[..., 36:39]
        w = raw[..., 39:42]
        size = raw[..., 42:45]
        vh = h - raw[..., 18:21]
        vo = o - raw[..., 22:25]
        zero = torch.zeros_like(vh)
        starts = torch.stack([h, o, h, o], dim=-2)
        ends = torch.stack([o, g, w, w], dim=-2)
        start_motion = torch.stack([vh, vo, vh, vo], dim=-2)
        end_motion = torch.stack([vo, zero, zero, zero], dim=-2)
        wall = w.unsqueeze(-2)
        half = size.unsqueeze(-2)
        top = wall[..., 2:3] + half[..., 2:3]
        delta = ends - starts
        metric_scale = self.design_config['relation_scale_m']
        motion_scale = self.common_spec['action_schema']['native_xyz_scale_m']
        roles = self.edge_roles.view(1, 1, 4, 4).expand(raw.shape[0], raw.shape[1], -1, -1)
        opening = (2.0 * (raw[..., 3:4] - 0.5)).unsqueeze(-2).expand(-1, -1, 4, -1)
        edges = torch.cat([
            delta / metric_scale,
            torch.sqrt((delta ** 2).sum(dim=-1, keepdim=True) + 1.0e-12) / metric_scale,
            torch.sqrt((delta[..., :2] ** 2).sum(dim=-1, keepdim=True) + 1.0e-12) / metric_scale,
            (end_motion - start_motion) / motion_scale,
            (starts[..., 2:3] - top) / 0.1,
            (ends[..., 2:3] - top) / 0.1,
            (torch.abs(starts - wall) - half) / metric_scale,
            (torch.abs(ends - wall) - half) / metric_scale,
            opening,
            half.expand(-1, -1, 4, -1) / metric_scale,
            roles
        ], dim=-1)
        learned = self.edge_encoder(edges).flatten(start_dim=-2)
        bypass = torch.cat([
            (h - w) / 0.2,
            (o - w) / 0.2,
            (g - w) / 0.2,
            w, size / 0.2,
            raw[..., 7:11], raw[..., 25:29],
            2.0 * (raw[..., 3:4] - 0.5), 2.0 * (raw[..., 21:22] - 0.5),
            vh / motion_scale, vo / motion_scale
        ], dim=-1)
        return torch.cat([learned, bypass], dim=-1)

    def training_targets(self, support_view, window_index):
        values = np.zeros((len(window_index), 16, 7), dtype=np.float32)
        displacement_scale = self.design_config['future_displacement_scale_m']
        for row, (episode_index, start) in enumerate(window_index):
            episode = support_view[episode_index]
            obs = np.asarray(episode['obs'], dtype=np.float32)
            valid = min(16, len(episode['actions']) - start)
            for k in range(valid):
                future = obs[start + k + 1]
                values[row, k, 0:3] = (future[0:3] - obs[start, 0:3]) / displacement_scale
                values[row, k, 3:6] = (future[4:7] - obs[start, 4:7]) / displacement_scale
                values[row, k, 6] = 2.0 * (future[3] - 0.5)
        return {'future_motion': values}

    def diffusion_targets(self, encoded_actions, targets, causal_context):
        return torch.cat([encoded_actions, targets['future_motion']], dim=-1)

    def training_loss(self, output, batch, common_diffusion_state):
        squared = (output[..., 4:11] - common_diffusion_state['noise'][..., 4:11]) ** 2
        mask = batch['mask']
        denominator = (mask.sum() * 7.0).clamp(min=1.0)
        auxiliary = (squared * mask.unsqueeze(-1)).sum() / denominator
        return self.design_config['aux_epsilon_weight'] * auxiliary, {'future_motion_epsilon': auxiliary.detach()}


def build_design(common_spec, config):
    return RelationalMotionDiffusion(common_spec, config)
