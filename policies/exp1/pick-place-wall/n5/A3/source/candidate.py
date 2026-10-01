import numpy as np
import torch
from experiment1.contracts import CandidateDesign


class RelationalOutcomeDiffusion(CandidateDesign):
    def build_modules(self, common_dp_factory):
        width = int(self.design_config['node_width'])
        hidden = int(self.design_config['encoder_hidden_width'])
        self.dp = common_dp_factory(4 * width + 32, diffusion_action_dim=11)
        self.node_encoder = torch.nn.Sequential(
            torch.nn.Linear(22, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, width))
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(2 * width + 7, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, width))
        self.node_update = torch.nn.Sequential(
            torch.nn.Linear(2 * width, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, width))

    def condition(self, causal_history, causal_context):
        r = causal_context['raw_history']
        h, o, g = r[..., 0:3], r[..., 4:7], r[..., 36:39]
        hp, op = r[..., 18:21], r[..., 22:25]
        w, half = r[..., 39:42], r[..., 42:45]
        q, qp = r[..., 7:11], r[..., 25:29]
        s, v = self.design_config['position_scale_m'], self.design_config['motion_scale_m']
        zero1, zero3, zero4 = torch.zeros_like(r[..., 3:4]), torch.zeros_like(h), torch.zeros_like(q)
        grip = 2.0 * r[..., 3:4] - 1.0
        dg = 10.0 * (r[..., 3:4] - r[..., 21:22])
        positions = torch.stack((h, o, g, w), dim=-2)
        velocity = torch.stack(((h - hp) / v, (o - op) / v, zero3, zero3), dim=-2)
        orientation = torch.stack((zero4, q, zero4, zero4), dim=-2)
        orientation_change = torch.stack((zero4, q - qp, zero4, zero4), dim=-2)
        attributes = torch.stack((torch.cat((grip, dg, zero1), dim=-1),
                                  zero3, zero3, half / s), dim=-2)
        identities = torch.eye(4, dtype=r.dtype, device=r.device)
        types = identities.reshape(1, 1, 4, 4).expand(r.shape[0], r.shape[1], 4, 4)
        node_features = torch.cat(((positions - w.unsqueeze(-2)) / s,
                                   positions[..., 2:3] / s, velocity,
                                   orientation, orientation_change, attributes, types), dim=-1)
        nodes = self.node_encoder(node_features)
        relative_position = (positions.unsqueeze(-3) - positions.unsqueeze(-2)) / s
        relative_velocity = velocity.unsqueeze(-3) - velocity.unsqueeze(-2)
        edge_geometry = torch.cat((relative_position,
                                   torch.linalg.vector_norm(relative_position, dim=-1, keepdim=True),
                                   relative_velocity), dim=-1)
        off_diagonal = (1.0 - identities).reshape(1, 1, 4, 4, 1)
        width = int(self.design_config['node_width'])
        for step in range(int(self.design_config['message_steps'])):
            receiver = nodes.unsqueeze(-2).expand(r.shape[0], r.shape[1], 4, 4, width)
            sender = nodes.unsqueeze(-3).expand(r.shape[0], r.shape[1], 4, 4, width)
            messages = self.edge_encoder(torch.cat((receiver, sender, edge_geometry), dim=-1))
            aggregated = (messages * off_diagonal).sum(dim=-2) / 3.0
            nodes = nodes + self.node_update(torch.cat((nodes, aggregated), dim=-1))
        skip = torch.cat(((o - h) / s, (g - o) / s, (h - w) / s, (o - w) / s,
                          half / s, w / self.design_config['world_scale_m'],
                          h[..., 2:3] / s, o[..., 2:3] / s, g[..., 2:3] / s,
                          grip, q, (h - hp) / v, (o - op) / v), dim=-1)
        return torch.cat((nodes.flatten(start_dim=2), skip), dim=-1)

    def training_targets(self, support_view, window_index):
        outcomes = np.zeros((len(window_index), 16, 7), dtype=np.float32)
        s = self.design_config['position_scale_m']
        for row, pair in enumerate(window_index):
            episode_index, start = pair
            episode = support_view[episode_index]
            valid = min(16, len(episode['actions']) - start)
            future = np.asarray(episode['obs'][start + 1:start + valid + 1], dtype=np.float32)
            outcomes[row, :valid, 0:3] = (future[:, 4:7] - future[:, 0:3]) / s
            outcomes[row, :valid, 3:6] = (future[:, 36:39] - future[:, 4:7]) / s
            outcomes[row, :valid, 6:7] = 2.0 * future[:, 3:4] - 1.0
        return {'future_relations': outcomes}

    def diffusion_targets(self, encoded_actions, targets, causal_context):
        return torch.cat((encoded_actions, targets['future_relations']), dim=-1)

    def training_loss(self, output, batch, common_diffusion_state):
        error = (output[..., 4:] - common_diffusion_state['noise'][..., 4:]) ** 2
        mask = batch['mask']
        denominator = (mask.sum() * 7.0).clamp_min(1.0)
        outcome_loss = (error * mask.unsqueeze(-1)).sum() / denominator
        return self.design_config['outcome_loss_weight'] * outcome_loss, {'outcome_epsilon_loss': outcome_loss.detach()}


def build_design(common_spec, config):
    return RelationalOutcomeDiffusion(common_spec, config)
