import numpy as np
import torch
from experiment1.contracts import CandidateDesign


class InteractionGraphDesign(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.origin = [0.0, 0.0, 0.0]

    def fit_support(self, support_view, common_spec):
        hands = np.concatenate([np.asarray(e['obs'][:-1, :3], dtype=np.float64) for e in support_view], axis=0)
        center = hands.mean(axis=0)
        self.origin = [float(center[0]), float(center[1]), 0.0]

    def build_modules(self, common_dp_factory):
        c = self.design_config
        width = c['node_width']
        self.dp = common_dp_factory(c['graph_output_dim'] + 39, diffusion_action_dim=10)
        self.node_encoder = torch.nn.Sequential(
            torch.nn.Linear(23, width), torch.nn.SiLU(), torch.nn.Linear(width, width), torch.nn.SiLU())
        self.messages = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(2 * width + 7, 2 * width), torch.nn.SiLU(), torch.nn.Linear(2 * width, width))
            for k in range(c['message_rounds'])])
        self.updates = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(2 * width, 2 * width), torch.nn.SiLU(), torch.nn.Linear(2 * width, width))
            for k in range(c['message_rounds'])])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(width) for k in range(c['message_rounds'])])
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(4 * width, 128), torch.nn.SiLU(), torch.nn.Linear(128, c['graph_output_dim']))

    def scene(self, raw):
        c = self.design_config
        pos = torch.stack([raw[..., :3], raw[..., 4:7], raw[..., 11:14], raw[..., 36:39]], dim=-2)
        old = torch.stack([raw[..., 18:21], raw[..., 22:25], raw[..., 29:32], raw[..., 36:39]], dim=-2)
        motion = (pos - old) / c['motion_scale_m']
        zero_q = torch.zeros_like(raw[..., :4])
        hand_q = torch.cat([torch.zeros_like(raw[..., :3]), torch.ones_like(raw[..., 3:4])], dim=-1)
        q = torch.stack([hand_q, raw[..., 7:11], raw[..., 14:18], zero_q], dim=-2)
        old_q = torch.stack([hand_q, raw[..., 25:29], raw[..., 32:36], zero_q], dim=-2)
        zero = torch.zeros_like(raw[..., 3:4])
        grip = torch.stack([raw[..., 3:4], zero, zero, zero], dim=-2)
        dgrip = torch.stack([raw[..., 3:4] - raw[..., 21:22], zero, zero, zero], dim=-2)
        role = torch.eye(4, dtype=raw.dtype, device=raw.device).view(1, 1, 4, 4).expand(raw.shape[0], raw.shape[1], 4, 4)
        origin = raw.new_tensor(self.origin)
        qscale = raw.new_tensor([0.1, 0.1, 0.1, 1.0])
        node = torch.cat([
            (pos - origin) / c['world_scale_m'],
            (pos - pos[..., :1, :]) / c['edge_scale_m'],
            motion, q / qscale, (q - old_q) / c['quaternion_delta_scale'],
            grip, dgrip / c['aperture_delta_scale'], role
        ], dim=-1)
        rel = (pos.unsqueeze(-3) - pos.unsqueeze(-2)) / c['edge_scale_m']
        rel_motion = motion.unsqueeze(-3) - motion.unsqueeze(-2)
        distance = torch.linalg.vector_norm(rel, dim=-1, keepdim=True)
        edge = torch.cat([rel, rel_motion, distance], dim=-1)
        return node, edge, motion

    def condition(self, causal_history, causal_context):
        raw = causal_context['raw_history']
        node, edge, motion = self.scene(raw)
        hidden = self.node_encoder(node)
        adjacency = raw.new_tensor([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]])
        edge_mask = adjacency.view(1, 1, 4, 4, 1)
        degree = adjacency.sum(dim=-1).view(1, 1, 4, 1)
        for k in range(self.design_config['message_rounds']):
            receivers = hidden.unsqueeze(-2).expand(-1, -1, 4, 4, -1)
            senders = hidden.unsqueeze(-3).expand(-1, -1, 4, 4, -1)
            messages = self.messages[k](torch.cat([receivers, senders, edge], dim=-1))
            aggregate = (messages * edge_mask).sum(dim=-2) / degree
            hidden = self.norms[k](hidden + self.updates[k](torch.cat([hidden, aggregate], dim=-1)))
        graph = self.readout(hidden.flatten(start_dim=-2))
        c = self.design_config
        qscale = raw.new_tensor([0.1, 0.1, 0.1, 1.0])
        direct = torch.cat([
            (raw[..., :3] - raw.new_tensor(self.origin)) / c['world_scale_m'],
            (raw[..., 4:7] - raw[..., :3]) / c['edge_scale_m'],
            (raw[..., 11:14] - raw[..., 4:7]) / c['edge_scale_m'],
            (raw[..., 36:39] - raw[..., 11:14]) / c['edge_scale_m'],
            raw[..., 7:11] / qscale, raw[..., 14:18] / qscale,
            raw[..., 25:29] / qscale, raw[..., 32:36] / qscale,
            raw[..., 3:4], raw[..., 21:22], motion[..., :3, :].flatten(start_dim=-2)
        ], dim=-1)
        return torch.cat([direct, graph], dim=-1)

    def training_targets(self, support_view, window_index):
        labels = np.zeros((len(window_index), 16, 6), dtype=np.float32)
        scale = self.design_config['future_motion_scale_m']
        for row, (episode_id, start) in enumerate(window_index):
            episode = support_view[episode_id]
            obs = episode['obs']
            valid = min(16, len(episode['actions']) - start)
            for k in range(valid):
                future = obs[start + k + 1]
                labels[row, k, :3] = (future[4:7] - obs[start, 4:7]) / scale
                labels[row, k, 3:] = (future[11:14] - obs[start, 11:14]) / scale
        return {'future_object_motion': labels}

    def diffusion_targets(self, encoded_actions, targets, causal_context):
        return torch.cat([encoded_actions, targets['future_object_motion']], dim=-1)

    def training_loss(self, output, batch, common_diffusion_state):
        mask = batch['mask'].to(dtype=output.dtype)
        error = (output[..., 4:] - common_diffusion_state['noise'][..., 4:]).square()
        motion_loss = (error * mask[..., None]).sum() / (mask.sum().clamp_min(1.0) * 6.0)
        extra = self.design_config['motion_loss_weight'] * motion_loss
        return extra, {'motion_epsilon_mse': motion_loss.detach()}

    def deployment_state_dict(self):
        return {'world_xy_origin_m': list(self.origin)}

    def load_deployment_state_dict(self, state):
        self.origin = [float(x) for x in state['world_xy_origin_m']]


def build_design(common_spec, config):
    return InteractionGraphDesign(common_spec, config)
