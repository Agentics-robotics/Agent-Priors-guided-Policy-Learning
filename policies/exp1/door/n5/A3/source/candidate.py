import numpy as np
import torch
from experiment1.contracts import CandidateDesign


def yaw_pair(raw):
    s, c = raw[..., 39], raw[..., 40]
    n = torch.sqrt(s * s + c * c).clamp_min(1e-8)
    good = s * s + c * c > 1e-12
    return torch.where(good, s / n, torch.zeros_like(s)), torch.where(good, c / n, torch.ones_like(c))


def local_vector(v, s, c):
    return torch.stack((c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1], v[..., 2]), dim=-1)


def distance(v):
    return torch.sqrt((v * v).sum(dim=-1, keepdim=True) + 1e-12)


def attitude(q, s, c):
    q = q / torch.sqrt((q * q).sum(dim=-1, keepdim=True)).clamp_min(1e-8)
    x, y, z, w = q.unbind(dim=-1)
    a = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)), dim=-1)
    b = torch.stack((2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)), dim=-1)
    return torch.cat((local_vector(a, s, c), local_vector(b, s, c)), dim=-1)


class RelationalJointMotion(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.world_scale = float(config['world_scale_m'])
        self.edge_scale = float(config['edge_scale_m'])
        self.motion_scale = float(config['motion_scale_m'])
        self.future_scale = float(config['future_scale_m'])
        self.aux_weight = float(config['aux_weight'])

    def build_modules(self, common_dp_factory):
        width = int(self.design_config['node_width'])
        hidden = int(self.design_config['message_hidden'])
        latent = int(self.design_config['readout_width'])
        rounds = int(self.design_config['message_rounds'])
        self.dp = common_dp_factory(43 + latent, diffusion_action_dim=10)
        self.register_buffer('node_types', torch.eye(3, dtype=torch.float32).reshape(1, 1, 3, 3))
        self.register_buffer('edge_mask', (1.0 - torch.eye(3, dtype=torch.float32)).reshape(1, 1, 3, 3, 1))
        self.node_encoder = torch.nn.Sequential(torch.nn.Linear(19, width), torch.nn.SiLU(), torch.nn.Linear(width, width))
        self.messages = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(2 * width + 10, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, width))
            for i in range(rounds)
        ])
        self.updates = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(2 * width, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, width))
            for i in range(rounds)
        ])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(width) for i in range(rounds)])
        self.readout = torch.nn.Sequential(torch.nn.Linear(3 * width, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, latent))

    def condition(self, causal_history, causal_context):
        raw = causal_context['raw_history']
        s, c = yaw_pair(raw)
        h, o, g = raw[..., :3], raw[..., 4:7], raw[..., 36:39]
        ph, po = raw[..., 18:21], raw[..., 22:25]
        dh, do = h - ph, o - po
        zero3 = torch.zeros_like(h)
        zero1 = torch.zeros_like(raw[..., 3:4])
        current_att = attitude(raw[..., 7:11], s, c)
        previous_att = attitude(raw[..., 25:29], s, c)
        zero6 = torch.zeros_like(current_att)
        positions = torch.stack((h, o, g), dim=-2)
        local_positions = torch.stack((local_vector(h - g, s, c), local_vector(o - g, s, c), zero3), dim=-2)
        motion = torch.stack((local_vector(dh, s, c), local_vector(do, s, c), zero3), dim=-2)
        poses = torch.stack((zero6, current_att, zero6), dim=-2)
        apertures = torch.stack((raw[..., 3:4], zero1, zero1), dim=-2)
        types = self.node_types.expand(raw.shape[0], raw.shape[1], -1, -1)
        node_input = torch.cat((positions / self.world_scale, local_positions / self.world_scale,
                                motion / self.motion_scale, poses, apertures, types), dim=-1)
        world_edge = positions.unsqueeze(-3) - positions.unsqueeze(-2)
        cabinet_edge = local_positions.unsqueeze(-3) - local_positions.unsqueeze(-2)
        motion_edge = motion.unsqueeze(-3) - motion.unsqueeze(-2)
        edges = torch.cat((world_edge / self.edge_scale, cabinet_edge / self.edge_scale,
                           distance(world_edge) / self.edge_scale, motion_edge / self.motion_scale), dim=-1)
        nodes = self.node_encoder(node_input)
        for message_net, update_net, norm in zip(self.messages, self.updates, self.norms):
            receivers = nodes.unsqueeze(-2).expand(-1, -1, -1, 3, -1)
            senders = nodes.unsqueeze(-3).expand(-1, -1, 3, -1, -1)
            messages = message_net(torch.cat((receivers, senders, edges), dim=-1))
            pooled = (messages * self.edge_mask).sum(dim=-2) / 2.0
            nodes = norm(nodes + update_net(torch.cat((nodes, pooled), dim=-1)))
        latent = self.readout(nodes.flatten(start_dim=-2))
        skip = torch.cat((h / self.world_scale, o / self.world_scale, g / self.world_scale,
                          ph / self.world_scale, po / self.world_scale,
                          raw[..., 3:4], raw[..., 21:22], torch.stack((s, c), dim=-1),
                          current_att, previous_att,
                          local_vector(h - o, s, c) / 0.1,
                          local_vector(g - o, s, c) / self.world_scale,
                          dh / self.motion_scale, do / self.motion_scale), dim=-1)
        return torch.cat((skip, latent), dim=-1)

    def training_targets(self, support_view, window_index):
        future = np.zeros((len(window_index), 16, 6), dtype=np.float32)
        for row, pair in enumerate(window_index):
            episode_index, t = pair
            raw = np.asarray(support_view[episode_index]['obs'], dtype=np.float32)
            total = len(support_view[episode_index]['actions'])
            indices = np.minimum(t + 1 + np.arange(16), total)
            dh = raw[indices, :3] - raw[t, :3]
            do = raw[indices, 4:7] - raw[t, 4:7]
            s, c = float(raw[t, 39]), float(raw[t, 40])
            norm = max(float(np.sqrt(s * s + c * c)), 1e-8)
            if s * s + c * c > 1e-12:
                s, c = s / norm, c / norm
            else:
                s, c = 0.0, 1.0
            lh = np.stack((c * dh[:, 0] + s * dh[:, 1], -s * dh[:, 0] + c * dh[:, 1], dh[:, 2]), axis=-1)
            lo = np.stack((c * do[:, 0] + s * do[:, 1], -s * do[:, 0] + c * do[:, 1], do[:, 2]), axis=-1)
            future[row] = np.concatenate((lh, lo), axis=-1) / self.future_scale
        return {'future_displacement': future}

    def diffusion_targets(self, encoded_actions, targets, causal_context):
        return torch.cat((encoded_actions, targets['future_displacement']), dim=-1)

    def training_loss(self, output, batch, common_diffusion_state):
        epsilon = output['epsilon'] if isinstance(output, dict) else output
        error = (epsilon[..., 4:10] - common_diffusion_state['noise'][..., 4:10]) ** 2
        mask = batch['mask']
        aux = (error * mask.unsqueeze(-1)).sum() / (mask.sum() * 6.0).clamp_min(1.0)
        return self.aux_weight * aux, {'future_motion_epsilon': aux.detach()}


def build_design(common_spec, config):
    return RelationalJointMotion(common_spec, config)
