import numpy as np
import torch
import torch.nn as nn
from experiment1.contracts import CandidateDesign


def yaw_pair(raw):
    s, c = raw[..., 39], raw[..., 40]
    n = torch.sqrt(s * s + c * c)
    return torch.where(n > 1.0e-6, s / n.clamp_min(1.0e-8), torch.zeros_like(s)), torch.where(n > 1.0e-6, c / n.clamp_min(1.0e-8), torch.ones_like(c))


def to_local(v, s, c):
    return torch.stack((c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1], v[..., 2]), dim=-1)


def pose_columns(q, s, c):
    q = q / torch.sqrt((q * q).sum(dim=-1, keepdim=True)).clamp_min(1.0e-8)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    cy = torch.stack((2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)), dim=-1)
    cz = torch.stack((2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)), dim=-1)
    return torch.cat((to_local(cy, s, c), to_local(cz, s, c)), dim=-1)


def numpy_local(v, s, c):
    return np.stack((c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1], v[..., 2]), axis=-1)


class RelationalJointDesign(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.node_dim = int(config['node_embedding'])
        self.hidden = int(config['encoder_hidden'])
        self.contact_scale = float(config['contact_scale_m'])
        self.scene_scale = float(config['scene_scale_m'])
        self.edge_scale = float(config['edge_scale_m'])
        self.delta_scale = float(config['step_displacement_scale_m'])
        self.pose_delta_scale = float(config['rotation_column_delta_scale'])
        self.aux_weight = float(config['geometry_epsilon_weight'])

    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(3 * self.node_dim + 13, diffusion_action_dim=10)
        self.node_encoder = nn.Sequential(nn.Linear(26, self.hidden), nn.SiLU(), nn.Linear(self.hidden, self.node_dim), nn.SiLU())
        self.message_encoder = nn.Sequential(nn.Linear(2 * self.node_dim + 7, self.hidden), nn.SiLU(), nn.Linear(self.hidden, self.node_dim), nn.SiLU())
        self.node_update = nn.Sequential(nn.Linear(2 * self.node_dim, self.hidden), nn.SiLU(), nn.Linear(self.hidden, self.node_dim), nn.SiLU())

    def condition(self, causal_history, causal_context):
        r = causal_context['raw_history']
        s, c = yaw_pair(r)
        h, o, g = r[..., :3], r[..., 4:7], r[..., 36:39]
        hp, op = r[..., 18:21], r[..., 22:25]
        zero = torch.zeros_like(r[..., 3:4])
        one = torch.ones_like(zero)
        zero3 = torch.zeros_like(h)
        zero6 = torch.cat((zero3, zero3), dim=-1)
        world_y = torch.cat((zero, one, zero), dim=-1)
        world_z = torch.cat((zero, zero, one), dim=-1)
        # The hand has fixed native orientation. These are world-reference axes,
        # not an inferred end-effector pose or an extra observation channel.
        hand_axes = torch.cat((to_local(world_y, s, c), to_local(world_z, s, c)), dim=-1)
        object_axes = pose_columns(r[..., 7:11], s, c)
        old_object_axes = pose_columns(r[..., 25:29], s, c)
        cabinet_axes = torch.cat((world_y, world_z), dim=-1)
        positions = [h, o, g]
        velocities = [to_local(h - hp, s, c) / self.delta_scale, to_local(o - op, s, c) / self.delta_scale, zero3]
        apertures = [torch.cat((r[..., 3:4], (r[..., 3:4] - r[..., 21:22]) / 0.1), dim=-1), torch.cat((zero, zero), dim=-1), torch.cat((zero, zero), dim=-1)]
        axes = [hand_axes, object_axes, cabinet_axes]
        axis_changes = [zero6, (object_axes - old_object_axes) / self.pose_delta_scale, zero6]
        roles = [torch.cat((one, zero, zero), dim=-1), torch.cat((zero, one, zero), dim=-1), torch.cat((zero, zero, one), dim=-1)]
        nodes = []
        for i in range(3):
            features = torch.cat((positions[i] / self.scene_scale, to_local(positions[i] - g, s, c) / self.scene_scale, velocities[i], apertures[i], axes[i], axis_changes[i], roles[i]), dim=-1)
            nodes.append(self.node_encoder(features))
        updated = []
        for i in range(3):
            messages = []
            for j in range(3):
                if i != j:
                    delta = positions[j] - positions[i]
                    distance = torch.sqrt((delta * delta).sum(dim=-1, keepdim=True) + 1.0e-12) / self.edge_scale
                    edge = torch.cat((nodes[i], nodes[j], to_local(delta, s, c) / self.edge_scale, delta / self.edge_scale, distance), dim=-1)
                    messages.append(self.message_encoder(edge))
            aggregate = torch.stack(messages, dim=-2).mean(dim=-2)
            updated.append(nodes[i] + self.node_update(torch.cat((nodes[i], aggregate), dim=-1)))
        bypass = torch.cat((to_local(h - o, s, c) / self.contact_scale, to_local(g - o, s, c) / self.scene_scale, h / self.scene_scale, torch.stack((s, c), dim=-1), r[..., 3:4], r[..., 21:22]), dim=-1)
        return torch.cat((updated[0], updated[1], updated[2], bypass), dim=-1)

    def training_targets(self, support_view, window_index):
        labels = np.zeros((len(window_index), 16, 6), dtype=np.float32)
        for index, pair in enumerate(window_index):
            episode, t = pair
            obs = support_view[episode]['obs']
            length = len(support_view[episode]['actions'])
            valid = min(16, length - t)
            start = obs[t]
            sc = np.asarray(start[39:41], dtype=np.float64)
            norm = float(np.sqrt((sc * sc).sum()))
            if norm > 1.0e-6:
                s, c = float(sc[0] / norm), float(sc[1] / norm)
            else:
                s, c = 0.0, 1.0
            # Labels are next states paired with each valid native action, all
            # expressed in the one causal chunk-start cabinet orientation.
            future = obs[t + 1:t + valid + 1]
            contact = numpy_local(future[:, :3] - future[:, 4:7], s, c) / self.contact_scale
            goal_error = numpy_local(start[36:39] - future[:, 4:7], s, c) / self.scene_scale
            labels[index, :valid, :3] = contact
            labels[index, :valid, 3:] = goal_error
        return {'future_geometry': torch.from_numpy(labels)}

    def diffusion_targets(self, encoded_actions, targets, causal_context):
        return torch.cat((encoded_actions, targets['future_geometry']), dim=-1)

    def training_loss(self, output, batch, common_diffusion_state):
        error = (output[..., 4:] - common_diffusion_state['noise'][..., 4:]) ** 2
        mask = batch['mask']
        loss = (error * mask.unsqueeze(-1)).sum() / (mask.sum() * 6.0).clamp_min(1.0)
        return self.aux_weight * loss, {'geometry_epsilon_loss': loss.detach()}


def build_design(common_spec, config):
    return RelationalJointDesign(common_spec, config)
