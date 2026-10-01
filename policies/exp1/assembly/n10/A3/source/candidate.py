import numpy as np
import torch
from experiment1.contracts import CandidateDesign


class TransportFrameJointMotion(CandidateDesign):
    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(42, diffusion_action_dim=10)

    def heading(self, context):
        r = context['raw_current']
        x = r[:, 36] - r[:, 39]
        y = r[:, 37] - r[:, 40] + self.design_config['heading_world_y_bias_m']
        degenerate = x.square() + y.square() < 1.0e-12
        x = torch.where(degenerate, torch.zeros_like(x), x)
        y = torch.where(degenerate, torch.ones_like(y), y)
        length = torch.sqrt(x.square() + y.square())
        return torch.stack((x / length, y / length), dim=-1)

    def to_local(self, vector, heading):
        cs, sn = heading[:, 0:1], heading[:, 1:2]
        return torch.stack((cs * vector[..., 0] + sn * vector[..., 1],
                            -sn * vector[..., 0] + cs * vector[..., 1],
                            vector[..., 2]), dim=-1)

    def to_world(self, vector, heading):
        cs, sn = heading[:, 0:1], heading[:, 1:2]
        return torch.stack((cs * vector[..., 0] - sn * vector[..., 1],
                            sn * vector[..., 0] + cs * vector[..., 1],
                            vector[..., 2]), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        frame = self.heading(chunk_start_context)
        return torch.cat((self.to_local(native_actions[..., 0:3], frame),
                          native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        frame = self.heading(chunk_start_context)
        return torch.cat((self.to_world(encoded_actions[..., 0:3], frame),
                          encoded_actions[..., 3:4]), dim=-1)

    def condition(self, causal_history, causal_context):
        r = causal_context['raw_history']
        frame = self.heading(causal_context)
        h, o = r[..., 0:3], r[..., 4:7]
        g, c = r[..., 36:39], r[..., 39:42]
        grasp = self.to_local(o - h, frame)
        lever = self.to_local(c - o, frame)
        seating = self.to_local(g - c, frame)
        s = self.design_config['relation_scale_m']
        fine = self.design_config['fine_scale_m']
        motion = self.common_spec['action_schema']['native_xyz_scale_m']
        origin = h.new_tensor(self.design_config['world_origin'])
        distances = torch.cat((
            torch.sqrt(grasp[..., 0:2].square().sum(-1, keepdim=True) + 1.0e-12),
            torch.sqrt(seating[..., 0:2].square().sum(-1, keepdim=True) + 1.0e-12)), dim=-1)
        heading_history = torch.zeros_like(r[..., 0:2]) + frame[:, None, :]
        return torch.cat((
            grasp / s, lever / s, seating / s,
            self.to_local(h - origin, frame) / self.design_config['world_scale_m'],
            2.0 * r[..., 3:4] - 1.0,
            r[..., 7:11], r[..., 25:29],
            self.to_local(h - r[..., 18:21], frame) / motion,
            self.to_local(o - r[..., 22:25], frame) / motion,
            r[..., 3:4] - r[..., 21:22],
            torch.cat((h[..., 2:3], o[..., 2:3], c[..., 2:3], g[..., 2:3]), dim=-1) / s,
            distances / s, heading_history,
            torch.asinh(grasp / fine), torch.asinh(seating / fine)
        ), dim=-1)

    def training_targets(self, support_view, window_index):
        displacement = np.zeros((len(window_index), 16, 6), dtype=np.float32)
        for row, (episode_id, t) in enumerate(window_index):
            episode = support_view[episode_id]
            obs = episode['obs']
            valid = min(16, len(episode['actions']) - t)
            displacement[row, :valid, 0:3] = obs[t + 1:t + 1 + valid, 0:3] - obs[t, 0:3]
            displacement[row, :valid, 3:6] = obs[t + 1:t + 1 + valid, 39:42] - obs[t, 39:42]
        return {'future_displacement_world': displacement}

    def diffusion_targets(self, encoded_actions, targets, causal_context):
        frame = self.heading(causal_context)
        displacement = targets['future_displacement_world']
        scale = self.design_config['future_displacement_scale_m']
        hand = self.to_local(displacement[..., 0:3], frame) / scale
        ring = self.to_local(displacement[..., 3:6], frame) / scale
        return torch.cat((encoded_actions, hand, ring), dim=-1)

    def training_loss(self, output, batch, common_diffusion_state):
        error = output[..., 4:10] - common_diffusion_state['noise'][..., 4:10]
        mask = batch['mask']
        auxiliary = (error.square() * mask[..., None]).sum() / (6.0 * mask.sum()).clamp_min(1.0)
        return self.design_config['motion_loss_weight'] * auxiliary, {'motion_epsilon': auxiliary.detach()}


def build_design(common_spec, config):
    return TransportFrameJointMotion(common_spec, config)
