import numpy as np
import torch
from experiment1.contracts import CandidateDesign


class ToolFrameEffectDiffusion(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.world_origin = [0.0, 0.0, 0.0]

    def fit_support(self, support_view, common_spec):
        self.world_origin = np.stack([np.asarray(e['obs'][0, 11:14], dtype=np.float64)
                                      for e in support_view]).mean(axis=0).tolist()

    def deployment_state_dict(self):
        return {'world_origin': list(self.world_origin)}

    def load_deployment_state_dict(self, state):
        self.world_origin = list(state['world_origin'])

    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(58, diffusion_action_dim=14)

    def normalized_quat(self, q):
        return q / torch.sqrt((q * q).sum(dim=-1, keepdim=True).clamp_min(1.0e-12))

    def quat_features(self, q):
        q = self.normalized_quat(q)
        q = torch.where(q[..., 3:4] < 0.0, -q, q)
        return torch.cat((q[..., :3] / self.design_config['rotation_scale'], q[..., 3:4]), dim=-1)

    def chunk_axes(self, context):
        q = self.normalized_quat(context['raw_current'][:, 7:11])
        x, y, z, w = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
        # Project the tool's body x axis into the horizontal plane.
        vx = 1.0 - 2.0 * (y * y + z * z)
        vy = 2.0 * (x * y + z * w)
        length = torch.sqrt((vx * vx + vy * vy).clamp_min(1.0e-12))
        c = torch.where(length > 1.0e-5, vx / length, torch.ones_like(vx))
        s = torch.where(length > 1.0e-5, vy / length, torch.zeros_like(vy))
        return c, s

    def to_local(self, v, c, s):
        c, s = c.unsqueeze(1), s.unsqueeze(1)
        return torch.cat((c * v[..., 0:1] + s * v[..., 1:2],
                          -s * v[..., 0:1] + c * v[..., 1:2], v[..., 2:3]), dim=-1)

    def to_world(self, v, c, s):
        c, s = c.unsqueeze(1), s.unsqueeze(1)
        return torch.cat((c * v[..., 0:1] - s * v[..., 1:2],
                          s * v[..., 0:1] + c * v[..., 1:2], v[..., 2:3]), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        c, s = self.chunk_axes(chunk_start_context)
        return torch.cat((self.to_local(native_actions[..., :3], c, s), native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        c, s = self.chunk_axes(chunk_start_context)
        return torch.cat((self.to_world(encoded_actions[..., :3], c, s), encoded_actions[..., 3:4]), dim=-1)

    def pose_features(self, r, base, origin, c, s):
        h, stick = r[..., base:base + 3], r[..., base + 4:base + 7]
        p, g = r[..., base + 11:base + 14], r[..., 36:39]
        return torch.cat((self.to_local(h - stick, c, s) / self.design_config['local_scale_m'],
                          self.to_local(p - stick, c, s) / self.design_config['scene_scale_m'],
                          self.to_local(g - p, c, s) / self.design_config['scene_scale_m'],
                          (stick - origin) / self.design_config['scene_scale_m'],
                          r[..., base + 3:base + 4],
                          self.quat_features(r[..., base + 7:base + 11]),
                          self.quat_features(r[..., base + 14:base + 18])), dim=-1)

    def condition(self, causal_history, causal_context):
        r = causal_context['raw_history']
        origin = torch.as_tensor(self.world_origin, dtype=r.dtype, device=r.device)
        c, s = self.chunk_axes(causal_context)
        motion = torch.cat((self.to_local(r[..., 0:3] - r[..., 18:21], c, s),
                            self.to_local(r[..., 4:7] - r[..., 22:25], c, s),
                            self.to_local(r[..., 11:14] - r[..., 29:32], c, s)), dim=-1)
        heights = torch.cat((r[..., 2:3], r[..., 6:7], r[..., 13:14], r[..., 38:39]), dim=-1)
        frame = torch.cat((c, s), dim=-1).unsqueeze(1).expand(-1, r.shape[1], -1)
        return torch.cat((self.pose_features(r, 0, origin, c, s),
                          self.pose_features(r, 18, origin, c, s),
                          motion / self.design_config['motion_scale_m'],
                          heights / self.design_config['height_scale_m'],
                          (r[..., 3:4] - r[..., 21:22]) / self.design_config['aperture_change_scale'],
                          frame), dim=-1)

    def training_targets(self, support_view, window_index):
        effects = np.zeros((len(window_index), 16, 10), dtype=np.float32)
        for i, (episode_index, t) in enumerate(window_index):
            episode = support_view[episode_index]
            obs = episode['obs']
            count = min(16, len(episode['actions']) - t)
            for k in range(count):
                future = obs[t + k + 1]
                now = obs[t]
                effects[i, k] = np.concatenate((future[0:3] - now[0:3],
                                                 future[4:7] - now[4:7],
                                                 future[11:14] - now[11:14],
                                                 future[3:4] - now[3:4]))
        return {'effect_world': effects}

    def diffusion_targets(self, encoded_actions, targets, causal_context):
        effect = targets['effect_world']
        c, s = self.chunk_axes(causal_context)
        local = torch.cat((self.to_local(effect[..., 0:3], c, s) / self.design_config['effect_scale_m'],
                           self.to_local(effect[..., 3:6], c, s) / self.design_config['effect_scale_m'],
                           self.to_local(effect[..., 6:9], c, s) / self.design_config['effect_scale_m'],
                           effect[..., 9:10] / self.design_config['effect_aperture_scale']), dim=-1)
        return torch.cat((encoded_actions, local), dim=-1)

    def training_loss(self, output, batch, common_diffusion_state):
        error = (output[..., 4:] - common_diffusion_state['noise'][..., 4:]) ** 2
        mask = batch['mask']
        loss = (error * mask.unsqueeze(-1)).sum() / (10.0 * mask.sum()).clamp_min(1.0)
        return self.design_config['effect_loss_weight'] * loss, {'effect_epsilon_mse': loss.detach()}


def build_design(common_spec, config):
    return ToolFrameEffectDiffusion(common_spec, config)
