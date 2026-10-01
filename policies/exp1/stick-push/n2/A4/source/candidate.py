import numpy as np
import torch
from experiment1.contracts import CandidateDesign


class CompactInteractionDesign(CandidateDesign):
    def build_modules(self, common_dp_factory):
        cfg = self.design_config
        width = cfg['edge_width']
        hidden = cfg['edge_hidden']
        self.condition_dim = 52 + 3 * width
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(24, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, width), torch.nn.SiLU())
        self.interaction_gate = torch.nn.Sequential(
            torch.nn.Linear(14, cfg['gate_hidden']), torch.nn.SiLU(),
            torch.nn.Linear(cfg['gate_hidden'], 3))
        self.motion_head = torch.nn.Sequential(
            torch.nn.Linear(self.condition_dim, cfg['motion_hidden']), torch.nn.SiLU(),
            torch.nn.Linear(cfg['motion_hidden'], 160))
        self.dp = common_dp_factory(self.condition_dim)

    def rooted_pose(self, raw, base):
        h = raw[..., base:base + 3]
        s = raw[..., base + 4:base + 7]
        p = raw[..., base + 11:base + 14]
        origin = raw.new_tensor(self.design_config['world_origin_m'])
        qorigin = raw.new_tensor([0.0, 0.0, 0.0, 1.0])
        qscale = raw.new_tensor([0.2, 0.2, 0.2, 1.0])
        return torch.cat([
            (h - origin) / raw.new_tensor([0.4, 0.4, 0.2]),
            raw[..., base + 3:base + 4],
            (s - h) / 0.1,
            (raw[..., base + 7:base + 11] - qorigin) / qscale,
            (p - h) / 0.3,
            (raw[..., base + 14:base + 18] - qorigin) / qscale
        ], dim=-1)

    def edge_features(self, src, dst, old_src, old_dst, src_q, dst_q, aperture, change, role):
        delta = dst - src
        old_delta = old_dst - old_src
        tag = torch.zeros_like(delta) + delta.new_tensor(role)
        return torch.cat([delta / 0.1, old_delta / 0.1,
                          (delta - old_delta) / 0.01,
                          src[..., 2:3] / 0.1, dst[..., 2:3] / 0.1,
                          aperture, change, src_q, dst_q, tag], dim=-1)

    def condition(self, causal_history, causal_context):
        raw = causal_context['raw_history']
        h, s, p, g = raw[..., 0:3], raw[..., 4:7], raw[..., 11:14], raw[..., 36:39]
        hp, sp, pp = raw[..., 18:21], raw[..., 22:25], raw[..., 29:32]
        aperture = raw[..., 3:4]
        da = aperture - raw[..., 21:22]
        vh, vs, vp = (h - hp) / 0.01, (s - sp) / 0.01, (p - pp) / 0.01
        qs, qp = raw[..., 7:11], raw[..., 14:18]
        absent_q = torch.zeros_like(qs)
        edges = torch.stack([
            self.edge_features(h, s, hp, sp, absent_q, qs, aperture, da, [1.0, 0.0, 0.0]),
            self.edge_features(s, p, sp, pp, qs, qp, aperture, da, [0.0, 1.0, 0.0]),
            self.edge_features(p, g, pp, g, qp, absent_q, aperture, da, [0.0, 0.0, 1.0])
        ], dim=-2)
        gate_input = torch.cat([aperture, da, (s - h) / 0.1,
                                h[..., 2:3] / 0.1, s[..., 2:3] / 0.1,
                                p[..., 2:3] / 0.1, vh, vs], dim=-1)
        weights = torch.softmax(self.interaction_gate(gate_input), dim=-1)
        learned = 3.0 * weights.unsqueeze(-1) * self.edge_encoder(edges)
        bypass = torch.cat([self.rooted_pose(raw, 0), self.rooted_pose(raw, 18),
                            (g - h) / 0.3, vh, vs, vp, da], dim=-1)
        return torch.cat([bypass, learned.flatten(start_dim=-2), weights], dim=-1)

    def training_targets(self, support_view, window_index):
        motion = np.zeros((len(window_index), 16, 10), dtype=np.float32)
        valid = np.zeros((len(window_index), 16), dtype=np.float32)
        scale = float(self.design_config['future_displacement_scale_m'])
        for row, (episode, t) in enumerate(window_index):
            obs = np.asarray(support_view[episode]['obs'], dtype=np.float32)
            length = len(support_view[episode]['actions'])
            current = obs[t]
            for j in range(min(16, length - t)):
                future = obs[t + j + 1]
                motion[row, j] = np.concatenate([
                    (future[0:3] - current[0:3]) / scale,
                    (future[4:7] - current[4:7]) / scale,
                    (future[11:14] - current[11:14]) / scale,
                    future[3:4] - current[3:4]
                ])
                valid[row, j] = 1.0
        return {'future_motion': motion, 'future_motion_valid': valid}

    def training_loss(self, output, batch, common_diffusion_state):
        condition = self.condition(batch['history'], batch['context'])
        predicted = self.motion_head(condition[:, -1]).reshape(-1, 16, 10)
        mask = batch['mask'] * batch['targets']['future_motion_valid']
        squared = (predicted - batch['targets']['future_motion']).square()
        mse = (squared * mask.unsqueeze(-1)).sum() / (mask.sum() * 10.0).clamp_min(1.0)
        return self.design_config['motion_loss_weight'] * mse, {'future_motion_mse': mse.detach()}


def build_design(common_spec, config):
    return CompactInteractionDesign(common_spec, config)
