import numpy as np
import torch
from experiment1.contracts import CandidateDesign


class InteractionBearingFrame(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.transfer_margin_m = float(config['fallback_transfer_margin_m'])
        self.fit_episode_count = 0

    def fit_support(self, support_view, common_spec):
        margins = []
        for episode in support_view:
            obs = np.asarray(episode['obs'][:-1], dtype=np.float32)
            actions = np.asarray(episode['actions'], dtype=np.float32)
            moving = (actions[:, 1] > 0.5) & (actions[:, 3] > 0.5) & (obs[:, 6] > 0.05)
            indices = np.flatnonzero(moving)
            if len(indices) > 0:
                i = int(indices[0])
                margins.append(float(obs[i, 6] - obs[i, 41] - obs[i, 44]))
        if len(margins) > 0:
            self.transfer_margin_m = float(np.median(np.asarray(margins, dtype=np.float64)))
        self.fit_episode_count = len(support_view)

    def deployment_state_dict(self):
        return {'transfer_margin_m': self.transfer_margin_m, 'fit_episode_count': self.fit_episode_count}

    def load_deployment_state_dict(self, state):
        self.transfer_margin_m = float(state['transfer_margin_m'])
        self.fit_episode_count = int(state['fit_episode_count'])

    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(100)
        self.approach_encoder = torch.nn.Sequential(
            torch.nn.Linear(21, 64), torch.nn.SiLU(), torch.nn.Linear(64, 16))
        self.lift_encoder = torch.nn.Sequential(
            torch.nn.Linear(22, 64), torch.nn.SiLU(), torch.nn.Linear(64, 16))
        self.transport_encoder = torch.nn.Sequential(
            torch.nn.Linear(25, 64), torch.nn.SiLU(), torch.nn.Linear(64, 16))

    def frame_info(self, context):
        r = context['raw_current']
        dx = r[:, 36:37] - r[:, 4:5]
        dy = r[:, 37:38] - r[:, 5:6]
        length = torch.sqrt(dx * dx + dy * dy)
        safe = length.clamp_min(1.0e-6)
        gx = torch.where(length > 1.0e-6, dx / safe, torch.zeros_like(dx))
        gy = torch.where(length > 1.0e-6, dy / safe, torch.ones_like(dy))
        ax = 0.5 * ((r[:, 4:5] - r[:, 0:1]) + (r[:, 22:23] - r[:, 18:19]))
        ay = 0.5 * ((r[:, 5:6] - r[:, 1:2]) + (r[:, 23:24] - r[:, 19:20]))
        distance_squared = ax * ax + ay * ay
        distance_fourth = distance_squared * distance_squared
        far = distance_fourth / (distance_fourth + self.design_config['approach_alignment_distance_m'] ** 4)
        gap = torch.linalg.vector_norm(r[:, 4:7] - r[:, 0:3], dim=-1, keepdim=True)
        closed = torch.sigmoid((self.design_config['closed_aperture_center'] - r[:, 3:4]) /
                               self.design_config['aperture_band'])
        proximity = torch.sigmoid((self.design_config['grasp_distance_m'] - gap) /
                                  self.design_config['grasp_distance_band_m'])
        mix = (1.0 - closed * proximity) * far
        angle_delta = torch.atan2(ax * gy - ay * gx, ax * gx + ay * gy)
        angle = torch.atan2(gx, gy) + mix * angle_delta
        fx, fy = torch.sin(angle), torch.cos(angle)
        return fx.unsqueeze(1), fy.unsqueeze(1), mix.unsqueeze(1)

    def frame(self, context):
        fx, fy, mix = self.frame_info(context)
        return fx, fy

    def rotate(self, vector, fx, fy):
        return torch.cat((fy * vector[..., 0:1] - fx * vector[..., 1:2],
                          fx * vector[..., 0:1] + fy * vector[..., 1:2],
                          vector[..., 2:3]), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        fx, fy = self.frame(chunk_start_context)
        xyz = self.rotate(native_actions[..., :3], fx, fy)
        return torch.cat((xyz, native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        fx, fy = self.frame(chunk_start_context)
        x = fy * encoded_actions[..., 0:1] + fx * encoded_actions[..., 1:2]
        y = -fx * encoded_actions[..., 0:1] + fy * encoded_actions[..., 1:2]
        return torch.cat((x, y, encoded_actions[..., 2:4]), dim=-1)

    def face_coordinates(self, point, wall, half):
        s = self.design_config['position_scale_m']
        delta = point - wall
        return torch.cat(((delta + half) / s, (half - delta) / s), dim=-1)

    def condition(self, causal_history, causal_context):
        r = causal_context['raw_history']
        fx, fy, mix = self.frame_info(causal_context)
        h, o, g = r[..., 0:3], r[..., 4:7], r[..., 36:39]
        hp, op = r[..., 18:21], r[..., 22:25]
        w, half = r[..., 39:42], r[..., 42:45]
        q, qp = r[..., 7:11], r[..., 25:29]
        s, v = self.design_config['position_scale_m'], self.design_config['motion_scale_m']
        grip = 2.0 * r[..., 3:4] - 1.0
        dg = 10.0 * (r[..., 3:4] - r[..., 21:22])
        loh = self.rotate(o - h, fx, fy) / s
        lhw = self.rotate(h - w, fx, fy) / s
        low = self.rotate(o - w, fx, fy) / s
        hv = self.rotate(h - hp, fx, fy) / v
        ov = self.rotate(o - op, fx, fy) / v
        hf = self.face_coordinates(h, w, half)
        of = self.face_coordinates(o, w, half)
        top = w[..., 2:3] + half[..., 2:3]
        hz, oz = h[..., 2:3], o[..., 2:3]
        gap = torch.linalg.vector_norm(o - h, dim=-1, keepdim=True)
        closed = torch.sigmoid((self.design_config['closed_aperture_center'] - r[..., 3:4]) /
                               self.design_config['aperture_band'])
        proximity = torch.sigmoid((self.design_config['grasp_distance_m'] - gap) /
                                  self.design_config['grasp_distance_band_m'])
        held = closed * proximity
        clear = torch.sigmoid((oz - top - self.transfer_margin_m) /
                              self.design_config['clearance_band_m'])
        beyond = torch.sigmoid((o[..., 1:2] - w[..., 1:2] - half[..., 1:2] -
                                self.design_config['far_side_margin_m']) /
                               self.design_config['far_side_band_m'])
        transport = 1.0 - (1.0 - clear) * (1.0 - beyond)
        gates = torch.cat((1.0 - held, held * (1.0 - transport), held * transport), dim=-1)
        core = torch.cat((lhw, loh, low, hv, ov, grip, q, qp,
                          h / self.design_config['world_scale_m'],
                          w / self.design_config['world_scale_m'], half / s,
                          fx * torch.ones_like(grip), fy * torch.ones_like(grip), hf, of, dg,
                          mix * torch.ones_like(grip)), dim=-1)
        approach = torch.cat((loh, hv, grip, dg, hz / s, oz / s, q, hf,
                              torch.linalg.vector_norm(o[..., :2] - h[..., :2], dim=-1, keepdim=True) / s), dim=-1)
        lift = torch.cat((loh, ov, hv, (oz - top) / s, (hz - top) / s, grip, of, q), dim=-1)
        carry = torch.cat((self.rotate(g - o, fx, fy) / s,
                           self.rotate(g - h, fx, fy) / s,
                           self.rotate(g - w, fx, fy) / s,
                           ov, hv, loh, of, grip), dim=-1)
        return torch.cat((core,
                          gates[..., 0:1] * self.approach_encoder(approach),
                          gates[..., 1:2] * self.lift_encoder(lift),
                          gates[..., 2:3] * self.transport_encoder(carry), gates), dim=-1)


def build_design(common_spec, config):
    return InteractionBearingFrame(common_spec, config)
