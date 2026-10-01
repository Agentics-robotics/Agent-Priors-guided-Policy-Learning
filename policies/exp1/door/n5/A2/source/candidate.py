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


def world_vector(v, s, c):
    return torch.stack((c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]), dim=-1)


def unit_xy(v):
    n = torch.sqrt((v * v).sum(dim=-1, keepdim=True)).clamp_min(1e-8)
    good = (v * v).sum(dim=-1, keepdim=True) > 1e-12
    fallback = torch.stack((torch.ones_like(v[..., 0]), torch.zeros_like(v[..., 0])), dim=-1)
    return torch.where(good, v / n, fallback)


def distance(v):
    return torch.sqrt((v * v).sum(dim=-1, keepdim=True) + 1e-12)


def attitude(q, s, c):
    q = q / torch.sqrt((q * q).sum(dim=-1, keepdim=True)).clamp_min(1e-8)
    x, y, z, w = q.unbind(dim=-1)
    a = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)), dim=-1)
    b = torch.stack((2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)), dim=-1)
    return torch.cat((local_vector(a, s, c), local_vector(b, s, c)), dim=-1)


class HingeExperts(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.circle_state = {}
        self.contact_scale = float(config['contact_scale_m'])
        self.world_scale = float(config['world_scale_m'])
        self.motion_scale = float(config['motion_scale_m'])

    def fit_support(self, support_view, common_spec):
        points = []
        for episode in support_view:
            raw = np.asarray(episode['obs'][:-1], dtype=np.float64)
            d = raw[:, 4:7] - raw[:, 36:39]
            s, c = raw[:, 39], raw[:, 40]
            norm = np.maximum(np.sqrt(s * s + c * c), 1e-8)
            s, c = s / norm, c / norm
            points.append(np.stack((c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]), axis=-1))
        p = np.concatenate(points, axis=0)
        system = np.concatenate((2.0 * p, np.ones((len(p), 1))), axis=1)
        rhs = (p * p).sum(axis=1)
        solution = np.linalg.lstsq(system, rhs, rcond=None)[0]
        center = solution[:2]
        radii = np.sqrt(((p - center) ** 2).sum(axis=1))
        radius = float(max(float(radii.mean()), 1e-4))
        self.circle_state = {
            'center_xy_goal_cabinet': center.tolist(),
            'radius_m': radius,
            'radial_fit_rms_m': float(np.sqrt(((radii - radius) ** 2).mean())),
            'count': int(len(p))
        }

    def deployment_state_dict(self):
        return dict(self.circle_state)

    def load_deployment_state_dict(self, state):
        self.circle_state = dict(state)

    def build_modules(self, common_dp_factory):
        self.register_buffer('center_xy', torch.tensor(self.circle_state['center_xy_goal_cabinet'], dtype=torch.float32))
        self.radius = float(self.circle_state['radius_m'])
        n = int(self.design_config['expert_count'])
        hidden = int(self.design_config['expert_hidden'])
        latent = int(self.design_config['expert_latent'])
        gate_hidden = int(self.design_config['gate_hidden'])
        self.dp = common_dp_factory(54 + latent + n)
        self.gate = torch.nn.Sequential(torch.nn.Linear(54, gate_hidden), torch.nn.SiLU(), torch.nn.Linear(gate_hidden, n))
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(54, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, latent))
            for i in range(n)
        ])

    def frame(self, raw):
        s, c = yaw_pair(raw)
        p = local_vector(raw[..., 4:7] - raw[..., 36:39], s, c)[..., :2]
        radial = unit_xy(p - self.center_xy)
        cw = c * radial[..., 0] - s * radial[..., 1]
        sw = s * radial[..., 0] + c * radial[..., 1]
        return sw, cw

    def features(self, raw):
        s, c = yaw_pair(raw)
        h, o, g = raw[..., :3], raw[..., 4:7], raw[..., 36:39]
        ph, po = raw[..., 18:21], raw[..., 22:25]
        radial = local_vector(o - g, s, c)[..., :2] - self.center_xy
        pradial = local_vector(po - g, s, c)[..., :2] - self.center_xy
        u, pu = unit_xy(radial), unit_xy(pradial)
        gu = unit_xy(-self.center_xy)
        sw, cw = self.frame(raw)
        goal_angle = torch.stack((u[..., 0] * gu[1] - u[..., 1] * gu[0], u[..., 0] * gu[0] + u[..., 1] * gu[1]), dim=-1)
        rel = h - o
        return torch.cat((
            local_vector(rel, sw, cw) / self.contact_scale,
            local_vector(ph - po, sw, cw) / self.contact_scale,
            local_vector(g - o, sw, cw) / self.world_scale,
            local_vector(h - ph, sw, cw) / self.motion_scale,
            local_vector(o - po, sw, cw) / self.motion_scale,
            u, pu, goal_angle,
            (distance(radial) - self.radius) / 0.05,
            (distance(pradial) - self.radius) / 0.05,
            h / self.world_scale, o / self.world_scale, g / self.world_scale, ph / self.world_scale,
            torch.stack((s, c), dim=-1), raw[..., 3:4], raw[..., 21:22],
            attitude(raw[..., 7:11], s, c), attitude(raw[..., 25:29], s, c),
            distance(rel) / self.contact_scale, distance(rel[..., :2]) / self.contact_scale,
            distance(g - o) / self.world_scale
        ), dim=-1)

    def condition(self, causal_history, causal_context):
        f = self.features(causal_context['raw_history'])
        weights = torch.softmax(self.gate(f), dim=-1)
        values = torch.stack([expert(f) for expert in self.experts], dim=-2)
        mixture = (weights.unsqueeze(-1) * values).sum(dim=-2)
        return torch.cat((f, mixture, weights), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        sw, cw = self.frame(chunk_start_context['raw_current'])
        v = local_vector(native_actions[..., :3], sw[:, None], cw[:, None])
        return torch.cat((v, native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        sw, cw = self.frame(chunk_start_context['raw_current'])
        v = world_vector(encoded_actions[..., :3], sw[:, None], cw[:, None])
        return torch.cat((v, encoded_actions[..., 3:4]), dim=-1)


def build_design(common_spec, config):
    return HingeExperts(common_spec, config)
