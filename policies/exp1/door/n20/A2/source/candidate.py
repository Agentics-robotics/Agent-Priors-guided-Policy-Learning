import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from experiment1.contracts import CandidateDesign


def yaw_pair(raw):
    s, c = raw[..., 39], raw[..., 40]
    n = torch.sqrt(s * s + c * c)
    return torch.where(n > 1.0e-6, s / n.clamp_min(1.0e-8), torch.zeros_like(s)), torch.where(n > 1.0e-6, c / n.clamp_min(1.0e-8), torch.ones_like(c))


def to_local(v, s, c):
    return torch.stack((c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1], v[..., 2]), dim=-1)


def to_world(v, s, c):
    return torch.stack((c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]), dim=-1)


def pose_columns(q, s, c):
    q = q / torch.sqrt((q * q).sum(dim=-1, keepdim=True)).clamp_min(1.0e-8)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    cy = torch.stack((2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)), dim=-1)
    cz = torch.stack((2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)), dim=-1)
    return torch.cat((to_local(cy, s, c), to_local(cz, s, c)), dim=-1)


def canonical_points(obs):
    v = np.asarray(obs[:, 4:6] - obs[:, 36:38], dtype=np.float64)
    sc = np.asarray(obs[:, 39:41], dtype=np.float64)
    sc = sc / np.maximum(np.sqrt((sc * sc).sum(axis=-1, keepdims=True)), 1.0e-8)
    s, c = sc[:, 0], sc[:, 1]
    return np.stack((c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]), axis=-1)


class ArticulationContactDesign(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.center = [0.0, 0.0]
        self.radius = 1.0
        self.rest = [1.0, 0.0]
        self.fit_points = 0
        self.fit_rmse = 0.0
        self.contact_scale = float(config['contact_scale_m'])
        self.scene_scale = float(config['scene_scale_m'])
        self.delta_scale = float(config['step_displacement_scale_m'])
        self.radius_scale = float(config['radius_error_scale_m'])
        self.stage_weight = float(config['stage_loss_weight'])
        self.approach_xy = float(config['approach_xy_threshold_m'])
        self.approach_height = float(config['approach_height_threshold_m'])
        self.open_displacement = float(config['opening_displacement_threshold_m'])
        self.width = int(config['expert_width'])
        self.embedding = int(config['expert_embedding'])

    def fit_support(self, support_view, common_spec):
        points = np.concatenate([canonical_points(e['obs'][:-1]) for e in support_view], axis=0)
        # Pooled circle in observed goal-relative cabinet coordinates. This is a
        # low-dimensional chart fit, not a dynamics model or trajectory planner.
        a = np.concatenate((2.0 * points, np.ones((len(points), 1), dtype=np.float64)), axis=1)
        b = (points * points).sum(axis=-1)
        solution = np.linalg.lstsq(a, b, rcond=None)[0]
        center = solution[:2]
        radius = float(np.sqrt(max(float(solution[2] + (center * center).sum()), 1.0e-8)))
        initial = np.concatenate([canonical_points(e['obs'][:1]) for e in support_view], axis=0)
        radial = initial - center
        radial = radial / np.maximum(np.sqrt((radial * radial).sum(axis=-1, keepdims=True)), 1.0e-8)
        rest = radial.mean(axis=0)
        rest = rest / max(float(np.sqrt((rest * rest).sum())), 1.0e-8)
        errors = np.sqrt(((points - center) * (points - center)).sum(axis=-1)) - radius
        self.center = center.tolist()
        self.radius = radius
        self.rest = rest.tolist()
        self.fit_points = int(len(points))
        self.fit_rmse = float(np.sqrt((errors * errors).mean()))

    def deployment_state_dict(self):
        return {'center_goal_cabinet_xy': self.center, 'radius_m': self.radius, 'rest_radial_cabinet_xy': self.rest, 'support_state_count': self.fit_points, 'circle_rmse_m': self.fit_rmse}

    def load_deployment_state_dict(self, state):
        self.center = list(state['center_goal_cabinet_xy'])
        self.radius = float(state['radius_m'])
        self.rest = list(state['rest_radial_cabinet_xy'])
        self.fit_points = int(state['support_state_count'])
        self.fit_rmse = float(state['circle_rmse_m'])

    def build_modules(self, common_dp_factory):
        self.register_buffer('chart_center', torch.tensor(self.center, dtype=torch.float32))
        self.register_buffer('chart_rest', torch.tensor(self.rest, dtype=torch.float32))
        self.dp = common_dp_factory(50 + self.embedding + 3)
        self.gate = nn.Sequential(nn.Linear(36, 32), nn.SiLU(), nn.Linear(32, 3))
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(50, self.width), nn.SiLU(), nn.Linear(self.width, self.embedding), nn.SiLU())
            for i in range(3)
        ])

    def chart(self, raw):
        s, c = yaw_pair(raw)
        center_x = raw[..., 36] + c * self.chart_center[0] - s * self.chart_center[1]
        center_y = raw[..., 37] + s * self.chart_center[0] + c * self.chart_center[1]
        dx, dy = raw[..., 4] - center_x, raw[..., 5] - center_y
        radius = torch.sqrt(dx * dx + dy * dy)
        rx = torch.where(radius > 1.0e-6, dx / radius.clamp_min(1.0e-8), c)
        ry = torch.where(radius > 1.0e-6, dy / radius.clamp_min(1.0e-8), s)
        # Frame x = clockwise tangent, y = outward radial, z = world up.
        return -rx, ry, radius

    def geometry(self, r):
        s, c = yaw_pair(r)
        sf, cf, radius = self.chart(r)
        h, o, g = r[..., :3], r[..., 4:7], r[..., 36:39]
        hp, op = r[..., 18:21], r[..., 22:25]
        rx, ry = -sf, cf
        rcx, rcy = c * rx + s * ry, -s * rx + c * ry
        phase = torch.stack((self.chart_rest[0] * rcy - self.chart_rest[1] * rcx, self.chart_rest[0] * rcx + self.chart_rest[1] * rcy), dim=-1)
        cabinet_x = torch.stack((c, s, torch.zeros_like(c)), dim=-1)
        gap = torch.cat((r[..., 3:4], r[..., 21:22], (r[..., 3:4] - r[..., 21:22]) / 0.1), dim=-1)
        d = h - o
        distances = torch.stack((torch.sqrt((d[..., :2] * d[..., :2]).sum(dim=-1) + 1.0e-12) / self.contact_scale, d[..., 2] / self.contact_scale, torch.sqrt(((g - o)[..., :2] * (g - o)[..., :2]).sum(dim=-1) + 1.0e-12) / self.scene_scale), dim=-1)
        return torch.cat((
            to_local(h - o, sf, cf) / self.contact_scale,
            to_local(g - o, sf, cf) / self.scene_scale,
            to_local(hp - op, sf, cf) / self.contact_scale,
            to_local(h - hp, sf, cf) / self.delta_scale,
            to_local(o - op, sf, cf) / self.delta_scale,
            gap, pose_columns(r[..., 7:11], sf, cf), pose_columns(r[..., 25:29], sf, cf),
            to_local(cabinet_x, sf, cf), phase, ((radius - self.radius) / self.radius_scale).unsqueeze(-1),
            h / self.scene_scale, o / self.scene_scale, g / self.scene_scale,
            torch.stack((s, c), dim=-1), distances
        ), dim=-1)

    def condition(self, causal_history, causal_context):
        features = self.geometry(causal_context['raw_history'])
        logits = self.gate(features[..., :36])
        weights = torch.softmax(logits, dim=-1)
        embeddings = torch.stack([expert(features) for expert in self.experts], dim=-2)
        mixture = (weights.unsqueeze(-1) * embeddings).sum(dim=-2)
        return torch.cat((features, mixture, logits), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        s, c, radius = self.chart(chunk_start_context['raw_current'])
        v = to_local(native_actions[..., :3], s.unsqueeze(-1), c.unsqueeze(-1))
        return torch.cat((v, native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        s, c, radius = self.chart(chunk_start_context['raw_current'])
        v = to_world(encoded_actions[..., :3], s.unsqueeze(-1), c.unsqueeze(-1))
        return torch.cat((v, encoded_actions[..., 3:4]), dim=-1)

    def training_targets(self, support_view, window_index):
        labels = []
        for episode, t in window_index:
            obs = support_view[episode]['obs']
            r = obs[t]
            movement = float(np.sqrt(((r[4:7] - obs[0, 4:7]) ** 2).sum()))
            planar = float(np.sqrt(((r[:2] - r[4:6]) ** 2).sum()))
            height = float(r[2] - r[6])
            if movement > self.open_displacement:
                label = 2
            elif planar > self.approach_xy or height > self.approach_height:
                label = 0
            else:
                label = 1
            labels.append(label)
        return {'contact_stage': torch.tensor(labels, dtype=torch.long)}

    def denoise(self, noisy_actions, timesteps, conditioning, causal_context):
        return {'epsilon': self.dp(noisy_actions, timesteps, conditioning), 'stage_logits': conditioning[:, -1, -3:]}

    def training_loss(self, output, batch, common_diffusion_state):
        losses = F.cross_entropy(output['stage_logits'], batch['targets']['contact_stage'].long(), reduction='none')
        valid = batch['mask'][:, 0]
        stage_loss = (losses * valid).sum() / valid.sum().clamp_min(1.0)
        return self.stage_weight * stage_loss, {'contact_stage_ce': stage_loss.detach()}


def build_design(common_spec, config):
    return ArticulationContactDesign(common_spec, config)
