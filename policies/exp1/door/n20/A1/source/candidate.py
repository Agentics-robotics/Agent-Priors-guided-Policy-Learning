import torch
from experiment1.contracts import CandidateDesign


def yaw_pair(raw):
    s = raw[..., 39]
    c = raw[..., 40]
    n = torch.sqrt(s * s + c * c)
    sn = s / n.clamp_min(1.0e-8)
    cn = c / n.clamp_min(1.0e-8)
    return torch.where(n > 1.0e-6, sn, torch.zeros_like(sn)), torch.where(n > 1.0e-6, cn, torch.ones_like(cn))


def to_local(v, s, c):
    return torch.stack((c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1], v[..., 2]), dim=-1)


def to_world(v, s, c):
    return torch.stack((c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]), dim=-1)


def pose_columns(q, s, c):
    q = q / torch.sqrt((q * q).sum(dim=-1, keepdim=True)).clamp_min(1.0e-8)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    col_y = torch.stack((2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)), dim=-1)
    col_z = torch.stack((2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)), dim=-1)
    return torch.cat((to_local(col_y, s, c), to_local(col_z, s, c)), dim=-1)


class CabinetCanonicalDesign(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.contact_scale = float(config['contact_scale_m'])
        self.scene_scale = float(config['scene_scale_m'])
        self.delta_scale = float(config['step_displacement_scale_m'])
        self.gap_delta_scale = float(config['aperture_delta_scale'])

    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(44)

    def condition(self, causal_history, causal_context):
        r = causal_context['raw_history']
        s, c = yaw_pair(r)
        h, o, g = r[..., :3], r[..., 4:7], r[..., 36:39]
        hp, op = r[..., 18:21], r[..., 22:25]
        gap = torch.cat((r[..., 3:4], r[..., 21:22], (r[..., 3:4] - r[..., 21:22]) / self.gap_delta_scale), dim=-1)
        return torch.cat((
            to_local(h - o, s, c) / self.contact_scale,
            to_local(g - o, s, c) / self.scene_scale,
            to_local(hp - op, s, c) / self.contact_scale,
            to_local(g - op, s, c) / self.scene_scale,
            to_local(h - hp, s, c) / self.delta_scale,
            to_local(o - op, s, c) / self.delta_scale,
            gap,
            pose_columns(r[..., 7:11], s, c),
            pose_columns(r[..., 25:29], s, c),
            h / self.scene_scale, o / self.scene_scale, g / self.scene_scale,
            torch.stack((s, c), dim=-1)
        ), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        s, c = yaw_pair(chunk_start_context['raw_current'])
        v = to_local(native_actions[..., :3], s.unsqueeze(-1), c.unsqueeze(-1))
        return torch.cat((v, native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        s, c = yaw_pair(chunk_start_context['raw_current'])
        v = to_world(encoded_actions[..., :3], s.unsqueeze(-1), c.unsqueeze(-1))
        return torch.cat((v, encoded_actions[..., 3:4]), dim=-1)


def build_design(common_spec, config):
    return CabinetCanonicalDesign(common_spec, config)
