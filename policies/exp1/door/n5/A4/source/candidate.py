import torch
from experiment1.contracts import CandidateDesign


def yaw_pair(raw):
    s = raw[..., 39]
    c = raw[..., 40]
    length = torch.sqrt(s * s + c * c).clamp_min(1e-8)
    valid = (s * s + c * c) > 1e-12
    return torch.where(valid, s / length, torch.zeros_like(s)), torch.where(valid, c / length, torch.ones_like(c))


def local_vector(v, s, c):
    return torch.stack((c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1], v[..., 2]), dim=-1)


def world_vector(v, s, c):
    return torch.stack((c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]), dim=-1)


def attitude(q, s, c):
    q = q / torch.sqrt((q * q).sum(dim=-1, keepdim=True)).clamp_min(1e-8)
    x, y, z, w = q.unbind(dim=-1)
    a = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)), dim=-1)
    b = torch.stack((2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)), dim=-1)
    return torch.cat((local_vector(a, s, c), local_vector(b, s, c)), dim=-1)


def length(v):
    return torch.sqrt((v * v).sum(dim=-1, keepdim=True) + 1e-12)


class CompactCabinetChart(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.contact_scale = float(config['contact_scale_m'])
        self.world_scale = float(config['world_scale_m'])
        self.motion_scale = float(config['motion_scale_m'])

    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(34)

    def condition(self, causal_history, causal_context):
        raw = causal_context['raw_history']
        s, c = yaw_pair(raw)
        h, o, g = raw[..., 0:3], raw[..., 4:7], raw[..., 36:39]
        ph, po = raw[..., 18:21], raw[..., 22:25]
        rel = h - o
        return torch.cat((
            local_vector(rel, s, c) / self.contact_scale,
            local_vector(g - o, s, c) / self.world_scale,
            local_vector(h - ph, s, c) / self.motion_scale,
            local_vector(o - po, s, c) / self.motion_scale,
            raw[..., 3:4], raw[..., 21:22],
            attitude(raw[..., 7:11], s, c),
            attitude(raw[..., 25:29], s, c),
            h / self.world_scale,
            torch.stack((s, c), dim=-1),
            length(rel) / self.contact_scale,
            length(rel[..., 0:2]) / self.contact_scale,
            length(g - o) / self.world_scale
        ), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        s, c = yaw_pair(chunk_start_context['raw_current'])
        xyz = local_vector(native_actions[..., :3], s[:, None], c[:, None])
        return torch.cat((xyz, native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        s, c = yaw_pair(chunk_start_context['raw_current'])
        xyz = world_vector(encoded_actions[..., :3], s[:, None], c[:, None])
        return torch.cat((xyz, encoded_actions[..., 3:4]), dim=-1)


def build_design(common_spec, config):
    return CompactCabinetChart(common_spec, config)
