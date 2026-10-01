import torch
from experiment1.contracts import CandidateDesign


def yaw_pair(raw):
    s, c = raw[..., 39], raw[..., 40]
    n = torch.sqrt(s * s + c * c)
    return torch.where(n > 1.0e-6, s / n.clamp_min(1.0e-8), torch.zeros_like(s)), torch.where(n > 1.0e-6, c / n.clamp_min(1.0e-8), torch.ones_like(c))


def to_local(v, s, c):
    return torch.stack((c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1], v[..., 2]), dim=-1)


def to_world(v, s, c):
    return torch.stack((c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]), dim=-1)


def rotation_chart(q, sh, ch):
    q = q / torch.sqrt((q * q).sum(dim=-1, keepdim=True)).clamp_min(1.0e-8)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # Left multiply by the inverse cabinet-yaw quaternion, xyzw order.
    local = torch.stack((ch * x + sh * y, ch * y - sh * x, ch * z - sh * w, ch * w + sh * z), dim=-1)
    # A deterministic hemisphere representative also handles exactly w=0.
    # At w=0 use the first nonzero vector component for the sign convention.
    anchor = torch.where(local[..., 3] != 0, local[..., 3], torch.where(local[..., 0] != 0, local[..., 0], torch.where(local[..., 1] != 0, local[..., 1], local[..., 2])))
    sign = torch.where(anchor < 0, -torch.ones_like(anchor), torch.ones_like(anchor))
    local = local * sign.unsqueeze(-1)
    # Modified Rodrigues parameters: full orientation in three coordinates.
    # Unit quaternion with w >= 0 implies denominator >= 1, no clipping.
    return local[..., :3] / (1.0 + local[..., 3:4])


class CompactCabinetDesign(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.contact_scale = float(config['contact_scale_m'])
        self.scene_scale = float(config['scene_scale_m'])
        self.delta_scale = float(config['step_displacement_scale_m'])

    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(25)

    def condition(self, causal_history, causal_context):
        r = causal_context['raw_history']
        s, c = yaw_pair(r)
        half_yaw = 0.5 * torch.atan2(s, c)
        sh, ch = torch.sin(half_yaw), torch.cos(half_yaw)
        h, o, g = r[..., :3], r[..., 4:7], r[..., 36:39]
        hp, op = r[..., 18:21], r[..., 22:25]
        return torch.cat((
            to_local(h - o, s, c) / self.contact_scale,
            to_local(g - o, s, c) / self.scene_scale,
            to_local(h - hp, s, c) / self.delta_scale,
            to_local(o - op, s, c) / self.delta_scale,
            r[..., 3:4], r[..., 21:22],
            rotation_chart(r[..., 7:11], sh, ch),
            rotation_chart(r[..., 25:29], sh, ch),
            h / self.scene_scale,
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
    return CompactCabinetDesign(common_spec, config)
