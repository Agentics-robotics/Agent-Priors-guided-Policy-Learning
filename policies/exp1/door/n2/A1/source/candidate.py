import torch
from experiment1.contracts import CandidateDesign


def cabinet_basis(raw):
    s0 = raw[..., 39]
    c0 = raw[..., 40]
    n = torch.sqrt(s0 * s0 + c0 * c0)
    safe = n.clamp_min(1.0e-8)
    s = torch.where(n > 1.0e-8, s0 / safe, torch.zeros_like(s0))
    c = torch.where(n > 1.0e-8, c0 / safe, torch.ones_like(c0))
    return s, c


def to_local(v, s, c):
    return torch.stack((c * v[..., 0] + s * v[..., 1],
                        -s * v[..., 0] + c * v[..., 1], v[..., 2]), dim=-1)


def to_world(v, s, c):
    return torch.stack((c * v[..., 0] - s * v[..., 1],
                        s * v[..., 0] + c * v[..., 1], v[..., 2]), dim=-1)


def rotation_columns(q):
    q = q / torch.sqrt((q * q).sum(dim=-1, keepdim=True)).clamp_min(1.0e-8)
    x, y, z, w = q.unbind(dim=-1)
    first = torch.stack((1.0 - 2.0 * (y * y + z * z),
                         2.0 * (x * y + z * w), 2.0 * (x * z - y * w)), dim=-1)
    third = torch.stack((2.0 * (x * z + y * w), 2.0 * (y * z - x * w),
                         1.0 - 2.0 * (x * x + y * y)), dim=-1)
    return first, third


class CabinetGeometry(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.contact_scale = float(config["contact_scale_m"])
        self.goal_scale = float(config["goal_scale_m"])
        self.motion_scale = float(config["motion_scale_m"])
        self.world_scale = float(config["world_scale_m"])

    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(46)

    def condition(self, causal_history, causal_context):
        raw = causal_context["raw_history"]
        s, c = cabinet_basis(raw)
        h, o, g = raw[..., 0:3], raw[..., 4:7], raw[..., 36:39]
        ph, po = raw[..., 18:21], raw[..., 22:25]
        hc = to_local(h - o, s, c) / self.contact_scale
        gc = to_local(g - o, s, c) / self.goal_scale
        pc = to_local(ph - po, s, c) / self.contact_scale
        dh = to_local(h - ph, s, c) / self.motion_scale
        do = to_local(o - po, s, c) / self.motion_scale
        q0, q2 = rotation_columns(raw[..., 7:11])
        p0, p2 = rotation_columns(raw[..., 25:29])
        pose = torch.cat((to_local(q0, s, c), to_local(q2, s, c),
                          to_local(p0, s, c), to_local(p2, s, c)), dim=-1)
        grip = torch.cat((raw[..., 3:4], raw[..., 21:22],
                          (raw[..., 3:4] - raw[..., 21:22]) / 0.1), dim=-1)
        distance = torch.cat((torch.sqrt(((h - o) ** 2).sum(dim=-1, keepdim=True) + 1.0e-12) / self.contact_scale,
                              torch.sqrt(((g - o) ** 2).sum(dim=-1, keepdim=True) + 1.0e-12) / self.goal_scale), dim=-1)
        world = torch.cat((h / self.world_scale, g / self.world_scale,
                           torch.stack((s, c), dim=-1),
                           (h - o) / self.contact_scale, (g - o) / self.goal_scale), dim=-1)
        return torch.cat((hc, gc, pc, dh, do, grip, pose, distance, world), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        s, c = cabinet_basis(chunk_start_context["raw_current"])
        xyz = to_local(native_actions[..., :3], s[:, None], c[:, None])
        return torch.cat((xyz, native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        s, c = cabinet_basis(chunk_start_context["raw_current"])
        xyz = to_world(encoded_actions[..., :3], s[:, None], c[:, None])
        return torch.cat((xyz, encoded_actions[..., 3:4]), dim=-1)


def build_design(common_spec, config):
    return CabinetGeometry(common_spec, config)
