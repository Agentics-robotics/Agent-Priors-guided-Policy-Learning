import torch
from experiment1.contracts import CandidateDesign


def cabinet_basis(raw):
    s0, c0 = raw[..., 39], raw[..., 40]
    n = torch.sqrt(s0 * s0 + c0 * c0)
    d = n.clamp_min(1.0e-8)
    s = torch.where(n > 1.0e-8, s0 / d, torch.zeros_like(s0))
    c = torch.where(n > 1.0e-8, c0 / d, torch.ones_like(c0))
    return s, c


def local(v, s, c):
    return torch.stack((c * v[..., 0] + s * v[..., 1],
                        -s * v[..., 0] + c * v[..., 1], v[..., 2]), dim=-1)


def world(v, s, c):
    return torch.stack((c * v[..., 0] - s * v[..., 1],
                        s * v[..., 0] + c * v[..., 1], v[..., 2]), dim=-1)


def rotation_columns(q):
    q = q / torch.sqrt((q * q).sum(dim=-1, keepdim=True)).clamp_min(1.0e-8)
    x, y, z, w = q.unbind(dim=-1)
    first = torch.stack((1.0 - 2.0 * (y * y + z * z),
                         2.0 * (x * y + z * w), 2.0 * (x * z - y * w)), dim=-1)
    third = torch.stack((2.0 * (x * z + y * w), 2.0 * (y * z - x * w),
                         1.0 - 2.0 * (x * x + y * y)), dim=-1)
    return torch.stack((first, third), dim=-2)


class CompactCabinetContact(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.contact_scale = float(config["contact_scale_m"])
        self.goal_scale = float(config["goal_scale_m"])
        self.motion_scale = float(config["motion_scale_m"])
        self.world_scale = float(config["world_scale_m"])

    def build_modules(self, common_dp_factory):
        self.register_buffer("vector_scales", torch.tensor(
            [self.contact_scale, self.goal_scale, self.motion_scale, self.motion_scale,
             1.0, 1.0, 1.0, 1.0], dtype=torch.float32)[:, None])
        self.register_buffer("distance_scales", torch.tensor(
            [self.contact_scale, self.goal_scale], dtype=torch.float32))
        self.dp = common_dp_factory(36)

    def condition(self, causal_history, causal_context):
        raw = causal_context["raw_history"]
        s, c = cabinet_basis(raw)
        h, o, g = raw[..., 0:3], raw[..., 4:7], raw[..., 36:39]
        ph, po = raw[..., 18:21], raw[..., 22:25]
        contact, goal = h - o, g - o
        displacement = torch.stack((contact, goal, h - ph, o - po), dim=-2)
        quaternion = torch.stack((raw[..., 7:11], raw[..., 25:29]), dim=-2)
        columns = rotation_columns(quaternion).flatten(start_dim=-3, end_dim=-2)
        vectors = torch.cat((displacement, columns), dim=-2)
        # One vectorized cabinet-frame transform; no feature-specific frame changes.
        intrinsic = (local(vectors, s[..., None], c[..., None]) / self.vector_scales).flatten(start_dim=-2)
        distances = torch.sqrt((displacement[..., :2, :] ** 2).sum(dim=-1) + 1.0e-12) / self.distance_scales
        aperture = torch.cat((raw[..., 3:4], raw[..., 21:22]), dim=-1)
        world_context = torch.cat((h / self.world_scale, g / self.world_scale,
                                   torch.stack((s, c), dim=-1)), dim=-1)
        return torch.cat((intrinsic, aperture, distances, world_context), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        s, c = cabinet_basis(chunk_start_context["raw_current"])
        xyz = local(native_actions[..., :3], s[:, None], c[:, None])
        return torch.cat((xyz, native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        s, c = cabinet_basis(chunk_start_context["raw_current"])
        xyz = world(encoded_actions[..., :3], s[:, None], c[:, None])
        return torch.cat((xyz, encoded_actions[..., 3:4]), dim=-1)


def build_design(common_spec, config):
    return CompactCabinetContact(common_spec, config)
