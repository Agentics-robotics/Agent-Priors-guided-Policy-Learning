import numpy as np
import torch
from experiment1.contracts import CandidateDesign


def yaw_basis(raw):
    s0, c0 = raw[..., 39], raw[..., 40]
    n = torch.sqrt(s0 * s0 + c0 * c0)
    d = n.clamp_min(1.0e-8)
    return (torch.where(n > 1.0e-8, s0 / d, torch.zeros_like(s0)),
            torch.where(n > 1.0e-8, c0 / d, torch.ones_like(c0)))


def local(v, s, c):
    return torch.stack((c * v[..., 0] + s * v[..., 1],
                        -s * v[..., 0] + c * v[..., 1], v[..., 2]), dim=-1)


def world(v, s, c):
    return torch.stack((c * v[..., 0] - s * v[..., 1],
                        s * v[..., 0] + c * v[..., 1], v[..., 2]), dim=-1)


def axes(q):
    q = q / torch.sqrt((q * q).sum(dim=-1, keepdim=True)).clamp_min(1.0e-8)
    x, y, z, w = q.unbind(dim=-1)
    a = torch.stack((1.0 - 2.0 * (y * y + z * z),
                     2.0 * (x * y + z * w), 2.0 * (x * z - y * w)), dim=-1)
    b = torch.stack((2.0 * (x * z + y * w), 2.0 * (y * z - x * w),
                     1.0 - 2.0 * (x * x + y * y)), dim=-1)
    return a, b


class ArticulatedPolar(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.geometry_state = {}
        self.latent_dim = int(config["phase_latent_dim"])
        self.hidden_dim = int(config["phase_hidden_dim"])
        self.align_radius = float(config["alignment_radius_m"])
        self.align_width = float(config["alignment_width_m"])
        self.contact_height = float(config["contact_height_m"])
        self.height_width = float(config["height_width_m"])

    def fit_support(self, support_view, common_spec):
        centers, radii, closed = [], [], []
        count = 0
        for episode in support_view:
            raw = np.asarray(episode["obs"][:-1], dtype=np.float64)
            v = raw[:, 4:7] - raw[:, 36:39]
            n = np.maximum(np.sqrt(raw[:, 39] ** 2 + raw[:, 40] ** 2), 1.0e-8)
            s, c = raw[:, 39] / n, raw[:, 40] / n
            p = np.stack((c * v[:, 0] + s * v[:, 1],
                          -s * v[:, 0] + c * v[:, 1]), axis=-1)
            # Algebraic circle fit in observed cabinet coordinates, relative to goal.
            matrix = np.concatenate((2.0 * p, np.ones((len(p), 1))), axis=-1)
            rhs = (p * p).sum(axis=-1)
            solution = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
            center = solution[:2]
            radius = float(np.sqrt(max(float(solution[2] + (center * center).sum()), 1.0e-8)))
            initial = p[0] - center
            initial = initial / max(float(np.sqrt((initial * initial).sum())), 1.0e-8)
            centers.append(center)
            radii.append(radius)
            closed.append(initial)
            count += len(raw)
        center = np.mean(np.stack(centers), axis=0)
        direction = np.mean(np.stack(closed), axis=0)
        direction = direction / max(float(np.sqrt((direction * direction).sum())), 1.0e-8)
        self.geometry_state = {
            "center_offset_m": [float(center[0]), float(center[1]), 0.0],
            "radius_m": float(np.mean(radii)),
            "closed_radial_xy": direction.tolist(),
            "support_episode_count": len(support_view),
            "support_observation_count": count,
            "fit_scope": "current D_N obs[0:T], separate planar circle fits, equal episode averaging"
        }

    def deployment_state_dict(self):
        return dict(self.geometry_state)

    def load_deployment_state_dict(self, state):
        self.geometry_state = dict(state)

    def build_modules(self, common_dp_factory):
        self.register_buffer("center_offset", torch.tensor(self.geometry_state["center_offset_m"], dtype=torch.float32))
        self.register_buffer("reference_radius", torch.tensor(self.geometry_state["radius_m"], dtype=torch.float32))
        self.register_buffer("closed_unit", torch.tensor(self.geometry_state["closed_radial_xy"], dtype=torch.float32))
        self.approach_encoder = torch.nn.Sequential(torch.nn.Linear(37, self.hidden_dim), torch.nn.SiLU(),
                                                   torch.nn.Linear(self.hidden_dim, self.latent_dim), torch.nn.SiLU())
        self.descent_encoder = torch.nn.Sequential(torch.nn.Linear(37, self.hidden_dim), torch.nn.SiLU(),
                                                  torch.nn.Linear(self.hidden_dim, self.latent_dim), torch.nn.SiLU())
        self.opening_encoder = torch.nn.Sequential(torch.nn.Linear(37, self.hidden_dim), torch.nn.SiLU(),
                                                  torch.nn.Linear(self.hidden_dim, self.latent_dim), torch.nn.SiLU())
        self.dp = common_dp_factory(56 + self.latent_dim)

    def polar_basis(self, raw):
        s, c = yaw_basis(raw)
        goal = raw[..., 36:39]
        offset = torch.zeros_like(goal) + self.center_offset
        center = goal + world(offset, s, c)
        radial = raw[..., 4:7] - center
        radius = torch.sqrt((radial[..., :2] ** 2).sum(dim=-1))
        safe = radius.clamp_min(1.0e-8)
        sr = torch.where(radius > 1.0e-8, radial[..., 1] / safe, s)
        cr = torch.where(radius > 1.0e-8, radial[..., 0] / safe, c)
        return sr, cr, s, c, center, radius

    def geometry(self, raw):
        sr, cr, s, c, center, radius = self.polar_basis(raw)
        h, o, g = raw[..., 0:3], raw[..., 4:7], raw[..., 36:39]
        ph, po = raw[..., 18:21], raw[..., 22:25]
        rel = h - o
        a, b = axes(raw[..., 7:11])
        pa, pb = axes(raw[..., 25:29])
        pose = torch.cat((local(a, sr, cr), local(b, sr, cr),
                          local(pa, sr, cr), local(pb, sr, cr)), dim=-1)
        grip = torch.cat((raw[..., 3:4], raw[..., 21:22],
                          (raw[..., 3:4] - raw[..., 21:22]) / 0.1), dim=-1)
        radial_x = c * cr + s * sr
        radial_y = -s * cr + c * sr
        cos_open = radial_x * self.closed_unit[0] + radial_y * self.closed_unit[1]
        sin_open = radial_y * self.closed_unit[0] - radial_x * self.closed_unit[1]
        goal_radial = local(g - center, sr, cr)[..., :2]
        goal_radial = goal_radial / torch.sqrt((goal_radial ** 2).sum(dim=-1, keepdim=True)).clamp_min(1.0e-8)
        distance = torch.cat((torch.sqrt((rel ** 2).sum(dim=-1, keepdim=True) + 1.0e-12) / 0.1,
                              torch.sqrt(((g - o) ** 2).sum(dim=-1, keepdim=True) + 1.0e-12) / 0.5), dim=-1)
        intrinsic = torch.cat((local(rel, sr, cr) / 0.1, local(ph - po, sr, cr) / 0.1,
                               local(g - o, sr, cr) / 0.5, local(h - ph, sr, cr) / 0.01,
                               local(o - po, sr, cr) / 0.01, pose, grip,
                               (radius / self.reference_radius.clamp_min(1.0e-6))[..., None],
                               torch.stack((cos_open, sin_open), dim=-1), goal_radial, distance), dim=-1)
        bypass = torch.cat((h / 0.5, g / 0.5, torch.stack((s, c, sr, cr), dim=-1),
                            local(o - g, s, c) / 0.5, local(rel, s, c) / 0.1), dim=-1)
        dxy = torch.sqrt((rel[..., :2] ** 2).sum(dim=-1) + 1.0e-12)
        aligned = torch.sigmoid((self.align_radius - dxy) / self.align_width)
        low = torch.sigmoid((self.contact_height - torch.abs(rel[..., 2])) / self.height_width)
        open_hint = 1.0 - torch.exp(-4.0 * torch.relu(1.0 - cos_open))
        pulling = aligned * low + (1.0 - aligned * low) * open_hint
        gates = torch.stack(((1.0 - pulling) * (1.0 - aligned),
                             (1.0 - pulling) * aligned, pulling), dim=-1)
        return intrinsic, bypass, gates

    def condition(self, causal_history, causal_context):
        intrinsic, bypass, gates = self.geometry(causal_context["raw_history"])
        phase = (gates[..., 0:1] * self.approach_encoder(intrinsic)
                 + gates[..., 1:2] * self.descent_encoder(intrinsic)
                 + gates[..., 2:3] * self.opening_encoder(intrinsic))
        return torch.cat((intrinsic, bypass, gates, phase), dim=-1)

    def encode_actions(self, native_actions, chunk_start_context):
        sr, cr, s, c, center, radius = self.polar_basis(chunk_start_context["raw_current"])
        xyz = local(native_actions[..., :3], sr[:, None], cr[:, None])
        return torch.cat((xyz, native_actions[..., 3:4]), dim=-1)

    def decode_actions(self, encoded_actions, chunk_start_context):
        sr, cr, s, c, center, radius = self.polar_basis(chunk_start_context["raw_current"])
        xyz = world(encoded_actions[..., :3], sr[:, None], cr[:, None])
        return torch.cat((xyz, encoded_actions[..., 3:4]), dim=-1)


def build_design(common_spec, config):
    return ArticulatedPolar(common_spec, config)
