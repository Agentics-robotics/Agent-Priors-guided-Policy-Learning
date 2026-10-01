import torch
from experiment1.contracts import CandidateDesign


class WorldRelations(CandidateDesign):
    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(72)

    def face_coordinates(self, point, wall, half):
        scale = self.design_config['position_scale_m']
        delta = point - wall
        return torch.cat(((delta + half) / scale, (half - delta) / scale), dim=-1)

    def condition(self, causal_history, causal_context):
        r = causal_context['raw_history']
        h, o, g = r[..., 0:3], r[..., 4:7], r[..., 36:39]
        hp, op = r[..., 18:21], r[..., 22:25]
        w, half = r[..., 39:42], r[..., 42:45]
        q, qp = r[..., 7:11], r[..., 25:29]
        s = self.design_config['position_scale_m']
        v = self.design_config['motion_scale_m']
        features = [
            (h - w) / s, (o - h) / s, (g - o) / s,
            (g - h) / s, (o - w) / s, (g - w) / s,
            h[..., 2:3] / s, o[..., 2:3] / s, g[..., 2:3] / s,
            w / self.design_config['world_scale_m'], half / s,
            2.0 * r[..., 3:4] - 1.0, q, qp,
            (h - hp) / v, (o - op) / v,
            (r[..., 3:4] - r[..., 21:22]) * self.design_config['aperture_motion_scale'],
            ((o - h) - (op - hp)) / v,
            hp[..., 2:3] / s, op[..., 2:3] / s,
            self.face_coordinates(h, w, half),
            self.face_coordinates(o, w, half),
            self.face_coordinates(g, w, half),
            q - qp,
            torch.linalg.vector_norm(o - h, dim=-1, keepdim=True) / s,
            torch.linalg.vector_norm(g - o, dim=-1, keepdim=True) / s
        ]
        return torch.cat(features, dim=-1)


def build_design(common_spec, config):
    return WorldRelations(common_spec, config)
