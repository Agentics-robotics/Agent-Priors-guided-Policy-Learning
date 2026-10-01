import torch
from experiment1.contracts import CandidateDesign


def canonical_quaternion(q):
    return torch.where(q[..., 3:4] < 0.0, -q, q)


class WorldRelations(CandidateDesign):
    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(57)

    def condition(self, causal_history, causal_context):
        r = causal_context["raw_history"]
        h, o, g = r[..., 0:3], r[..., 4:7], r[..., 36:39]
        c, s = r[..., 39:42], r[..., 42:45]
        ph, po = r[..., 18:21], r[..., 22:25]
        lower, upper = c - s, c + s
        features = [
            (o - h) / 0.05,
            (g - o) / 0.2,
            (g - h) / 0.2,
            (h - c) / 0.2,
            (o - c) / 0.2,
            (g - c) / 0.2,
            c / 0.5,
            s / 0.1,
            (h - ph) / 0.01,
            (o - po) / 0.01,
            r[..., 3:4], r[..., 21:22],
            canonical_quaternion(r[..., 7:11]),
            canonical_quaternion(r[..., 25:29]),
            (h - lower) / 0.1, (upper - h) / 0.1,
            (o - lower) / 0.1, (upper - o) / 0.1,
            h[..., 2:3] / 0.2,
            o[..., 2:3] / 0.2,
            g[..., 2:3] / 0.2,
            torch.linalg.vector_norm(o - h, dim=-1, keepdim=True) / 0.05,
            torch.linalg.vector_norm(g - o, dim=-1, keepdim=True) / 0.2,
        ]
        return torch.cat(features, dim=-1)


def build_design(common_spec, config):
    return WorldRelations(common_spec, config)
