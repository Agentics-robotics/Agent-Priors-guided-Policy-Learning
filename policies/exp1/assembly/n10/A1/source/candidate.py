import torch
from experiment1.contracts import CandidateDesign


class MetricRelations(CandidateDesign):
    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(45)

    def condition(self, causal_history, causal_context):
        r = causal_context['raw_history']
        h = r[..., 0:3]
        o = r[..., 4:7]
        c = r[..., 39:42]
        g = r[..., 36:39]
        grasp = o - h
        seating = g - c
        offset = h.new_tensor(self.design_config['world_origin'])
        metric = self.design_config['relation_scale_m']
        fine = self.design_config['fine_scale_m']
        motion = self.common_spec['action_schema']['native_xyz_scale_m']
        grasp_xy_sq = grasp[..., 0:2].square().sum(-1, keepdim=True)
        seat_xy_sq = seating[..., 0:2].square().sum(-1, keepdim=True)
        distances = torch.cat((torch.sqrt(grasp_xy_sq + 1.0e-12),
                               torch.sqrt(seat_xy_sq + 1.0e-12)), dim=-1)
        proximity = torch.exp(-distances.square() / (fine * fine))
        seat_direction = seating / torch.sqrt(seating.square().sum(-1, keepdim=True) + fine * fine)
        return torch.cat((
            (h - offset) / self.design_config['world_scale_m'],
            2.0 * r[..., 3:4] - 1.0,
            grasp / metric,
            (c - o) / metric,
            seating / metric,
            r[..., 7:11],
            (h - r[..., 18:21]) / motion,
            (o - r[..., 22:25]) / motion,
            r[..., 3:4] - r[..., 21:22],
            r[..., 25:29],
            torch.asinh(grasp / fine),
            torch.asinh(seating / fine),
            torch.cat((h[..., 2:3], o[..., 2:3], c[..., 2:3], g[..., 2:3]), dim=-1) / metric,
            distances / metric,
            proximity,
            seat_direction
        ), dim=-1)


def build_design(common_spec, config):
    return MetricRelations(common_spec, config)
