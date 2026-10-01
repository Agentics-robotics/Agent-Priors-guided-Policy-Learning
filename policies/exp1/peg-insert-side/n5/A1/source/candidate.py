import torch
from experiment1.contracts import CandidateDesign


def length(x):
    return torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + 1.0e-12)


class MetricRelations(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        if common_spec['observation_schema']['raw_dim'] != 42:
            raise ValueError('This design requires the declared peg-head observation channel')

    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(51)

    def condition(self, causal_history, causal_context):
        r = causal_context['raw_history']
        hand = r[..., 0:3]
        peg = r[..., 4:7]
        hand_prev = r[..., 18:21]
        peg_prev = r[..., 22:25]
        goal = r[..., 36:39]
        head = r[..., 39:42]
        reach = peg - hand
        insert = goal - head
        shaft = head - peg
        reach_prev = peg_prev - hand_prev
        hand_delta = hand - hand_prev
        peg_delta = peg - peg_prev
        near = self.design_config['near_scale_m']
        relation = self.design_config['reach_scale_m']
        insertion = self.design_config['insertion_scale_m']
        step_scale = self.common_spec['action_schema']['native_xyz_scale_m']
        heights = torch.cat((hand[..., 2:3], peg[..., 2:3], head[..., 2:3], goal[..., 2:3]), dim=-1) / relation
        aperture = torch.cat((r[..., 3:4], r[..., 21:22], (r[..., 3:4] - r[..., 21:22]) / 0.1), dim=-1)
        # Retain robot/world placement at a coarse, physical scale, rather than
        # normalizing the narrow correlated layout coordinates by their sample std.
        origin = r.new_tensor([0.0, 0.6, 0.0])
        world = torch.cat(((hand - origin) / 0.5, (goal - origin) / 0.5), dim=-1)
        distances = torch.cat((length(reach[..., :2]) / relation,
                               reach[..., 2:3] / relation,
                               length(insert[..., 1:3]) / relation,
                               insert[..., 0:1] / insertion,
                               length(reach) / relation,
                               length(insert) / insertion), dim=-1)
        return torch.cat((reach / relation,
                          insert / insertion,
                          shaft / self.design_config['shaft_scale_m'],
                          reach_prev / relation,
                          hand_delta / step_scale,
                          peg_delta / step_scale,
                          reach / (near + torch.abs(reach)),
                          insert / (near + torch.abs(insert)),
                          heights, aperture,
                          r[..., 7:11], r[..., 25:29],
                          world, distances), dim=-1)


def build_design(common_spec, config):
    return MetricRelations(common_spec, config)
