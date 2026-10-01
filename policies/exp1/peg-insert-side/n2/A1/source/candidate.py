import torch
from experiment1.contracts import CandidateDesign


class MetricRelations(CandidateDesign):
    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(44)

    def condition(self, causal_history, causal_context):
        r = causal_context['raw_history']
        h = r[..., 0:3]
        p = r[..., 4:7]
        tip = r[..., 39:42]
        goal = r[..., 36:39]
        previous_h = r[..., 18:21]
        previous_p = r[..., 22:25]
        metric = self.design_config['metric_scale_m']
        near = self.design_config['near_scale_m']
        step = self.common_spec['action_schema']['native_xyz_scale_m']
        origin = h.new_tensor(self.design_config['workspace_origin_m'])
        workspace = (h - origin) / self.design_config['workspace_scale_m']
        heights = torch.cat((h[..., 2:3], p[..., 2:3], tip[..., 2:3], goal[..., 2:3]), dim=-1) / metric
        tip_step = torch.cat((torch.zeros_like(tip[:, :1]), tip[:, 1:] - tip[:, :1]), dim=1) / step
        # All geometric errors are signed world-axis vectors. Saturating near
        # features supplement, never replace, the unbounded metric features.
        return torch.cat(((p - h) / metric,
                          (tip - h) / metric,
                          (goal - tip) / metric,
                          (tip - p) / metric,
                          torch.tanh((p - h) / near),
                          torch.tanh((goal - tip) / near),
                          workspace, heights,
                          r[..., 3:4], r[..., 7:11],
                          r[..., 7:11] - r[..., 25:29],
                          (h - previous_h) / step,
                          (p - previous_p) / step,
                          r[..., 3:4] - r[..., 21:22],
                          tip_step), dim=-1)


def build_design(common_spec, config):
    return MetricRelations(common_spec, config)
