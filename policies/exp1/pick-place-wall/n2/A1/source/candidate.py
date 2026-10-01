import torch
from experiment1.contracts import CandidateDesign


class WorldRelations(CandidateDesign):
    def features(self, raw):
        h = raw[..., 0:3]
        o = raw[..., 4:7]
        hp = raw[..., 18:21]
        op = raw[..., 22:25]
        g = raw[..., 36:39]
        w = raw[..., 39:42]
        s = raw[..., 42:45]
        grasp_scale = self.design_config['grasp_scale_m']
        task_scale = self.design_config['task_scale_m']
        wall_scale = self.design_config['wall_scale_m']
        motion_scale = self.common_spec['action_schema']['native_xyz_scale_m']
        top = w[..., 2:3] + s[..., 2:3]
        contact_xy = torch.sqrt(((o[..., :2] - h[..., :2]) ** 2).sum(dim=-1, keepdim=True) + 1.0e-12)
        travel_xy = torch.sqrt(((g[..., :2] - o[..., :2]) ** 2).sum(dim=-1, keepdim=True) + 1.0e-12)
        return torch.cat([
            (o - h) / grasp_scale,
            (g - o) / task_scale,
            (g - h) / task_scale,
            (h - w) / wall_scale,
            (o - w) / wall_scale,
            (g - w) / wall_scale,
            w,
            s / wall_scale,
            2.0 * (raw[..., 3:4] - 0.5),
            2.0 * (raw[..., 21:22] - 0.5),
            (h - hp) / motion_scale,
            (o - op) / motion_scale,
            raw[..., 7:11],
            raw[..., 25:29],
            (torch.abs(h - w) - s) / task_scale,
            (torch.abs(o - w) - s) / task_scale,
            (torch.abs(g - w) - s) / task_scale,
            (h[..., 2:3] - top) / task_scale,
            (o[..., 2:3] - top) / task_scale,
            (g[..., 2:3] - top) / task_scale,
            contact_xy / grasp_scale,
            travel_xy / task_scale
        ], dim=-1)

    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(54)

    def condition(self, causal_history, causal_context):
        return self.features(causal_context['raw_history'])


def build_design(common_spec, config):
    return WorldRelations(common_spec, config)
