import torch
from experiment1.contracts import CandidateDesign


class MetricChainDesign(CandidateDesign):
    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(52)

    def pose_features(self, raw, base):
        cfg = self.design_config
        hand = raw[..., base:base + 3]
        aperture = raw[..., base + 3:base + 4]
        stick = raw[..., base + 4:base + 7]
        stick_q = raw[..., base + 7:base + 11]
        pushed = raw[..., base + 11:base + 14]
        pushed_q = raw[..., base + 14:base + 18]
        goal = raw[..., 36:39]
        origin = raw.new_tensor(cfg['world_origin_m'])
        q_origin = raw.new_tensor([0.0, 0.0, 0.0, 1.0])
        q_scale = raw.new_tensor(cfg['quaternion_scale'])
        return torch.cat([
            (hand - origin) / raw.new_tensor(cfg['world_scale_m']),
            aperture,
            (stick - hand) / raw.new_tensor(cfg['grasp_scale_m']),
            (pushed - stick) / raw.new_tensor(cfg['tool_scale_m']),
            (goal - pushed) / raw.new_tensor(cfg['goal_scale_m']),
            (stick_q - q_origin) / q_scale,
            (pushed_q - q_origin) / q_scale
        ], dim=-1)

    def condition(self, causal_history, causal_context):
        raw = causal_context['raw_history']
        step_scale = self.design_config['step_displacement_scale_m']
        changes = torch.cat([
            (raw[..., 0:3] - raw[..., 18:21]) / step_scale,
            (raw[..., 4:7] - raw[..., 22:25]) / step_scale,
            (raw[..., 11:14] - raw[..., 29:32]) / step_scale,
            raw[..., 3:4] - raw[..., 21:22]
        ], dim=-1)
        return torch.cat([self.pose_features(raw, 0),
                          self.pose_features(raw, 18), changes], dim=-1)


def build_design(common_spec, config):
    return MetricChainDesign(common_spec, config)
