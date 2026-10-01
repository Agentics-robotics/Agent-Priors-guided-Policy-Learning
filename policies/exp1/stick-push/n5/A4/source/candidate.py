import numpy as np
import torch
from experiment1.contracts import CandidateDesign


class CompactKinematicRelations(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.world_origin = [0.0, 0.0, 0.0]

    def fit_support(self, support_view, common_spec):
        origins = np.stack([np.asarray(e['obs'][0, 11:14], dtype=np.float64) for e in support_view])
        self.world_origin = origins.mean(axis=0).tolist()

    def deployment_state_dict(self):
        return {'world_origin': list(self.world_origin)}

    def load_deployment_state_dict(self, state):
        self.world_origin = list(state['world_origin'])

    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(39)

    def quaternion_features(self, q):
        q = q / torch.sqrt((q * q).sum(dim=-1, keepdim=True).clamp_min(1.0e-12))
        q = torch.where(q[..., 3:4] < 0.0, -q, q)
        return torch.cat((q[..., :3] / self.design_config['rotation_scale'], q[..., 3:4]), dim=-1)

    def condition(self, causal_history, causal_context):
        r = causal_context['raw_history']
        origin = torch.as_tensor(self.world_origin, dtype=r.dtype, device=r.device)
        h, stick, pushed, goal = r[..., 0:3], r[..., 4:7], r[..., 11:14], r[..., 36:39]
        qs = self.quaternion_features(r[..., 7:11])
        qp = self.quaternion_features(r[..., 14:18])
        qs_previous = self.quaternion_features(r[..., 25:29])
        qp_previous = self.quaternion_features(r[..., 32:36])
        # A minimal spatial tree plus a fixed world anchor: 21 coordinates.
        spatial = torch.cat(((stick - h) / self.design_config['grasp_scale_m'],
                             (pushed - stick) / self.design_config['scene_scale_m'],
                             (goal - pushed) / self.design_config['scene_scale_m'],
                             (stick - origin) / self.design_config['scene_scale_m'],
                             r[..., 3:4], qs, qp), dim=-1)
        # The native previous pose is represented by an invertible temporal change.
        temporal = torch.cat(((h - r[..., 18:21]) / self.design_config['motion_scale_m'],
                              (stick - r[..., 22:25]) / self.design_config['motion_scale_m'],
                              (pushed - r[..., 29:32]) / self.design_config['motion_scale_m'],
                              (r[..., 3:4] - r[..., 21:22]) / self.design_config['aperture_change_scale'],
                              qs - qs_previous, qp - qp_previous), dim=-1)
        return torch.cat((spatial, temporal), dim=-1)


def build_design(common_spec, config):
    return CompactKinematicRelations(common_spec, config)
