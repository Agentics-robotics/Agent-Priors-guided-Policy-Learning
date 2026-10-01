import numpy as np
import torch
from experiment1.contracts import CandidateDesign


class MetricRelations(CandidateDesign):
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
        self.dp = common_dp_factory(65)

    def quaternion_features(self, q):
        # Canonical sign removes only the quaternion double cover, not orientation.
        q = q / torch.sqrt((q * q).sum(dim=-1, keepdim=True).clamp_min(1.0e-12))
        q = torch.where(q[..., 3:4] < 0.0, -q, q)
        return torch.cat((q[..., :3] / self.design_config['rotation_scale'], q[..., 3:4]), dim=-1)

    def pose_features(self, raw, base, origin):
        h = raw[..., base:base + 3]
        s = raw[..., base + 4:base + 7]
        p = raw[..., base + 11:base + 14]
        g = raw[..., 36:39]
        return torch.cat(((s - h) / self.design_config['grasp_scale_m'],
                          (p - s) / self.design_config['scene_scale_m'],
                          (g - p) / self.design_config['scene_scale_m'],
                          (p - h) / self.design_config['scene_scale_m'],
                          (s - origin) / self.design_config['scene_scale_m'],
                          raw[..., base + 3:base + 4],
                          self.quaternion_features(raw[..., base + 7:base + 11]),
                          self.quaternion_features(raw[..., base + 14:base + 18])), dim=-1)

    def condition(self, causal_history, causal_context):
        r = causal_context['raw_history']
        origin = torch.as_tensor(self.world_origin, dtype=r.dtype, device=r.device)
        h, s, p, g = r[..., 0:3], r[..., 4:7], r[..., 11:14], r[..., 36:39]
        motion = torch.cat((h - r[..., 18:21], s - r[..., 22:25], p - r[..., 29:32]), dim=-1)
        heights = torch.cat((h[..., 2:3], s[..., 2:3], p[..., 2:3], g[..., 2:3]), dim=-1)
        ranges = torch.cat((torch.linalg.vector_norm(s - h, dim=-1, keepdim=True),
                            torch.linalg.vector_norm(p - s, dim=-1, keepdim=True),
                            torch.linalg.vector_norm(g - p, dim=-1, keepdim=True)), dim=-1)
        return torch.cat((self.pose_features(r, 0, origin), self.pose_features(r, 18, origin),
                          motion / self.design_config['motion_scale_m'],
                          (r[..., 3:4] - r[..., 21:22]) / self.design_config['aperture_change_scale'],
                          heights / self.design_config['height_scale_m'],
                          ranges / self.design_config['scene_scale_m']), dim=-1)


def build_design(common_spec, config):
    return MetricRelations(common_spec, config)
