import torch
from experiment1.contracts import CandidateDesign


def body_axes(q):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-6)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack((1 - 2 * (y * y + z * z),
                        2 * (x * y + w * z),
                        2 * (x * z - w * y),
                        2 * (x * y - w * z),
                        1 - 2 * (x * x + z * z),
                        2 * (y * z + w * x)), dim=-1)


class MetricRelations(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.length_scale = float(config['length_scale_m'])
        self.motion_scale = float(config['motion_scale_m'])
        self.workspace_scale = float(config['workspace_scale_m'])
        self.workspace_origin = list(config['workspace_origin_m'])

    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(43)

    def condition(self, causal_history, causal_context):
        raw = causal_context['raw_history']
        hand = raw[..., 0:3]
        site = raw[..., 4:7]
        old_hand = raw[..., 18:21]
        old_site = raw[..., 22:25]
        goal = raw[..., 36:39]
        ring = raw[..., 39:42]
        grasp = site - hand
        align = goal - ring
        ring_motion = torch.cat((torch.zeros_like(ring[:, :1]),
                                 ring[:, 1:] - ring[:, :-1]), dim=1)
        axes = body_axes(raw[..., 7:11])
        old_axes = body_axes(raw[..., 25:29])
        origin = raw.new_tensor(self.workspace_origin)
        heights = torch.cat((hand[..., 2:3], ring[..., 2:3], goal[..., 2:3]), dim=-1)
        ranges = torch.cat((torch.linalg.vector_norm(grasp[..., :2], dim=-1, keepdim=True),
                            torch.linalg.vector_norm(align[..., :2], dim=-1, keepdim=True)), dim=-1)
        return torch.cat((grasp / self.length_scale,
                          align / self.length_scale,
                          (site - ring) / self.length_scale,
                          (old_site - old_hand) / self.length_scale,
                          (hand - old_hand) / self.motion_scale,
                          (site - old_site) / self.motion_scale,
                          ring_motion / self.motion_scale,
                          axes,
                          (axes - old_axes) / 0.1,
                          (raw[..., 3:4] - 0.5) * 2.0,
                          (raw[..., 3:4] - raw[..., 21:22]) / 0.1,
                          (hand - origin) / self.workspace_scale,
                          heights / self.length_scale,
                          ranges / self.length_scale), dim=-1)


def build_design(common_spec, config):
    return MetricRelations(common_spec, config)
