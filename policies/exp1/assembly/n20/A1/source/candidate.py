import torch
from experiment1.contracts import CandidateDesign


def rotation6(q):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1.0e-6)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack((1 - 2 * (y * y + z * z),
                        2 * (x * y + w * z),
                        2 * (x * z - w * y),
                        2 * (x * y - w * z),
                        1 - 2 * (x * x + z * z),
                        2 * (y * z + w * x)), dim=-1)


class TwoCenterMetric(CandidateDesign):
    def __init__(self, common_spec, config):
        super().__init__(common_spec, config)
        self.edge_scale = float(config['edge_scale_m'])
        self.fine_scale = float(config['fine_scale_m'])
        self.world_scale = float(config['world_scale_m'])
        self.motion_scale = float(config['motion_scale_m'])

    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(56)

    def condition(self, causal_history, causal_context):
        raw = causal_context['raw_history']
        hand, obj = raw[..., 0:3], raw[..., 4:7]
        phand, pobj = raw[..., 18:21], raw[..., 22:25]
        goal, ring = raw[..., 36:39], raw[..., 39:42]
        grasp = obj - hand
        place = goal - ring
        edges = torch.cat((grasp, place, ring - hand, obj - ring,
                           goal - hand, pobj - phand), dim=-1) / self.edge_scale
        fine_edges = torch.cat((grasp, place), dim=-1)
        fine = fine_edges / torch.sqrt(fine_edges.square() + self.fine_scale ** 2)
        origin = raw.new_tensor([0.0, 0.6, 0.0])
        world_hand = (hand - origin) / self.world_scale
        heights = torch.cat((hand[..., 2:3], obj[..., 2:3],
                             ring[..., 2:3], goal[..., 2:3]), dim=-1) / 0.2
        grips = torch.cat((raw[..., 3:4], raw[..., 21:22],
                           raw[..., 3:4] - raw[..., 21:22]), dim=-1)
        motion = torch.cat((hand - phand, obj - pobj), dim=-1) / self.motion_scale
        orientation = torch.cat((rotation6(raw[..., 7:11]),
                                 rotation6(raw[..., 25:29])), dim=-1)
        radial = torch.cat((torch.linalg.vector_norm(grasp[..., :2], dim=-1, keepdim=True),
                            grasp[..., 2:3],
                            torch.linalg.vector_norm(place[..., :2], dim=-1, keepdim=True),
                            place[..., 2:3]), dim=-1) / self.edge_scale
        return torch.cat((edges, fine, world_hand, heights, grips,
                          motion, orientation, radial), dim=-1)


def build_design(common_spec, config):
    return TwoCenterMetric(common_spec, config)
