import torch
from experiment1.contracts import CandidateDesign


def canonical_quaternion(q):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1.0e-8)
    return torch.where(q[..., 0:1] < 0.0, -q, q)


class MetricRelations(CandidateDesign):
    def build_modules(self, common_dp_factory):
        self.dp = common_dp_factory(41)

    def condition(self, causal_history, causal_context):
        raw = causal_context['raw_history']
        hand = raw[..., 0:3]
        handle = raw[..., 4:7]
        ring = raw[..., 39:42]
        goal = raw[..., 36:39]
        grasp = handle - hand
        insertion = goal - ring
        distance_xy = torch.linalg.vector_norm(grasp[..., :2], dim=-1, keepdim=True)
        distance_3d = torch.linalg.vector_norm(grasp, dim=-1, keepdim=True)
        alignment_xy = torch.linalg.vector_norm(insertion[..., :2], dim=-1, keepdim=True)
        nearby = torch.exp(-(distance_xy / self.design_config['grasp_radius_m']).square())
        contact = nearby * torch.sigmoid((0.08 - grasp[..., 2:3].abs()) / 0.015)
        contact = contact * torch.sigmoid((0.8 - raw[..., 3:4]) / 0.1)
        raised = torch.sigmoid((ring[..., 2:3] - self.design_config['lift_midpoint_m']) / 0.02)
        aligned = torch.exp(-(alignment_xy / self.design_config['alignment_radius_m']).square())
        world_hand = torch.cat((hand[..., 0:1] / 0.3, (hand[..., 1:2] - 0.7) / 0.3, hand[..., 2:3] / 0.3), dim=-1)
        return torch.cat((
            grasp / 0.1,
            insertion / 0.1,
            (ring - handle) / 0.1,
            (goal - hand) / 0.1,
            world_hand,
            handle[..., 2:3] / 0.1, ring[..., 2:3] / 0.1, goal[..., 2:3] / 0.1,
            raw[..., 3:4],
            (hand - raw[..., 18:21]) / 0.01,
            (handle - raw[..., 22:25]) / 0.01,
            raw[..., 3:4] - raw[..., 21:22],
            canonical_quaternion(raw[..., 7:11]),
            canonical_quaternion(raw[..., 25:29]),
            distance_xy / 0.1, distance_3d / 0.1, alignment_xy / 0.1,
            nearby, contact, raised, aligned
        ), dim=-1)


def build_design(common_spec, config):
    return MetricRelations(common_spec, config)
